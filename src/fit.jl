"""
Estimate a generalized linear model with high dimensional categorical variables

### Arguments
* `df`: a Table
* `FormulaTerm`: A formula created using [`@formula`](@ref)
* `distribution`: A `Distribution`. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid distributions.
* `link`: A `Link` function. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid link functions.
* `CovarianceEstimator`: A method to compute the variance-covariance matrix
* `save::Union{Bool, Symbol} = false`: Should residuals and eventual estimated fixed effects saved in a dataframe? Use `save = :residuals` to only save residuals. Use `save = :fe` to only save fixed effects.
* `method::Symbol`: A symbol for the method. Default is :cpu. Alternatively,  :gpu requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`.
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 1000`: Maximum number of iterations
* `maxiter_center::Integer = 10000`: Maximum number of iterations for centering procedure.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `dev_tol::Real` : Tolerance level for the first stopping condition of the maximization routine.
* `rho_tol::Real` : Tolerance level for the stephalving in the maximization routine.
* `step_tol::Real` : Tolerance level that accounts for rounding errors inside the stephalving routine
* `center_tol::Real` : Tolerance level for the stopping condition of the centering algorithm. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.

### Examples
```julia
using GLM, RDatasets, Distributions, Random, GLFixedEffectModels

rng = MersenneTwister(1234)
df = dataset("datasets", "iris")
df.binary = 0.0
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesDummy = categorical(df.Species)

m = @formula binary ~ SepalWidth + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), GLM.LogitLink() , start = [0.2] )
```
"""
function nlreg(@nospecialize(df),
    @nospecialize(formula::FormulaTerm),
    distribution::Distribution,
    link::GLM.Link,
    @nospecialize(vcov::CovarianceEstimator = Vcov.simple());
    @nospecialize(weights::Union{Symbol, Nothing} = nothing),
    @nospecialize(subset::Union{AbstractVector, Nothing} = nothing),
    @nospecialize(start::Union{AbstractVector, Nothing} = nothing),
    maxiter_center::Integer = 10000,    # maximum number of iterations in pseudo-demeaning
    maxiter::Integer = 1000,           # maximum number of iterations
    contrasts::Dict = Dict{Symbol, Any}(),
    dof_add::Integer = 0,
    @nospecialize(save::Union{Bool, Symbol} = false),
    method::Symbol = :cpu,
    drop_singletons = true,
    double_precision::Bool = true,
    dev_tol::Real = 1.0e-8, # tolerance level for the first stopping condition of the maximization routine.
    rho_tol::Real = 1.0e-4, # tolerance level for the stephalving in the maximization routine.
    step_tol::Real = 1.0e-8, # tolerance level that accounts for rounding errors inside the stephalving routine
    center_tol::Real = double_precision ? 1e-8 : 1e-6, # tolerance level for the stopping condition of the centering algorithm.
    @nospecialize(vcovformula::Union{Symbol, Expr, Nothing} = nothing),
    @nospecialize(subsetformula::Union{Symbol, Expr, Nothing} = nothing),
    verbose::Bool = false # Print output on each iteration.
    )


    df = DataFrame(df; copycols = false)
    # to deprecate
    if vcovformula != nothing
        if (vcovformula == :simple) | (vcovformula == :(simple()))
            vcov = Vcov.Simple()
        elseif (vcovformula == :robust) | (vcovformula == :(robust()))
            vcov = Vcov.Robust()
        else
            vcov = Vcov.cluster(StatsModels.termvars(@eval(@formula(0 ~ $(vcovformula.args[2]))))...)
        end
    end
    if subsetformula != nothing
        subset = eval(evaluate_subset(df, subsetformula))
    end

    ##############################################################################
    ##
    ## Parse formula
    ##
    ##############################################################################

    formula_origin = formula
    if  !omitsintercept(formula) & !hasintercept(formula)
        formula = FormulaTerm(formula.lhs, InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_endo, formula_iv = parse_iv(formula)
    has_iv = formula_iv != nothing
    has_weights = weights != nothing
    if has_iv
        error("Instrumental variables are not allowed.")
    end
    if has_weights
        @warn "Weights are not implemented yet, will be ignored."
    end

    ##############################################################################
    ##
    ## Save keyword argument
    ##
    ##############################################################################
    if !(save isa Bool)
        if save ∉ (:residuals, :fe)
            throw("the save keyword argument must be a Bool or a Symbol equal to :residuals or :fe")
        end
    end
    save_residuals = (save == :residuals) | (save == true)

    ##############################################################################
    ##
    ## Construct new dataframe after removing missing values
    ##
    ##############################################################################

    # create a dataframe without missing values & negative weights
    vars = StatsModels.termvars(formula)
    all_vars = unique(vars)

    # TODO speedup: this takes 4.8k
    esample = completecases(df, all_vars)
    if has_weights
        esample .&= BitArray(!ismissing(x) & (x > 0) for x in df[!, weights])
    end
    esample .&= Vcov.completecases(df, vcov)
    if subset != nothing
        if length(subset) != size(df, 1)
            throw("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    fes, ids, formula = parse_fixedeffect(df, formula)
    has_fes = !isempty(fes)
    if !has_fes
        error("No fixed effects detected, exiting. Please use GLM.jl for GLM's without fixed effects.")
    end
    if has_fes
        if drop_singletons
            for fe in fes
                drop_singletons!(esample, fe)
            end
        end
    end
    save_fe = (save == :fe) | ((save == true) & has_fes)

    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")

    has_intercept = hasintercept(formula)
    has_fe_intercept = false
    if has_fes
        if any(fe.interaction isa Ones for fe in fes)
            has_fe_intercept = true
        end
    end

    # Compute data for std errors
    vcov_method = Vcov.materialize(view(df, esample, :), vcov)

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################
    exo_vars = unique(StatsModels.termvars(formula))
    # TODO speedup: 8.3k
    subdf = StatsModels.columntable(disallowmissing(view(df, esample, exo_vars)))
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), GLFixedEffectModel, has_fe_intercept)

    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    all(isfinite, Xexo) || throw("Some observations for the exogeneous variables are infinite")

    response_name, coef_names = coefnames(formula_schema)
    if !(coef_names isa Vector)
        coef_names = typeof(coef_names)[coef_names]
    end

    # Weights are currently not implemented
    # if has_weights
    #     weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
    # else
    #     weights = Weights(Ones{Float64}(sum(esample)))
    # end
    # all(isfinite, values(weights)) || throw("Weights are not finite")

    # compute tss now before potentially demeaning y
    tss_total = tss(y, has_intercept | has_fe_intercept, Weights(Ones{Float64}(sum(esample))))

    # used to compute tss even without save_fe
    if save_fe
        oldy = deepcopy(y)
        oldX = deepcopy(Xexo)
    end

    # construct fixed effects object and solver
    fes = FixedEffect[_subset(fe, esample) for fe in fes]
    weights = Weights(Ones{Float64}(sum(esample)))
    feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method})

    # some constants
    coeflength = length(coef_names) # TODO this should really be better

    if start != nothing
        (length(start) == coeflength) || error("Invalid length of `start` argument.")
        beta = start
    else
        beta = 0.1 .* ones(Float64, coeflength)
    end

    eta = Xexo * beta
    mu = GLM.linkinv.(Ref(link),eta)
    wt = ones(Float64, nobs, 1)
    dev = sum(devresid.(Ref(distribution), y, mu))
    nulldev = sum(devresid.(Ref(distribution), mean(y), mu))

    X = Xexo
    Xhat = Xexo
    crossx = Matrix{Float64}(undef, nobs, 0)
    residuals = y # just for initialization
    basecoef = BitArray{size(X,2)}

    # Stuff that we need in outside scope
    emp = Array{Float64,2}(undef,2,2)
    score = hessian = emp

    outer_iterations = 0
    outer_converged = false

    for i = 1:maxiter

        verbose && println("Iteration $(i)")

        # Compute IWLS weights and dependent variable
        mymueta = GLM.mueta.(Ref(link),eta)
        wtildesq = mymueta.*mymueta ./  GLM.glmvar.(Ref(distribution),mu)

        nu = (y - mu) ./ mymueta
        # make a copy of nu because it's being changed by solve_residuals!
        nu_orig = deepcopy(nu)

        # Update weights and FixedEffectSolver object
        weights = Weights(wtildesq)
        all(isfinite, values(weights)) || throw("Weights are not finite")
        sqrtw = sqrt.(values(weights))
        FixedEffects.update_weights!(feM, weights)

        # # Pseudo-demean variables
        iterations = Int[]
        convergeds = Bool[]
        nudemean, b, c = FixedEffects.solve_residuals!(nu, feM; maxiter = maxiter_center, tol = center_tol)
        append!(iterations, b)
        append!(convergeds, c)

        Xdemean, b, c = FixedEffects.solve_residuals!(Xexo, feM; maxiter = maxiter_center, tol = center_tol)
        append!(iterations, b)
        append!(convergeds, c)

        # to obtain the correct projections, we need to weigh the demeaned nu and X
        nudemean = sqrtw .* nudemean
        Xdemean = Diagonal(sqrtw) * Xdemean

        iterations = maximum(iterations)
        converged = all(convergeds)

        if converged == false
            @warn "Convergence of annihilation procedure not achieved in $(iterations) iterations; try increasing maxiter_center or decreasing center_tol."
        end

        basecolXexo = GLFixedEffectModels.basecol(Xdemean)
        Xexo2 = GLFixedEffectModels.getcols(Xdemean, basecolXexo)
        Xhat = Xexo2
        basecoef = basecolXexo
        crossx = cholesky!(Symmetric(Xhat' * Xhat))

        beta_update = crossx \ (Xhat' * nudemean)

        # # Update \eta
        eta_update = nu_orig - (nudemean - Xdemean * beta_update) ./ sqrtw

        verbose && println("Old dev: $dev")
        devold = dev
        rho = 1.0
        while true
            mu = GLM.linkinv.(Ref(link),eta + rho .* eta_update)
            dev = sum(GLM.devresid.(Ref(distribution), y, mu))
            verbose && println("dev = $(dev)")
            if !isinf(dev) && dev <= devold
                eta = eta + rho .* eta_update
                beta = beta + rho .* beta_update
                verbose && println("beta = $(beta)")
                residuals = y - eta
                break
            end

            rho = rho / 2.0
            if rho < rho_tol
                error("Backtracking failed.")
            end
        end

        if ((devold - dev)/dev < dev_tol)
            verbose && println("Iter $i : converged (deviance)")
            outer_converged = true
        end
        if (norm(beta_update) < step_tol )
            verbose && println("Iter $i : converged (step size)")
            outer_converged = true
        end

        if outer_converged
            # # Compute concentrated Score and Hessian
            score = Xdemean .* nudemean
            hessian = Symmetric(Xdemean' * Xdemean)
            outer_iterations = i
            break
        else
            verbose && println("Iter $i : not converged")
            verbose && println("---------------------------------")
        end

        if i == maxiter
            @warn "Convergence not achieved in $(i) iterations; try increasing maxiter or dev_tol."
            outer_iterations = maxiter
        end

    end

    coef = beta

    ##############################################################################
    ##
    ## Optionally save objects in a new dataframe
    ##
    ##############################################################################

    augmentdf = DataFrame()
    if save_residuals
        if nobs < length(esample)
            augmentdf.residuals = Vector{Union{Float64, Missing}}(missing, length(esample))
            augmentdf[esample, :residuals] = residuals
        else
            augmentdf[!, :residuals] = residuals
        end
    end
    if save_fe
        oldX = getcols(oldX, basecoef)
        # update FixedEffectSolver
        weights = Weights(Ones{Float64}(sum(esample)))
        feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method})
        newfes, b, c = solve_coefficients!(eta - oldX * coef, feM; tol = center_tol, maxiter = maxiter_center)
        for j in 1:length(fes)
            if nobs < length(esample)
                augmentdf[!, ids[j]] = Vector{Union{Float64, Missing}}(missing, length(esample))
                augmentdf[esample, ids[j]] = newfes[j]
            else
                augmentdf[!, ids[j]] = newfes[j]
            end
        end
    end

    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################

    # Compute degrees of freedom
    dof_absorb = 0
    if has_fes
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if (vcov isa Vcov.ClusterCovariance) && any(isnested(fe, v.refs) for v in values(vcov_method.clusters))
                dof_absorb += 1 # if fe is nested you still lose 1 degree of freedom
            else
                #only count groups that exists
                dof_absorb += nunique(fe)
            end
        end
    end
    _n_coefs = size(X, 2) + dof_absorb + dof_add
    dof_residual_ = max(1, nobs - _n_coefs)

    nclusters = nothing
    if vcov isa Vcov.ClusterCovariance
        nclusters = map(x -> length(levels(x)), vcov_method.clusters)
    end

    vcov_data = VcovData(Xhat, crossx, residuals, dof_residual_, score, hessian)

    # Compute standard error
    matrix_vcov = StatsBase.vcov(vcov_data, vcov_method)

    ##############################################################################
    ##
    ## Return regression result
    ##
    ##############################################################################

    # add omitted variables
    if !all(basecoef)
        newcoef = zeros(length(basecoef))
        newmatrix_vcov = fill(NaN, (length(basecoef), length(basecoef)))
        newindex = [searchsortedfirst(cumsum(basecoef), i) for i in 1:length(coef)]
        for i in eachindex(newindex)
            newcoef[newindex[i]] = coef[i]
            for j in eachindex(newindex)
                newmatrix_vcov[newindex[i], newindex[j]] = matrix_vcov[i, j]
            end
        end
        coef = newcoef
        matrix_vcov = newmatrix_vcov
    end

    return GLFixedEffectModel(coef, # Vector of coefficients
        matrix_vcov,    # Covariance matrix
        vcov,           # CovarianceEstimator
        nclusters,
        outer_iterations,
        outer_converged,
        esample,
        augmentdf,
        distribution,
        link,
        coef_names,
        response_name,
        formula_origin, # Original formula
        formula_schema,
        nobs,   # Number of observations
        dof_residual_,  # nobs - degrees of freedoms
        dev, # Deviance of the fitted model
        nulldev, # null deviance
        score,   # concentrated gradient
        hessian  # concentrated hessian
    )

end
