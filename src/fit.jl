"""
Estimate a generalized linear model with high dimensional categorical variables

### Arguments
* `df`: a Table
* `FormulaTerm`: A formula created using [`@formula`](@ref)
* `distribution`: A `Distribution`. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid distributions.
* `link`: A `Link` function. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid link functions.
* `CovarianceEstimator`: A method to compute the variance-covariance matrix
* `save::Vector{Symbol} = Symbol[]`: Should residuals/predictions/eta's/estimated fixed effects be saved in the dataframe `augmentdf`? Can contain any subset of `[:residuals,:eta,:mu,:fe]`.
* `method::Symbol`: A symbol for the method. Default is :cpu. Alternatively,  :gpu requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`.
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 1000`: Maximum number of iterations
* `maxiter_center::Integer = 10000`: Maximum number of iterations for centering procedure.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `dev_tol::Real` : Tolerance level for the first stopping condition of the maximization routine.
* `rho_tol::Real` : Tolerance level for the stephalving in the maximization routine.
* `step_tol::Real` : Tolerance level that accounts for rounding errors inside the stephalving routine
* `center_tol::Real` : Tolerance level for the stopping condition of the centering algorithm. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `separation::Symbol = :none` : method to detect/deal with separation. Currently supported values are `:none`, `:ignore` and `:mu`. See readme for details.
* `separation_mu_lbound::Real = -Inf` : Lower bound for the Clarkson-Jennrich separation detection heuristic.
* `separation_mu_ubound::Real = Inf` : Upper bound for the Clarkson-Jennrich separation detection heuristic.
* `separation_ReLU_tol::Real = 1e-4` : Tolerance level for the ReLU algorithm.
* `separation_ReLU_maxiter::Integer = 1000` : Maximal number of iterations for the ReLU algorithm.

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
    save::Vector{Symbol} = Symbol[],
    method::Symbol = :cpu,
    drop_singletons = true,
    double_precision::Bool = true,
    dev_tol::Real = 1.0e-8, # tolerance level for the first stopping condition of the maximization routine.
    rho_tol::Real = 1.0e-8, # tolerance level for the stephalving in the maximization routine.
    step_tol::Real = 1.0e-8, # tolerance level that accounts for rounding errors inside the stephalving routine
    center_tol::Real = double_precision ? 1e-8 : 1e-6, # tolerance level for the stopping condition of the centering algorithm.
    separation::Vector{Symbol} = Symbol[], # method to detect and/or deal with separation
    separation_mu_lbound::Real = -Inf,
    separation_mu_ubound::Real = Inf,
    separation_ReLU_tol::Real = 1e-4,
    separation_ReLU_maxiter::Integer = 100,
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
    if  !StatsModels.omitsintercept(formula) & !StatsModels.hasintercept(formula)
        formula = StatsModels.FormulaTerm(formula.lhs, StatsModels.InterceptTerm{true}() + formula.rhs)
    end
    formula, formula_endo, formula_iv = FixedEffectModels.parse_iv(formula)
    has_iv = formula_iv != StatsModels.FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    has_weights = weights !== nothing
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
    save_residuals = (:residuals ∈ save)

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
    # if has_weights
    #     esample .&= BitArray(!ismissing(x) & (x > 0) for x in df[!, weights])
    # end
    if subset != nothing
        if length(subset) != size(df, 1)
            throw("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    esample .&= Vcov.completecases(df, vcov)

    formula, formula_fes = FixedEffectModels.parse_fe(formula)
    has_fes = formula_fes != FormulaTerm(ConstantTerm(0), ConstantTerm(0))
    fes, ids, fekeys = FixedEffectModels.parse_fixedeffect(df, formula_fes)

    has_fe_intercept = any(fe.interaction isa UnitWeights for fe in fes)

    # remove intercept if absorbed by fixed effects
    if has_fe_intercept
        formula = FormulaTerm(formula.lhs, tuple(InterceptTerm{false}(), (term for term in FixedEffectModels.eachterm(formula.rhs) if !isa(term, Union{ConstantTerm,InterceptTerm}))...))
    end
    has_intercept = hasintercept(formula)

    if has_fes
        if drop_singletons
            before_n = sum(esample)
            for fe in fes
                FixedEffectModels.drop_singletons!(esample, fe)
            end
            after_n = sum(esample)
            dropped_n = before_n - after_n
            if dropped_n > 0
                @info "$(dropped_n) observations detected as singletons. Dropping them ..."
            end
        end
    end

    save_fe = (:fe ∈ save) & has_fes 

    ##############################################################################
    ##
    ## Dataframe --> Matrix
    ##
    ##############################################################################
    exo_vars = unique(StatsModels.termvars(formula))
    subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in exo_vars)...))
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), GLFixedEffectModel, has_fe_intercept)

    # Obtain y
    # for a Vector{Float64}, conver(Vector{Float64}, y) aliases y
    y = convert(Vector{Float64}, response(formula_schema, subdf))
    oldy = deepcopy(response(formula_schema, df))
    #y = y[esample]
    all(isfinite, y) || throw("Some observations for the dependent variable are infinite")

    # Obtain X
    Xexo = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
    oldX = deepcopy(Xexo)
    #Xexo = Xexo[esample,:]
    all(isfinite, Xexo) || throw("Some observations for the exogeneous variables are infinite")

    basecoef = trues(size(Xexo,2)) # basecoef contains the information of the dropping of the regressors.
    # while esample contains the information of the dropping of the observations.

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

    ########################################################################
    ##
    ## Presolve:
    ## Step 1. Check Multicollinearity among X.
    ## Step 2. Check Separation
    ##
    ########################################################################

    # construct fixed effects object and solver
    fes = FixedEffect[_subset(fe, esample) for fe in fes]

    # pre separation detection check for collinearity
    Xexo, basecoef = detect_linear_dependency_among_X!(Xexo, basecoef; coefnames=coef_names)

    #####################################################################
    ##
    ## checking separation is basically looking for 
    #####################################################################
    if :simplex ∈ separation
        @warn "simplex not implemented, will ignore."
    end
    
    if link isa LogLink
        if :fe ∈ separation
            esample, y, Xexo, fes = detect_sep_fe!(esample, y, Xexo, fes; sep_at = 0)
        end
        if :ReLU ∈ separation
            esample, y, Xexo, fes = detect_sep_relu!(esample, y, Xexo, fes; 
                double_precision = double_precision, 
                dtol = center_tol, dmaxiter = maxiter,
                rtol = separation_ReLU_tol, rmaxiter = separation_ReLU_maxiter,
                method = method, verbose = verbose, 
                )
        end
    elseif link isa Union{ProbitLink, LogitLink}
        @assert all(0 .<= y .<= 1) "Dependent variable is not in the domain of the link function."
        if :fe ∈ separation
            esample, y, Xexo, fes = detect_sep_fe!(esample, y, Xexo, fes; sep_at = 0)
            esample, y, Xexo, fes = detect_sep_fe!(esample, y, Xexo, fes; sep_at = 1)
        end
        if :ReLU ∈ separation
            @warn "ReLU separation detection for ProbitLink/LogitLink is expermental, please interpret with caution."
            esample, y, Xexo, fes = detect_sep_relu!(esample, y, Xexo, fes;
                double_precision = double_precision, 
                dtol = center_tol, dmaxiter = maxiter,
                rtol = separation_ReLU_tol, rmaxiter = separation_ReLU_maxiter,
                method = method, verbose = verbose, 
                )
            esample, y, Xexo, fes = detect_sep_relu!(esample, 1 .- y[:], Xexo, fes;
                double_precision = double_precision, 
                dtol = center_tol, dmaxiter = maxiter,
                rtol = separation_ReLU_tol, rmaxiter = separation_ReLU_maxiter,
                method = method, verbose = verbose, 
                )
            y = 1 .- y
        end
    else
        @warn "Link function type $(typeof(link)) not support for ReLU separation detection. Skip separation detection."
    end
    
    # post separation detection check for collinearity
    Xexo, basecoef = detect_linear_dependency_among_X!(Xexo, basecoef; coefnames=coef_names)

    weights = Weights(Ones{Float64}(sum(esample)))
    feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method})

    # make one copy after deleting NAs + dropping singletons + detecting separations (fe + relu)
    nobs = sum(esample)
    (nobs > 0) || throw("sample is empty")

    # compute tss now before potentially demeaning y
    tss_total = FixedEffectModels.tss(y, has_intercept | has_fe_intercept, weights)

    # Compute data for std errors
    vcov_method = Vcov.materialize(view(df, esample, :), vcov) # is earlier in fixedeffectmodels

    # mark this as the start of a rerun when collinearity among X and fe is detected, rerun from here.
    @label rerun

    coeflength = sum(basecoef)
    if start !== nothing
        (length(start) == coeflength) || error("Invalid length of `start` argument.")
        beta = start
    else
        beta = 0.1 .* ones(Float64, coeflength)
    end

    #Xexo = oldX[esample,:]
    Xexo = GLFixedEffectModels.getcols(Xexo, basecoef) # get Xexo from oldX and basecoef and esample

    eta = Xexo * beta
    mu = GLM.linkinv.(Ref(link),eta)
    wt = ones(Float64, nobs, 1)
    dev = sum(devresid.(Ref(distribution), y, mu))
    nulldev = sum(devresid.(Ref(distribution), mean(y), mu))

    Xhat = Xexo
    crossx = Matrix{Float64}(undef, nobs, 0)
    residuals = y[:] # just for initialization

    # Stuff that we need in outside scope
    emp = Array{Float64,2}(undef,2,2)
    score = hessian = emp

    outer_iterations = 0
    outer_converged = false

    for i = 1:maxiter

        verbose && println("Iteration $(i)")

        # Compute IWLS weights and dependent variable
        mymueta = GLM.mueta.(Ref(link),eta)

        # Check for separation
        # use the bounds to detect 
        min_mueta = minimum(mymueta)
        max_mueta = maximum(mymueta)
        min_mu = minimum(mu)
        max_mu = maximum(mu)
        if (min_mueta < separation_mu_lbound) | (max_mueta > separation_mu_ubound) | (min_mu < separation_mu_lbound) | (max_mu > separation_mu_ubound)
            problematic = ((mymueta .< separation_mu_lbound) .| (mymueta .> separation_mu_ubound) .| (mu .< separation_mu_lbound) .| (mu .> separation_mu_ubound))
            @warn "$(sum(problematic)) observation(s) exceed the lower or upper bounds. Likely reason is statistical separation."
            # deal with it
            if :mu ∈ separation
                mymueta[mymueta .< separation_mu_lbound] .= separation_mu_lbound 
                mymueta[mymueta .> separation_mu_ubound] .= separation_mu_ubound
                mu[mu .< separation_mu_lbound] .= separation_mu_lbound 
                mu[mu .> separation_mu_ubound] .= separation_mu_ubound
            end
            # The following would remove the observations that are outside of the bounds, and restarts the estimation.
            # Inefficient.
            # if separation == :restart
            #     df_new = df[setdiff(1:size(df,1),indices),:]
            #     println("Separation detected. Restarting...")
            #     return nlreg(df_new,formula_origin,distribution,link,vcov,
            #         weights=nothing,subset=subset,start=beta,maxiter_center=maxiter_center, maxiter=maxiter, 
            #         contrasts=contrasts,dof_add=dof_add,save=save,
            #         method=method,drop_singletons=drop_singletons,double_precision=double_precision,
            #         dev_tol=dev_tol, rho_tol=rho_tol, step_tol=step_tol, center_tol=center_tol, 
            #         vcovformula=vcovformula,subsetformula=subsetformula,verbose=verbose)
            # end
        end

        wtildesq = mymueta.*mymueta ./  GLM.glmvar.(Ref(distribution),mu)

        nu = (y - mu) ./ mymueta
        # make a copy of nu because it's being changed by solve_residuals!
        nu_orig = deepcopy(nu)

        # Update weights and FixedEffectSolver object
        weights = Weights(wtildesq)
        all(isfinite, weights) || throw("IWLS Weights are not finite. Possible reason is separation.")
        sqrtw = sqrt.(weights)
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
        if all(basecolXexo)
        else
            remaining_cols = findall(basecoef)
            regressor_ind_to_be_dropped = remaining_cols[.~basecolXexo]
            basecoef[regressor_ind_to_be_dropped] .= 0 # update basecoef

            # throw info
            @info "Multicollinearity detected among columns of X and FixedEffects. Dropping regressors: $(join(coef_names[regressor_ind_to_be_dropped]," "))"

            @goto rerun
        end

        Xexo2 = GLFixedEffectModels.getcols(Xdemean, basecolXexo)
        Xhat = Xexo2
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
                residuals = y - mu
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
            if verbose
                println("Xdemean")
                display(Xdemean .* nudemean)
                display(Xhat .* nu)
            end
            break
        else
            verbose && println("Iter $i : not converged. Δdev = $((devold - dev)/dev), ||Δβ|| = $(norm(beta_update))")
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
        oldX = oldX[esample,:]
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
    if :mu ∈ save 
        if nobs < length(esample)
            augmentdf.mu = Vector{Union{Float64, Missing}}(missing, length(esample))
            augmentdf[esample, :mu] = mu
        else
            augmentdf[!, :mu] = mu
        end
    end
    if :eta ∈ save 
        if nobs < length(esample)
            augmentdf.eta = Vector{Union{Float64, Missing}}(missing, length(esample))
            augmentdf[esample, :eta] = eta
        else
            augmentdf[!, :eta] = eta
        end
    end

    ##############################################################################
    ##
    ## Test Statistics
    ##
    ##############################################################################

    # Compute degrees of freedom
    dof_absorb = 0
    dof_coef_and_fe = sum(basecoef) + dof_add - 1 # -1 for the constant
    if has_fes
        for fe in fes
            # adjust degree of freedom only if fe is not fully nested in a cluster variable:
            if (vcov isa Vcov.ClusterCovariance) && any(FixedEffectModels.isnested(fe, v.groups) for v in values(vcov_method.clusters))
                dof_absorb += 1 # if fe is nested you still lose 1 degree of freedom
            else
                #only count groups that exists
                dof_absorb += FixedEffectModels.nunique(fe)
            end
            dof_coef_and_fe = dof_coef_and_fe + FixedEffectModels.nunique(fe)
        end
    end
    _n_coefs = sum(basecoef) + dof_absorb + dof_add
    dof_residual_ = max(1, nobs - _n_coefs)

    nclusters = nothing
    if vcov isa Vcov.ClusterCovariance
        nclusters = Vcov.nclusters(vcov_method)
    end
    resid_vcov = if size(score, 2) >= 1
        score[:, 1] ./ Xhat[:, 1]
    else
        residuals
    end

    vcov_data = VcovDataGLM(Xhat, crossx, inv(crossx), resid_vcov, dof_residual_)#, hessian)
    # hessian is unnecessary since in all cases vcov takes the inv(cholesky(hessian)) which is the same as inv(crossx)
    """
    This option works if purely using Vcov.jl:
    if vcov isa Vcov.ClusterCovariance
        vcov_data = Vcov.VcovData(Xhat, crossx, score[:, 1] ./ Xhat[:, 1], dof_residual_)
    elseif vcov isa Vcov.RobustCovariance
        vcov_data = Vcov.VcovData(Xhat, crossx, score[:, 1] ./ Xhat[:, 1], nobs)
    else
        vcov_data = Vcov.VcovData(Xhat, crossx, ones(dof_residual_), dof_residual_)
    end
    """

    # Compute standard error
    matrix_vcov = StatsAPI.vcov(vcov_data, vcov_method)
    oldy = oldy[esample]
    # would need to change if weights are added
    ϕ_ll = dev/length(oldy)
    ll = sum(glfe_loglik_obs.(Ref(distribution), oldy, mu, 1, ϕ_ll))

    ϕ_nll = nulldev/length(oldy)
    mu_nll = has_intercept || has_fes ? mean(oldy) : linkinv(link, zero(eltype(oldy))/1)
    null_ll = sum(glfe_loglik_obs.(Ref(distribution), oldy, mu_nll, 1, ϕ_nll))

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
        ll,
        null_ll,
        distribution,
        link,
        coef_names,
        response_name,
        formula_origin, # Original formula
        formula_schema,
        nobs,   # Number of observations
        dof_coef_and_fe, # Number of coefficients
        dof_residual_,  # nobs - degrees of freedoms
        dev, # Deviance of the fitted model
        nulldev, # null deviance
        score,   # concentrated gradient
        hessian  # concentrated hessian
    )

end
