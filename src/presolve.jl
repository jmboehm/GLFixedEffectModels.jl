function detect_sep_fe!(esample::BitVector, y::Vector{<: Real}, Xexo::Matrix{<: Real}, fes::AbstractVector{<: FixedEffect}; sep_at::Real=0)
    # update esample, y, Xexo and fes
    before_n = sum(esample)
    sub_esample = trues(before_n)

    @assert before_n == length(y) "esample and y have different length."

    for fe in fes
        cache = trues(fe.n)
        level_visited = falses(fe.n)
        @assert before_n == length(fe.refs) "esample and fe have different length."
        for i in 1:before_n
            # for cache = true finally, all y[i] must equal to sep_at
            if y[i] == sep_at
                cache[fe.refs[i]] *= true
            else
                cache[fe.refs[i]] *= false
            end
            level_visited[fe.refs[i]] = true
        end
        for i in 1:before_n
            # if all y in this ref are equal to sep_at, exclude these ys
            if cache[fe.refs[i]] & level_visited[fe.refs[i]]
                sub_esample[i] = false
            end
        end
    end
    after_n = sum(sub_esample)
    dropped_n = before_n - after_n
    if dropped_n > 0
        @info "$(dropped_n) observations detected as separated using the FE method. Dropping them ..."
    end

    # drop them in esample
    remaining = findall(esample)
    esample[remaining[.~sub_esample]] .= false

    # drop them in fe
    fes = FixedEffect[_subset(fe, sub_esample) for fe in fes]

    return esample, y[sub_esample], Xexo[sub_esample, :], fes
end


function detect_sep_simplex(y::Vector{<: Real}, Xexo::Matrix{<: Real}, fes::AbstractVector{<: FixedEffect})
end


function detect_sep_relu!(esample::BitVector, y::Vector{<: Real}, Xexo::Matrix{<: Real}, fes::AbstractVector{<: FixedEffect}; 
    double_precision::Bool = true,
    dtol::Real = sqrt(eps(double_precision ? Float64 : Float32)), # tol for solve_coefficients and solve_residuals
    dmaxiter::Integer = 100_000, # maxiter for solve_coefficients and solve_residuals
    rtol::Real = 1e-4, # tol for ReLU
    rmaxiter::Integer = 100, # maxiter for ReLU
    method::Symbol = :cpu,
    verbose::Bool = false 
    )
    verbose && @info "identifying separations using ReLU."
    @assert all(GLFixedEffectModels.basecol(Xexo)) "There are Multicollinearity in the data, this should be done with before running ReLU."

    before_n = sum(esample)
    @assert before_n == length(y)

    Xexo_copy = deepcopy(Xexo)

    u = (y.==0)
    K = sum(u) / rtol^2
    w = (y.>0) .* K + .~(y.>0)
    weights = Weights(w)
    feM = AbstractFixedEffectSolver{double_precision ? Float64 : Float32}(fes, weights, Val{method})

    outer_converged = false

    for iter in 1:rmaxiter
        iterations = Int[]
        convergeds = Bool[]
        verbose && println("* iter $(iter)")

        Xexo = deepcopy(Xexo_copy)
        Xexo, b, c = solve_residuals!(Xexo, feM; tol = dtol, maxiter = dmaxiter)
        append!(iterations, b)
        append!(convergeds, c)
        crossx = cholesky!(Symmetric(Xexo' * Xexo))
        beta = crossx \ (Xexo' * u)
        xb = Xexo_copy * beta
        
        Xexo = deepcopy(Xexo_copy)
        newfes, b, c = solve_coefficients!(u - xb, feM; tol = dtol, maxiter = dmaxiter)
        append!(iterations, b)
        append!(convergeds, c)

        iterations = maximum(iterations)
        converged = all(convergeds)

        if converged == false
            @warn "Convergence of annihilation procedure not achieved in $(iterations) iterations; try increasing dmaxiter or decreasing dtol."
            @warn "cannot identify separated obs because can't solve lsmr. Skipping ..."
            return esample, y, Xexo, fes
        end

        xbd = sum(newfes) + xb
        xbd[abs.(xbd) .< rtol] .= 0

        if all(xbd.>=0)
            outer_converged = true
            is_sep = xbd .> 0
            @info "$(sum(is_sep)) observations detected as separated using the ReLU method. Dropping them ..."
            sub_esample = .~is_sep

            # drop them in esample
            remaining = findall(esample)
            esample[remaining[is_sep]] .= false

            # drop them in fe
            fes = FixedEffect[GLFixedEffectModels._subset(fe, sub_esample) for fe in fes]

            # drop them in y
            y = y[sub_esample]

            # drop them in Xexo
            Xexo = Xexo[sub_esample,:]

            return esample, y, Xexo, fes
        else
            verbose && println("negative xbd: $(sum(xbd.<0))")
        end

        u = xbd
        u[u .< 0] .= 0
    end

    if ~outer_converged
        @warn "cannot identify separated obs. Maximal iteration reached. Skipping ..."
        return esample, y, Xexo, fes
    end
end

function detect_linear_dependency_among_X!(X::Matrix{<: Real}, basecoef::BitVector; coefnames::Vector)
    # assert this is a model without IV or weights, this code is written at GLFixedEffectModel@0.3.1
    before_n = sum(basecoef)
    @assert before_n == size(X,2) "Dimension of basecoef and X doesn't match"
    base = GLFixedEffectModels.basecol(X)
    if all(base)
        return X, basecoef
    else
        X = GLFixedEffectModels.getcols(X, base)
        remaining_cols = findall(basecoef)
        regressor_ind_to_be_dropped = remaining_cols[.~base]
        basecoef[regressor_ind_to_be_dropped] .= 0 # update basecoef
        @info "Multicollinearity detected among columns of X. Dropping regressors: $(join(coefnames[regressor_ind_to_be_dropped]," "))"
        return X, basecoef
    end
end
