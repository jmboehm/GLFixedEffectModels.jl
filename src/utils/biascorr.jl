function BiasCorr(model::GLFixedEffectModel,L::Int64,panel_structure::Any,df::DataFrame)
    # TO-DO: add choice of L (binwidth), see Hahn and Kuersteiner (2011)
    @assert ncol(model.augmentdf) > 1 "please save the entire augmented data frame in order to do bias correction"
    @assert panel_structure in ["classic", "network"] "you can only choose 'classic' or 'network' for panel_structure"
    # TO-DO: different panel_structure only makes a difference when binwidth is not 0
    @assert typeof(model.distribution) <: Binomial "currently only support binomial distribution"
    @assert typeof(model.link) <: Union{GLM.LogitLink,GLM.ProbitLink} "currently only support probit and logit link"
    @assert ncol(model.augmentdf) in [3,4] "We only support two-way and three-way FE at the moment"
    link = model.link
    y = df[model.esample,model.yname]
    residuals = model.augmentdf.residuals
    η = y - residuals
    μ = GLM.linkinv.(Ref(link),η)
    μη = GLM.mueta.(Ref(link),η)
    score = model.gradient
    hessian = model.hessian

    ## Use the values in saved FE to distinguish groups (Is this reliable?)
    fes = select(model.augmentdf, Not(:residuals))


    if typeof(model.link) <: GLM.LogitLink
        v = y .- μ
        w = μη
        z = w .* (1.0 .- 2.0 .* μ)
    else
        w = μη ./ (μ.*(1.0 .- μ))
        v = w .* (y - μ)
        w = w .* μη
        z = - η .* w
    end
    
    MX_times_z = score ./ v .* z
    P = size(MX_times_z)[2]
    b = zeros(P)

    if panel_structure == "classic"
        @assert size(fes)[2] != 3 "panel.structure == 'classic' expects a two-way fixed effects model."
        for fe in eachcol(fes)
            b += groupSums(MX_times_z, w, getGroupSeg(fe)) ./ 2.0
        end
    else
        for fe in eachcol(fes)
            b += groupSums(MX_times_z, w, getGroupSeg(fe)) ./ 2.0
        end
    end

    β = model.coef + hessian \ b
    return GLFixedEffectModel(
        β,
        model.vcov,
        model.vcov_type,
        model.nclusters,
        model.iterations,
        model.converged,
        model.esample,
        model.augmentdf,
        model.distribution,
        model.link,
        model.coefnames,
        model.yname,
        model.formula,
        model.formula_schema,
        model.nobs,
        model.dof_residual,
        model.deviance,
        model.nulldeviance,
        model.gradient,
        model.hessian
    )

end

function groupSums(M::Array{Float64,2},w::Array{Float64,1},group_seg::Array{Array{Bool,1},1})
    P = size(M)[2] # number of regressos P
    b_temp = zeros(P)

    for seg_index in group_seg
        numerator = zeros(P)
        for p in 1:P
            numerator[p] += sum(M[seg_index,p])
        end

        denominator = sum(w[seg_index])
        b_temp += numerator./denominator
    end
    return b_temp
end

function getGroupSeg(fe::Array{T,1} where T <: Any)
    theLevels = levels(fe)

    list_of_index = Array{Array{Bool,1},1}(undef,0)
    
    for level in theLevels
        push!(list_of_index,fe .== level)
    end

    return list_of_index 
end