"""
    BiasCorr(model::GLFixedEffectModel,df::DataFrame;L::Int64=0,panel_structure::Any="classic")

Asymptotic bias correction after fitting binary choice models with a two-/three-way error.

# Arguments
## Required Arguments
- `model::Integer`: a `GLFixedEffectModel` object which can be obtained by using `nlreg()`.
- `df::DataFrame`: the Data Frame on which you just run `nlreg()`.
## Optional Arguments
- `L:Int64`: choice of binwidth, see Hahn and Kuersteiner (2011). The default value is 0.
- `panel_structure`: choose from "classic" or "network". The default value is "classic".

# Notice
This function only supports binomial distribution and probit/logit link. It also only supports two-way and three-way FE at the moment.
Always turn on the save option when running `nlreg()` before invoking `BiasCorr()`.
"""
function BiasCorr(model::GLFixedEffectModel,df::DataFrame;L::Int64=0,panel_structure::Any="classic")
    # TO-DO: add choice of L (binwidth)
    # UPDATE: The procedures that use the bandwidth L should be applied to the time dimension. 
    #         alpaca::biasCorr in R didn't automatically distinguish between individual fe and time fe. 
    #         When constructing Bhat (notation inhereted from Fernández-Val and Weidner, Annual Review of Economics (2018), pp120),
    #         alpaca distinguish i dimension and t dimension according to the variable positions in the formula,
    #         but this didn't show up in their documentation. 
    #         This can lead to inaccurate results (one can test alpaca::biasCorr with L>0 and with two different formula y~x|i+t and y~x|t+i).
    #         Is this the best practice? should we also implemented the bandwidth this way? At least we need to include this in the user manual.
    # TO-DO: check if df is sorted. It must be sorted to produce the right result when panel_structure == "network" and L > 0
    @assert ncol(model.augmentdf) > 1 "please save the entire augmented data frame in order to do bias correction"
    @assert panel_structure in ["classic", "network"] "you can only choose 'classic' or 'network' for panel_structure"
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
        # assuming the time fe in the fomula always comes the last
        b += groupSumsSpectral(score ./ v .* w, v, w, L,getGroupSeg(fes[!,1]))
    else
        # assuming the in the two-way network structure, the first two FEs are always it and jt.
        for fe in eachcol(fes)
            b += groupSums(MX_times_z, w, getGroupSeg(fe)) ./ 2.0
        end
        # assuming the ij fixed effect in the fomula always comes the last in the formula
        # and assuming that fes is sorted
        if size(fes)[2] == 3
            b += groupSumsSpectral(score ./ v .* w, v, w, L, getGroupSeg(fes[!,3]))
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

function groupSumsSpectral(M::Array{Float64,2}, v::Array{Float64,1}, w::Array{Float64,1}, L::Int64, group_seg::Array{Array{Bool,1},1})
    # TO-DO: Need to make sure the slice M[seg_index,p], v[seg_index] are sorted from early to late observations
    P = size(M)[2] # number of regressos P
    b_temp = zeros(P)

    for seg_index in group_seg
        T = sum(seg_index)
        numerator = zeros(P)
        for p in 1:P
            for l in 1:L
                for t in (l+1):T
                    numerator[p] += M[seg_index,p][t] * v[seg_index][t-l] * T / (T-l)
                end
            end
        end
        denominator = sum(w[seg_index])
        b_temp += numerator./denominator
    end
    return b_temp
end