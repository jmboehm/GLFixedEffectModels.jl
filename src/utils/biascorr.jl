##############################################
#                 model types                #
##############################################

# (Binomial) all possible FEs in a ijt-structured data set:
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  panel_structure |  ?-way  |        FE notation       |   need correction ?    |    is supported ?    | corresponding literature
# -----------------+---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    2    |  i + t                   |          YES           |         YES          |  Fernández-Val and Weidner (2016) 
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#     classic      |    2    |  i + j (pseudo panel)    |         YES (?)        |          ?           |  
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    3    |  i + j + t               |           NO           |          NO          |  Fernández-Val and Weidner (2018) 
# -----------------+---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    2    |  it + jt                 |          YES (?)       |         YES          |  Hinz, Stammann and Wanner (2020) & Fernández-Val and Weidner (2016) 
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#     network      |    2    |  it + ij, jt + ij        |           ?            |          ?           |  ?
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    2    |  it + jt + ij            |          YES           |         YES          |  Hinz, Stammann and Wanner (2020)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# (Poisson) all possible FEs in a ijt-structured data set:
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  panel_structure |  ?-way  |        FE notation       |   need correction ?    |    is supported ?    | corresponding literature
# -----------------+---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    2    |  i + t, j + t            |           NO           |          NO          |  Fernández-Val and Weidner (2016) 
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#     classic      |    2    |  i + j (pseudo panel)    |           ?            |          ?           |  
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    3    |  i + j + t               |           NO           |          NO          |  Fernández-Val and Weidner (2018) 
# -----------------+---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    2    |  it + jt                 | YES (on standard error)|                      |  Weidner and Zylkin (2020)
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#     network      |    2    |  it + ij, jt + ij        |           ?            |          ?           |  ?
#                  +---------+--------------------------+------------------------+----------------------+-------------------------------------------------------------------
#                  |    2    |  it + jt + ij            |          YES           |                      |  Weidner and Zylkin (2020)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###################################################################
#                   Main Function bias_correction()               #
###################################################################

"""
    bias_correction(model::GLFixedEffectModel,df::DataFrame;i_symb::Union{Symbol,Nothing}=nothing,j_symb::Union{Symbol,Nothing}=nothing,t_symb::Union{Symbol,Nothing}=nothing,L::Int64=0,panel_structure::Symbol=:classic)

Asymptotic bias correction after fitting binary choice models with a two-/three-way error.

# Arguments
## Required Arguments
- `model::Integer`: a `GLFixedEffectModel` object which can be obtained by using `nlreg()`.
- `df::DataFrame`: the Data Frame on which you just run `nlreg()`.
## Optional Arguments
- `L:Int64`: choice of binwidth, see Hahn and Kuersteiner (2011). The default value is 0.
- `panel_structure`: choose from "classic" or "network". The default value is "classic".
- `i_symb`: the variable name for i index in the data frame `df`
- `j_symb`: the variable name for j index in the data frame `df`
- `t_symb`: the variable name for t index in the data frame `df`

# Available Model
We only support the following models:
- Binomial regression, Logit link, Two-way, Classic
- Binomial regression, Probit link, Two-way, Classic
- Binomial regression, Logit link, Two-way, Network
- Binomial regression, Probit link, Two-way, Network
- Binomial regression, Logit link, Three-way, Network
- Binomial regression, Probit link, Three-way, Network
- Poisson regression, Log link, Three-way, Network
- Poisson regression, Log link, Two-way, Network
"""
function bias_correction(model::GLFixedEffectModel,df::DataFrame;i_symb::Union{Symbol,Nothing}=nothing,j_symb::Union{Symbol,Nothing}=nothing,t_symb::Union{Symbol,Nothing}=nothing,L::Int64=0,panel_structure::Symbol=:classic)
    @assert :mu in propertynames(model.augmentdf) && :eta in propertynames(model.augmentdf) "please save :eta and :mu before bias correction"
    @assert panel_structure in [:classic, :network] "you can only choose :classic or :network for panel_structure"
    @assert typeof(model.distribution) <: Union{Binomial,Poisson} "currently only support binomial regression and poisson regression"
    @assert typeof(model.link) <: Union{GLM.LogitLink,GLM.ProbitLink,GLM.LogLink} "currently only support probit and logit link (binomial regression), and log link (poisson regression)"

    ######################## Parse FEs ########################
    # fes_in_formula is the fe symbols that appears in the formula after the "|" which might contain FE interaction
    # cleaned_fe_symb is an array of all the single FE that constructs fes_in_formula
    fes_in_formula, cleaned_fe_symb = parse_formula_get_FEs(model,panel_structure,df)
    @assert length(fes_in_formula) in [2,3] "We only support two-way and three-way FE at the moment"
    @assert all([symb ∈ [i_symb,j_symb,t_symb] for symb in cleaned_fe_symb]) "not all FEs in the formula is mentioned in i_symb, j_symb and t_symb"
    @assert all([(symb ∈ cleaned_fe_symb || symb === nothing) for symb in [i_symb,j_symb,t_symb]]) "redundant FEs in i_symb, j_symb or t_symb (not mentioned in the formula)"
    if panel_structure == :classic
        fe_dict = Dict(:i => i_symb,:j => j_symb,:t => t_symb)
    elseif panel_structure == :network
        fe_dict = Dict{Symbol,Union{Array{Symbol,1},Nothing}}()
        if Symbol("fe_",i_symb,"&fe_",j_symb) ∈ fes_in_formula
            push!(fe_dict, :ij => [i_symb,j_symb])
        else 
            push!(fe_dict, :ij => nothing)
        end

        if Symbol("fe_",i_symb,"&fe_",t_symb) ∈ fes_in_formula
            push!(fe_dict, :it => [i_symb,t_symb])
        else 
            push!(fe_dict, :it => nothing)
        end

        if Symbol("fe_",j_symb,"&fe_",t_symb) ∈ fes_in_formula
            push!(fe_dict, :jt => [j_symb,t_symb])
        else 
            push!(fe_dict, :jt => nothing)
        end
    end
    ##########################################################

    ########## sort df ############ 
    df.old_ind = rownumber.(eachrow(df))
    df = sort(df, [t_symb,j_symb,i_symb][[t_symb,j_symb,i_symb] .!== nothing])
    ###############################

    # check if we currently support the combination of the distribution, the link, num of FEs and the panel_structure
    available_types = [
        (LogitLink, Binomial, 2, :classic), # Fernández-Val and Weidner (2016, 2018) 
        (ProbitLink, Binomial, 2, :classic), # Fernández-Val and Weidner (2016, 2018)
        (LogitLink, Binomial, 2, :network), # Hinz, Stammann and Wanner (2020) & Fernández-Val and Weidner (2016)
        (ProbitLink, Binomial, 2, :network), # Hinz, Stammann and Wanner (2020) & Fernández-Val and Weidner (2016)
        (LogitLink, Binomial, 3, :network), # Hinz, Stammann and Wanner (2020)
        (ProbitLink, Binomial, 3, :network), # Hinz, Stammann and Wanner (2020)
        (LogLink, Poisson, 2, :network), # Weidner and Zylkin (2021), JIE
        (LogLink, Poisson, 3, :network) # Weidner and Zylkin (2021), JIE
    ]
    this_model_type = (model.link,model.distribution,length(fes_in_formula),panel_structure)
    @assert model_type_checker(this_model_type, available_types) "We currently don't support this combination of the distribution, the link, num of FEs and panel_structure"

    if model.link isa GLM.LogitLink && model.distribution isa Binomial
        return biascorr_logit(model,df,fe_dict,L,panel_structure)
    elseif model.link isa GLM.ProbitLink && model.distribution isa Binomial
        return biascorr_probit(model,df,fe_dict,L,panel_structure)
    elseif model.link isa GLM.LogLink && model.distribution isa Poisson
        return biascorr_poisson(model,df,fe_dict,L,panel_structure)
    end
end


############################################################
#           Bias Correction In Different Models            #
############################################################
function biascorr_probit(model::GLFixedEffectModel,df::DataFrame,fes::Dict,L::Int64,panel_structure::Symbol)
    link = model.link
    y = df[model.esample[df.old_ind],model.yname]
    η = model.augmentdf.eta[df.old_ind]
    μ = model.augmentdf.mu[df.old_ind]
    μη = GLM.mueta.(Ref(link),η)
    score = model.gradient[df.old_ind,:]
    hessian = model.hessian
    
    w = μη ./ (μ.*(1.0 .- μ))
    v = w .* (y - μ)
    w = w .* μη
    z = - η .* w

    if panel_structure == :classic
        b = classic_b_binomial(score,v,z,w,L,fes,df)
    elseif panel_structure == :network
        b = network_b_binomial(score,v,z,w,L,fes,df)
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
        model.y,
        model.mu,
        model.distribution,
        model.link,
        model.coefnames,
        model.yname,
        model.formula,
        model.formula_schema,
        model.nobs,
        model.dof,
        model.dof_residual,
        model.deviance,
        model.nulldeviance,
        model.gradient,
        model.hessian
    )

end

function biascorr_logit(model::GLFixedEffectModel,df::DataFrame,fes::Dict,L::Int64,panel_structure::Symbol)
    link = model.link
    y = df[model.esample[df.old_ind],model.yname]
    η = model.augmentdf.eta[df.old_ind]
    μ = model.augmentdf.mu[df.old_ind]
    μη = GLM.mueta.(Ref(link),η)
    score = model.gradient[df.old_ind,:]
    hessian = model.hessian

    v = y .- μ
    w = μη
    z = w .* (1.0 .- 2.0 .* μ)
    if panel_structure == :classic
        b = classic_b_binomial(score,v,z,w,L,fes,df)
    elseif panel_structure == :network
        b = network_b_binomial(score,v,z,w,L,fes,df)
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
        model.y,
        model.mu,
        model.distribution,
        model.link,
        model.coefnames,
        model.yname,
        model.formula,
        model.formula_schema,
        model.nobs,
        model.dof,
        model.dof_residual,
        model.deviance,
        model.nulldeviance,
        model.gradient,
        model.hessian
    )
end

function biascorr_poisson(model::GLFixedEffectModel,df::DataFrame,fes::Dict,L::Int64,panel_structure::Symbol)
    if L > 0
        printstyled("bandwidth not allowed in poisson regression bias correction. Treating L as 0...",color=:yellow)
    end
    # @assert panel_structure == :network
    y = df[model.esample[df.old_ind],model.yname] # Do we need to subset y with model.esample? TO-DO: test with cases with esample is not all ones.
    λ = model.augmentdf.mu[df.old_ind]
    @assert all([fe_key==:ij || fe_symb !== nothing for (fe_key,fe_symb) in fes]) "You need either a three-way FE model or a two-way FE model with i#j being left out"
    
    # print("pre-demeaning")
    # @time begin
    # type: it + jt + ij, Need bias correction and standard error correction
    i_ind = df[model.esample[df.old_ind],fes[:it][1]]
    j_ind = df[model.esample[df.old_ind],fes[:jt][1]]
    t_ind = df[model.esample[df.old_ind],fes[:it][2]]
    i_levels = levels(i_ind)
    j_levels = levels(j_ind)
    t_levels = levels(t_ind)
    I = length(i_levels)
    J = length(j_levels)
    # @assert I==J "number of exporters is different from number of importers"
    T = length(t_levels)
    
    # assume balanced panel
    y_sum_by_ij = similar(y)
    λ_sum_by_ij = similar(λ)
    for groupSeg in get_group_seg(i_ind, j_ind)
        y_sum_by_ij[groupSeg] .= sum(y[groupSeg])
        λ_sum_by_ij[groupSeg] .= sum(λ[groupSeg])
    end
    ϑ = λ ./ λ_sum_by_ij
    y_sum_ij = reshape(y_sum_by_ij,(I,J,T))[:,:,1]
    λ_sum_ij = reshape(λ_sum_by_ij,(I,J,T))[:,:,1]
    ϑ_ijt = reshape(ϑ,(I,J,T))
    λ_ijt = reshape(λ,(I,J,T))
    # Construct S
    # S = y - ϑ .* y_sum_by_ij
    S = y - λ
    S_ijt = reshape(S,(I,J,T))
    # println(H)
    # Construct G
    # Construct x̃ (demeaned x)
    # See footnote 33 of Weidner and Zylkin (2020)
    X = df[model.esample[df.old_ind], model.coefnames] |> Array{Float64,2}
    weights = FixedEffects.Weights(λ)
    all(isfinite, weights) || throw("Weights are not finite")
    fes_fixedeffectarray = Array{FixedEffect,1}()
    for (fe_key,fe_symb) in fes
        if fe_symb !== nothing
            fe_fixedeffectobject = FixedEffect(df[model.esample[df.old_ind], fe_symb[1]], df[model.esample[df.old_ind], fe_symb[2]])
            push!(fes_fixedeffectarray,fe_fixedeffectobject)
        end
    end
    feM = AbstractFixedEffectSolver{Float64}(fes_fixedeffectarray, weights, Val{:cpu}) # CPU/GPU??? might need more attention in the future when implement GPU
    Xdemean, b, converged = FixedEffects.solve_residuals!(X, feM)
    if !all(converged)
        @warn "Convergence of annihilation procedure not achieved in default iterations; try increasing maxiter_center or decreasing center_tol."
    end
    K = size(Xdemean,2)
    Xdemean_ijtk = reshape(Xdemean,(I,J,T,K))

    if all([fe_symb !== nothing for (fe_key,fe_symb) in fes]) # it + jt + ij
        ###############################
        #  Point Estimate Correction  #
        ###############################
        # Construct B̂, D̂, and Ŵ
        N = I = J
        B̂ = zeros(K)
        ##newW## Ŵ = zeros(K,K)
        B!(B̂,K,I,J,T,ϑ_ijt,λ_sum_ij,y_sum_ij,Xdemean_ijtk,S_ijt)
        B̂ = B̂ ./ (N - 1)
        ##newW## Ŵ = Ŵ ./ (N*(N-1))
        D̂ = zeros(K)
        D!(D̂,K,I,J,T,ϑ_ijt,λ_sum_ij,y_sum_ij,Xdemean_ijtk,S_ijt)
        D̂ = D̂ ./ (N - 1)
        Ŵ = model.hessian./(N*(N-1))
        β = model.coef - Ŵ \ (B̂ + D̂) ./ (N-1)

        ###############################
        #  Standard Error Correction  #
        ###############################
        new_vcov = get_new_vcov(I,J,T,K,Xdemean_ijtk,model.hessian,ϑ_ijt,λ_sum_ij,S_ijt)
    else # it + jt 
        β = model.coef
        new_vcov = get_new_vcov_twoway(I,J,T,K,Xdemean_ijtk,model.hessian,λ_ijt,S_ijt)
    end

    return GLFixedEffectModel(
        β,
        new_vcov,
        model.vcov_type,
        model.nclusters,
        model.iterations,
        model.converged,
        model.esample,
        model.augmentdf,
        model.y,
        model.mu,
        model.distribution,
        model.link,
        model.coefnames,
        model.yname,
        model.formula,
        model.formula_schema,
        model.nobs,
        model.dof,
        model.dof_residual,
        model.deviance,
        model.nulldeviance,
        model.gradient,
        model.hessian
    )
end

###########################################
#           Internal Functions            #
###########################################

function group_sums(M::Array{Float64,2},w::Array{Float64,1},group_seg::Vector{Vector{Int64}})
    P = size(M,2) # number of regressos P
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

function get_group_seg(fe::Array{T,1} where T <: Any)
    p = sortperm(fe)
    q = fe[p]
    res = Vector{Vector{Int64}}()
    grp = Vector{Int64}()
    is_first = true
    last = fe[end]
    for (i,v) in enumerate(q)
        if !is_first && last != v
            push!(res,grp)
            grp = [p[i]]
        else
            push!(grp, p[i])
        end
        last = v
        is_first = false
    end
    push!(res,grp)
    return res
end

function get_group_seg(fe1::Array{T,1},fe2::Array{T,1}) where T <: Any
    fe = collect(zip(fe1,fe2))
    p = sortperm(fe)
    q = fe[p]
    res = Vector{Vector{Int64}}()
    grp = Vector{Int64}()
    is_first = true
    last = q[1]
    for (i,v) in enumerate(q)
        if !is_first && last != v
            push!(res,grp)
            grp = [p[i]]
        else
            push!(grp, p[i])
        end
        last = v
        is_first = false
    end
    push!(res,grp)
    return res 
end

function group_sums_spectral(M::Array{Float64,2}, v::Array{Float64,1}, w::Array{Float64,1}, L::Int64, group_seg::Vector{Vector{Int64}})
    # TO-DO: Need to make sure the slice M[seg_index,p], v[seg_index] are sorted from early to late observations
    P = size(M)[2] # number of regressos P
    b_temp = zeros(P)

    for seg_index in group_seg
        T = length(seg_index)
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

function model_type_checker(x::Tuple,avail_list::Array{T} where T <: Any)
    # x is a 4-d tuple:
    # x[1] is the model's link
    # x[2] is the model's distribution
    # x[3] is the ttl number of FEs
    # x[4] is the panel_structure that takes [:classic,:network]
    for avail_type in avail_list
        if x[1] isa avail_type[1] && x[2] isa avail_type[2] && x[3] == avail_type[3] && x[4] == avail_type[4]
            return true
        end
    end
    return false
end
    
function parse_formula_get_FEs(model,panel_structure::Symbol,df::DataFrame)
    # This function returns the set of FE symbols according to the formula
    # If the panel_structure is network, it returns the interaction terms only
    # If the panel_structure is classic, it checks to see if there is interaction term in the formula. If true, it throws error. It returns all the fe symbols otherwise
    vars = StatsModels.termvars(model.formula) # all the vars excluding interactions
    vars_name_to_be_set_diff = Symbol.("fe_",vars) # all the vars symbols excluding interactions (add "fe_" in the front)
    fes, ids, formula = FixedEffectModels.parse_fixedeffect(df, model.formula) # id: include fe_i, fe_j, fe_t, and possible interactions if panel_structure = :network
    if panel_structure == :network
        network_fes = setdiff(ids,vars_name_to_be_set_diff)
        if length(network_fes) != 0
            no_interaction_fe = setdiff(ids,network_fes)
            return network_fes, Symbol.(s[4:end] for s in String.(no_interaction_fe))
        else
            throw("no network fe is found")
        end
    elseif panel_structure == :classic
        classic_fes = intersect(ids,vars_name_to_be_set_diff)
        if issetequal(classic_fes, ids)
            return ids, Symbol.(s[4:end] for s in String.(ids))
        else
            throw("no fe interaction is allowed in the classic model")
        end
    end
end

function classic_b_binomial(score::Array{Float64,2},v::Array{Float64,1},z::Array{Float64,1},w::Array{Float64,1},L::Int64,fes::Dict,df::DataFrame)
    P = size(score)[2]
    b = zeros(P)
    if fes[:t] === nothing
        pseudo_panel = true
        if L > 0
            printstyled("bandwidth not allowed in classic ij-pseudo panel. Treating L as 0...")
        end
    end
    for (fe_key,fe_symb) in fes
        if fe_symb !== nothing
            b += group_sums(score ./ v .* z, w, get_group_seg(df[!,fe_symb])) ./ 2.0
        end
        if fe_key != :t && L > 0 && !pseudo_panel
            b += group_sums_spectral(score ./ v .* w, v, w, L,get_group_seg(df[!,fe_key]))
        end
    end
    return b
end

function network_b_binomial(score::Array{Float64,2},v::Array{Float64,1},z::Array{Float64,1},w::Array{Float64,1},L::Int64,fes::Dict,df::DataFrame)
    P = size(score)[2]
    b = zeros(P)
    for (fe_key,fe_symb) in fes
        if fe_symb !== nothing
            b += group_sums(score ./ v .* z, w, get_group_seg(df[!,fe_symb[1]], df[!,fe_symb[2]])) ./ 2.0
        end
    end
    if L > 0 && fes[:ij] !== nothing
        b += group_sums_spectral(score ./ v .* w, v, w, L,get_group_seg(df[!,fes[:ij][1]], df[!,fes[:ij][2]]))
    end
    return b
end

function G_ijtsr(i,j,t,s,r,ϑ_ijt,y_sum_ij)
    if r!=s && s!=t && t!=r
        return - 2 * ϑ_ijt[i,j,r] * ϑ_ijt[i,j,s] * ϑ_ijt[i,j,t] * y_sum_ij[i,j]
    end
    if r==t && t!=s 
        return ϑ_ijt[i,j,t] * (1 - 2*ϑ_ijt[i,j,t]) * ϑ_ijt[i,j,s] * y_sum_ij[i,j]
    end
    if t==s && s!=r
        return ϑ_ijt[i,j,s] * (1 - 2*ϑ_ijt[i,j,s]) * ϑ_ijt[i,j,r] * y_sum_ij[i,j]
    end
    if s==r && r!=t
        return ϑ_ijt[i,j,s] * (1 - 2*ϑ_ijt[i,j,s]) * ϑ_ijt[i,j,t] * y_sum_ij[i,j]
    end
    if t==s && t==r
        return - ϑ_ijt[i,j,t] * (1 - ϑ_ijt[i,j,t]) * (1 - 2*ϑ_ijt[i,j,t]) * y_sum_ij[i,j]
    end
end
function H_ij(i,j,ϑ_ijt,y_sum_ij)
    # H[i,j,:,:]
    return (- ϑ_ijt[i,j,:] * ϑ_ijt[i,j,:]' + Diagonal(ϑ_ijt[i,j,:])) .* y_sum_ij[i,j] 
end
function H̄_ij(i,j,ϑ_ijt,λ_sum_ij)
    # H̄[i,j,:,:]
    return (- ϑ_ijt[i,j,:] * ϑ_ijt[i,j,:]' + Diagonal(ϑ_ijt[i,j,:])) .* λ_sum_ij[i,j] 
end

function Λ_ij(i,j,λ_ijt)
    # H̄[i,j,:,:]
    return Diagonal(λ_ijt[i,j,:])
end

function G_ij_times_x_ijk(i,j,k,Xdemean_ijtk,T,ϑ_ijt,y_sum_ij)
    # [Gij * xij,k]_st
    result = zeros(T,T)
    for s ∈ 1:T
        for t ∈ 1:T
            for r ∈ 1:T
                result[s,t] += G_ijtsr(i,j,r,s,t,ϑ_ijt,y_sum_ij) * Xdemean_ijtk[i,j,r,k]
            end
        end
    end
    return result
end

function B!(B̂,K,I,J,T,ϑ_ijt,λ_sum_ij,y_sum_ij,Xdemean_ijtk,S_ijt)
    for k ∈ 1:K
        for i ∈ 1:I
            # Construct: H̄_pseudo_inv
            #            Hx̃S'
            #            Gx̃
            #            SS'
            #            x̃Hx̃
            H̄_sum_along_j_fix_i = zeros(T,T)
            Hx̃S_fix_i = zeros(T,T)
            Gx̃_fix_i = zeros(T,T)
            SS_fix_i = zeros(T,T)
            @turbo for j ∈ 1:J
                #if i != j # uncomment to not include terms where i==j
                    H̄_sum_along_j_fix_i += H̄_ij(i,j,ϑ_ijt,λ_sum_ij)
                    Hx̃S_fix_i += H_ij(i,j,ϑ_ijt,y_sum_ij) * Xdemean_ijtk[i,j,:,k] * transpose(S_ijt[i,j,:])
                    Gx̃_fix_i += G_ij_times_x_ijk(i,j,k,Xdemean_ijtk,T,ϑ_ijt,y_sum_ij)
                    SS_fix_i += S_ijt[i,j,:] * transpose(S_ijt[i,j,:])
                    ##newW## Ŵ += Xdemean_ijtk[i,j,:,:]' * H̄[i,j,:,:] * Xdemean_ijtk[i,j,:,:]
                #end
            end
            H̄_pseudo_inv = pinv(H̄_sum_along_j_fix_i)
            term1 = - H̄_pseudo_inv * Hx̃S_fix_i
            term2 = Gx̃_fix_i * H̄_pseudo_inv * SS_fix_i * H̄_pseudo_inv ./ 2.0
            B̂[k] += tr(term1 + term2)
        end
    end
end

function D!(D̂,K,I,J,T,ϑ_ijt,λ_sum_ij,y_sum_ij,Xdemean_ijtk,S_ijt)
    for k ∈ 1:K
        for j ∈ 1:J
            # Construct: H̄_pseudo_inv
            #            Hx̃S'
            #            Gx̃
            #            SS'
            H̄_sum_along_i_fix_j = zeros(T,T)
            Hx̃S_fix_j = zeros(T,T)
            Gx̃_fix_j = zeros(T,T)
            SS_fix_j = zeros(T,T)
            @turbo for i ∈ 1:I
                #if i != j # uncomment to not include terms where i==j
                    H̄_sum_along_i_fix_j += H̄_ij(i,j,ϑ_ijt,λ_sum_ij)
                    Hx̃S_fix_j += H_ij(i,j,ϑ_ijt,y_sum_ij) * Xdemean_ijtk[i,j,:,k] * transpose(S_ijt[i,j,:])
                    Gx̃_fix_j += G_ij_times_x_ijk(i,j,k,Xdemean_ijtk,T,ϑ_ijt,y_sum_ij)
                    SS_fix_j += S_ijt[i,j,:] * transpose(S_ijt[i,j,:])
                #end
            end
            H̄_pseudo_inv = pinv(H̄_sum_along_i_fix_j)
            term1 = - H̄_pseudo_inv * Hx̃S_fix_j
            term2 = Gx̃_fix_j * H̄_pseudo_inv * SS_fix_j * H̄_pseudo_inv ./ 2.0
            D̂[k] += tr(term1 + term2)
        end
    end
end

function Ω(I,J,T,K,Xdemean_ijtk,Ŵ,ϑ_ijt,λ_sum_ij,S_ijt)
    corrected_Ω = zeros(K,K)
    HH = inv_W_ϕ_constructor(I,J,T,ϑ_ijt,λ_sum_ij)
    for i in 1:I
        for j in 1:J
            # construct Var(S_ij|x_ij)
            # Xdemean_ijtk[i,j,:,:] is T-by-K
            bias1 = Xdemean_ijtk[i,j,:,:] * inv(Ŵ) * transpose(Xdemean_ijtk[i,j,:,:])
            bias2 = d_ij_constructor(i,j,I,J,T) * HH * transpose(d_ij_constructor(i,j,I,J,T))
            lev_correction = LinearAlgebra.I(T) - ( H̄_ij(i,j,ϑ_ijt,λ_sum_ij) * (bias1+bias2) )
            EŜŜ = S_ijt[i,j,:] * transpose(S_ijt[i,j,:])
            ESS = lev_correction \ EŜŜ
            corrected_Ω += transpose(Xdemean_ijtk[i,j,:,:]) * ESS * Xdemean_ijtk[i,j,:,:]
        end
    end
    return corrected_Ω ./ (I*J)
end

function Ω_twoway(I,J,T,K,Xdemean_ijtk,Ŵ,λ_ijt,S_ijt)
    corrected_Ω = zeros(K,K)
    HH = inv_W_ϕ_constructor_twoway(I,J,T,λ_ijt)
    for i in 1:I
        for j in 1:J
            # construct Var(S_ij|x_ij)
            # Xdemean_ijtk[i,j,:,:] is T-by-K
            bias1 = Xdemean_ijtk[i,j,:,:] * inv(Ŵ) * transpose(Xdemean_ijtk[i,j,:,:])
            bias2 = d_ij_constructor(i,j,I,J,T) * HH * transpose(d_ij_constructor(i,j,I,J,T))
            lev_correction = LinearAlgebra.I(T) - (Λ_ij(i,j,λ_ijt) * (bias1+bias2) )
            EŜŜ = S_ijt[i,j,:] * transpose(S_ijt[i,j,:])
            ESS = lev_correction \ EŜŜ
            corrected_Ω += transpose(Xdemean_ijtk[i,j,:,:]) * ESS * Xdemean_ijtk[i,j,:,:]
        end
    end
    return corrected_Ω ./ (I*J)
end

function inv_W_ϕ_constructor(I,J,T,ϑ_ijt,λ_sum_ij)
    H_ij_full = Array{Array{Float64,2},2}(undef,I,J)
    for i in 1:I
        for j in 1:J
            H_ij_full[i,j] = H̄_ij(i,j,ϑ_ijt,λ_sum_ij)
        end
    end
    Hαα = zeros(I*T,I*T)
    Hγγ = zeros(J*T,J*T)
    H̄_sum_along_i_fix_j = sum(H_ij_full,dims=1)
    H̄_sum_along_j_fix_i = sum(H_ij_full,dims=2)
    for i in 1:I
        Hαα[((i-1)*T+1) : i*T, ((i-1)*T+1) : i*T] = H̄_sum_along_j_fix_i[i]
    end
    for j in 1:J
        Hγγ[((j-1)*T+1) : j*T, ((j-1)*T+1) : j*T] = H̄_sum_along_i_fix_j[j]
    end
    Hαγ = cell2full(H_ij_full)
    Hϕϕ = [Hαα Hαγ;transpose(Hαγ) Hγγ]
    return pinv(Hϕϕ)
end

function inv_W_ϕ_constructor_twoway(I,J,T,λ_ijt)
    H_ij_full = Array{Array{Float64,2},2}(undef,I,J)
    for i in 1:I
        for j in 1:J
            H_ij_full[i,j] = Λ_ij(i,j,λ_ijt)
        end
    end
    Hαα = zeros(I*T,I*T)
    Hγγ = zeros(J*T,J*T)
    H̄_sum_along_i_fix_j = sum(H_ij_full,dims=1)
    H̄_sum_along_j_fix_i = sum(H_ij_full,dims=2)
    for i in 1:I
        Hαα[((i-1)*T+1) : i*T, ((i-1)*T+1) : i*T] = H̄_sum_along_j_fix_i[i]
    end
    for j in 1:J
        Hγγ[((j-1)*T+1) : j*T, ((j-1)*T+1) : j*T] = H̄_sum_along_i_fix_j[j]
    end
    Hαγ = cell2full(H_ij_full)
    Hϕϕ = [Hαα Hαγ;transpose(Hαγ) Hγγ]
    return pinv(Hϕϕ)
end

function d_ij_constructor(i,j,I,J,T)
    i_indicator = falses(1,I)
    j_indicator = falses(1,J)
    i_indicator[i] = true
    j_indicator[j] = true
    idenTT = LinearAlgebra.I(T)
    return LinearAlgebra.kron(hcat(i_indicator,j_indicator),idenTT)
end

function get_new_vcov(I,J,T,K,Xdemean_ijtk,Ŵ,ϑ_ijt,λ_sum_ij,S_ijt)
    corrected_Ω = Ω(I,J,T,K,Xdemean_ijtk,Ŵ,ϑ_ijt,λ_sum_ij,S_ijt)
    inv_Ŵ = inv(Ŵ)
    new_vcov = inv_Ŵ * corrected_Ω * inv_Ŵ
    return new_vcov .* ((I*J) * (I*J)/(I*J-1))
end

function get_new_vcov_twoway(I,J,T,K,Xdemean_ijtk,Ŵ,λ_ijt,S_ijt)
    corrected_Ω = Ω_twoway(I,J,T,K,Xdemean_ijtk,Ŵ,λ_ijt,S_ijt)
    inv_Ŵ = inv(Ŵ)
    new_vcov = inv_Ŵ * corrected_Ω * inv_Ŵ
    return new_vcov .* ((I*J) * (I*J)/(I*J-1))
end

function cell2full(cell::Matrix{Matrix{Float64}})
    sentinel = cell[1,1]
    I,J = size(cell)
    T = size(sentinel)[1]
    full = zeros(I*T,J*T)
    for i in 1:I
        for j in 1:J
            full[ ((i-1)*T+1):(i*T), ((j-1)*T+1):(j*T) ] += cell[i,j]
        end
    end
    return full
end