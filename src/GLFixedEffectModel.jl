
##############################################################################
##
## Type GLFixedEffectModel
##
##############################################################################

struct GLFixedEffectModel <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix
    vcov_type::CovarianceEstimator
    nclusters::Union{NamedTuple, Nothing}

    iterations::Int64
    converged::Bool

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame
    y::Vector{Float64}      # Dependent variable
    mu::Vector{Float64}     # Fitted values

    distribution::Distribution
    link::GLM.Link

    coefnames::Vector       # Name of coefficients
    yname::Union{String, Symbol} # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_schema

    nobs::Int64             # Number of observations
    dof::Int64              # Degrees of freedom: nparams (including fixed effects)
    dof_residual::Int64     # nobs - degrees of freedoms, adjusted for clustering

    deviance::Float64            # Deviance of the fitted model
    nulldeviance::Float64        # Null deviance, i.e. deviance of model with constant only

    gradient::Matrix{Float64}   # concentrated gradient
    hessian::Matrix{Float64}    # concentrated hessian

end

FixedEffectModels.has_fe(x::GLFixedEffectModel) = FixedEffectModels.has_fe(x.formula)


# Check API at  https://github.com/JuliaStats/StatsBase.jl/blob/65351de819ca64941cb81c047e4b77157446f7c5/src/statmodels.jl
# fields
StatsAPI.coef(x::GLFixedEffectModel) = x.coef
StatsAPI.coefnames(x::GLFixedEffectModel) = x.coefnames
StatsAPI.responsename(x::GLFixedEffectModel) = string(x.yname)
StatsAPI.vcov(x::GLFixedEffectModel) = x.vcov
StatsAPI.nobs(x::GLFixedEffectModel) = x.nobs
StatsAPI.dof(x::GLFixedEffectModel) = x.dof
StatsAPI.dof_residual(x::GLFixedEffectModel) = x.dof_residual
StatsAPI.islinear(x::GLFixedEffectModel) = (x.link == IdentityLink() ? true : false)
StatsAPI.deviance(x::GLFixedEffectModel) = x.deviance
StatsAPI.nulldeviance(x::GLFixedEffectModel) = x.nulldeviance

pseudo_r2(x::GLFixedEffectModel) = r2(x, :McFadden)
pseudo_adjr2(x::GLFixedEffectModel) = adjr2(x, :McFadden)

function StatsAPI.confint(x::GLFixedEffectModel, level::Real = 0.95)
    scale = quantile(Normal(), 1. - (1. - level)/2.)
    se = stderror(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end
function StatsAPI.loglikelihood(m::GLFixedEffectModel)
    # would need to change if weights are added
    y   = m.y
    mu  = m.mu
    d   = m.distribution
    ϕ = deviance(m)/length(y)
    sum(GLM.loglik_obs.(Ref(d), y, mu, 1, ϕ))
end
function StatsAPI.nullloglikelihood(m::GLFixedEffectModel)
    y      = m.y
    d      = m.distribution
    hasint = hasintercept(m.formula) || has_fe(m)
    ll  = zero(eltype(y))
    mu = hasint ? mean(y) : linkinv(m.link, zero(ll)/1)
    ϕ = nulldeviance(m)/length(y)
    sum(GLM.loglik_obs.(Ref(d), y, mu, 1, ϕ))
end
# TODO: check whether this is equal to x.gradient
StatsAPI.score(x::GLFixedEffectModel) = error("score is not yet implemented for $(typeof(x)).")

function StatsAPI.predict(x::GLFixedEffectModel)
    ("mu" ∉ names(x.augmentdf)) && error("Predicted response `mu` has not been saved. Run nlreg with :mu included in the keyword vector `save`.")
    x.augmentdf.mu
end
function StatsAPI.predict(x::GLFixedEffectModel, df::AbstractDataFrame)
    error("predict is not yet implemented for $(typeof(x)).")
end
function StatsAPI.residuals(x::GLFixedEffectModel)
    ("residuals" ∉ names(x.augmentdf)) && error("Residuals have not been saved. Run nlreg with :fe included in the keyword vector `save`.")
    x.augmentdf.residuals
end

# predict, residuals, modelresponse
# function StatsBase.predict(x::FixedEffectModel, df::AbstractDataFrame)
#     has_fe(x) && throw("predict is not defined for fixed effect models. To access the fixed effects, run `reg` with the option save = true, and access fixed effects with `fe()`")
#     cols, nonmissings = StatsModels.missing_omit(StatsModels.columntable(df), MatrixTerm(x.formula_schema.rhs))
#     new_x = modelmatrix(x.formula_schema, cols)
#     if all(nonmissings)
#         out = new_x * x.coef
#     else
#         out = Vector{Union{Float64, Missing}}(missing, size(df, 1))
#         out[nonmissings] = new_x * x.coef
#     end
#     return out
# end

# function StatsBase.residuals(x::FixedEffectModel, df::AbstractDataFrame)
#     if !has_fe(x)
#         cols, nonmissings = StatsModels.missing_omit(StatsModels.columntable(df), x.formula_schema)
#         new_x = modelmatrix(x.formula_schema, cols)
#         y = response(x.formula_schema, df)
#         if all(nonmissings)
#             out =  y -  new_x * x.coef
#         else
#             out = Vector{Union{Float64, Missing}}(missing,  size(df, 1))
#             out[nonmissings] = y -  new_x * x.coef
#         end
#         return out
#     else
#         size(x.augmentdf, 2) == 0 && throw("To access residuals in a fixed effect regression,  run `reg` with the option save = true, and then access residuals with `residuals()`")
#        residuals(x)
#    end
# end

# function StatsBase.residuals(x::FixedEffectModel)
#     !has_fe(x) && throw("To access residuals,  use residuals(x, df::AbstractDataFrame")
#     x.augmentdf.residuals
# end

function FixedEffectModels.fe(x::GLFixedEffectModel)
   !has_fe(x) && throw("fe() is not defined for fixed effect models without fixed effects")
   x.augmentdf[!, 2:size(x.augmentdf, 2)]
end


function title(x::GLFixedEffectModel)
    return "Generalized Linear Fixed Effect Model"
end

function top(x::GLFixedEffectModel)
    # make distribution and link a bit nicer
    dist = string(typeof(x.distribution))
    m = match(r"\w*",dist)
    if !isnothing(m)
        dist = m.match
    end
    link = string(typeof(x.link))
    m = match(r"\w*",link)
    if !isnothing(m)
        link = m.match
    end
    out = [
            "Distribution" sprint(show, dist, context = :compact => true);
            "Link" sprint(show, link, context = :compact => true);
            "Number of obs" sprint(show, nobs(x), context = :compact => true);
            "Degrees of freedom" sprint(show, nobs(x) - dof_residual(x), context = :compact => true);
            "Deviance" format_scientific(deviance(x));
            "Pseudo-R2" format_scientific(pseudo_r2(x));
            "Pseudo-Adj. R2" format_scientific(pseudo_adjr2(x));
            ]
    if has_fe(x)
        out = vcat(out,
            ["Iterations" sprint(show, x.iterations, context = :compact => true);
             "Converged" sprint(show, x.converged, context = :compact => true);
             ])
    end
    return out
end

function Base.show(io::IO, x::GLFixedEffectModel)
    show(io, coeftable(x))
end

function StatsAPI.coeftable(x::GLFixedEffectModel)
    ctitle = title(x)
    ctop = top(x)
    cc = coef(x)
    se = stderror(x)
    coefnms = coefnames(x)
    conf_int = confint(x)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    CoefTable2(
        hcat(cc, se, tt, ccdf.(Ref(FDist(1, dof_residual(x))), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4, ctitle, ctop)
end


##############################################################################
##
## Display Result
##
##############################################################################



## Coeftable2 is a modified Coeftable allowing for a top String matrix displayed before the coefficients.
## Pull request: https://github.com/JuliaStats/StatsBase.jl/pull/119

struct CoefTable2
    mat::Matrix
    colnms::Vector
    rownms::Vector
    pvalcol::Integer
    title::AbstractString
    top::Matrix{AbstractString}
    function CoefTable2(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0,
                        title::AbstractString = "", top::Matrix = Any[])
        nr,nc = size(mat)
        0 <= pvalcol <= nc || throw("pvalcol = $pvalcol should be in 0,...,$nc]")
        length(colnms) in [0,nc] || throw("colnms should have length 0 or $nc")
        length(rownms) in [0,nr] || throw("rownms should have length 0 or $nr")
        length(top) == 0 || (size(top, 2) == 2 || throw("top should have 2 columns"))
        new(mat,colnms,rownms,pvalcol, title, top)
    end
end


## format numbers in the p-value column
function format_scientific(pv::Number)
    return @sprintf("%.3f", pv)
end

function Base.show(io::IO, ct::CoefTable2)
    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms;
    pvc = ct.pvalcol; title = ct.title;   top = ct.top
    if length(rownms) == 0
        rownms = AbstractString[lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    if length(rownms) > 0
        rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
        else
            # if only intercept, rownms is empty collection, so previous would return error
        rnwidth = 4
    end
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(show, mat[i,j]; context=:compact => true) for i in 1:nr, j in 1:nc]
    if pvc != 0                         # format the p-values column
        for i in 1:nr
            str[i, pvc] = format_scientific(mat[i, pvc])
        end
    end
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i, j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    totalwidth = sum(widths) + rnwidth
    if length(title) > 0
        halfwidth = div(totalwidth - length(title), 2)
        println(io, " " ^ halfwidth * string(title) * " " ^ halfwidth)
    end
    if length(top) > 0
        for i in 1:size(top, 1)
            top[i, 1] = top[i, 1] * ":"
        end
        println(io, "=" ^totalwidth)
        halfwidth = div(totalwidth, 2) - 1
        interwidth = 2 +  mod(totalwidth, 2)
        for i in 1:(div(size(top, 1) - 1, 2)+1)
            print(io, top[2*i-1, 1])
            print(io, lpad(top[2*i-1, 2], halfwidth - length(top[2*i-1, 1])))
            print(io, " " ^interwidth)
            if size(top, 1) >= 2*i
                print(io, top[2*i, 1])
                print(io, lpad(top[2*i, 2], halfwidth - length(top[2*i, 1])))
            end
            println(io)
        end
    end
    println(io,"=" ^totalwidth)
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    println(io,"-" ^totalwidth)
    for i in 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
    println(io,"=" ^totalwidth)
end


##############################################################################
##
## Schema
##
##############################################################################
function StatsModels.apply_schema(t::FormulaTerm, schema::StatsModels.Schema, Mod::Type{GLFixedEffectModel}, has_fe_intercept)
    schema = StatsModels.FullRank(schema)
    if has_fe_intercept
        push!(schema.already, InterceptTerm{true}())
    end
    FormulaTerm(apply_schema(t.lhs, schema.schema, StatisticalModel),
                StatsModels.collect_matrix_terms(apply_schema(t.rhs, schema, StatisticalModel)))
end
