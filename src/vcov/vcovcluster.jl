struct ClusterCovariance{T} <: CovarianceEstimator
    clusters::T
end

cluster(x::Symbol) = ClusterCovariance((x,))
cluster(args...) = ClusterCovariance(args)

function group(args...)
    v = categorical(args[1])
    if length(args) == 1
        refs = v.refs
    else
        refs, ngroups = convert(Vector{UInt}, v.refs), length(levels(v))
        for j = 2:length(args)
            v = categorical(args[j])
            refs, ngroups = pool_combine!(refs, v, ngroups)
        end
    end
    factorize!(refs)
end
function pool_combine!(refs::Array{T, N}, dv::CategoricalVector, ngroups::Integer) where {T, N}
    for i in 1:length(refs)
        # if previous one is NA or this one is NA, set to NA
        refs[i] = (dv.refs[i] == 0 || refs[i] == zero(T)) ? zero(T) : refs[i] + (dv.refs[i] - 1) * ngroups
    end
    return refs, ngroups * length(levels(dv))
end

function factorize!(refs::Vector{T}) where {T}
    uu = sort!(unique(refs))
    has_missing = uu[1] == 0
    dict = Dict{T, Int}(zip(uu, (1-has_missing):(length(uu)-has_missing)))
    newrefs = zeros(UInt32, length(refs))
    for i in 1:length(refs)
         newrefs[i] = dict[refs[i]]
    end
    Tout = has_missing ? Union{Int, Missing} : Int
    CategoricalArray{Tout, 1}(newrefs, CategoricalPool(collect(1:(length(uu)-has_missing))))
end

completecases(df::AbstractDataFrame, v::ClusterCovariance) = DataFrames.completecases(df, collect(v.clusters))

function materialize(df::AbstractDataFrame, v::ClusterCovariance)
    ClusterCovariance(NamedTuple{v.clusters}(ntuple(i -> group(df[!, v.clusters[i]]), length(v.clusters))))
end

function df_FStat(x::RegressionModel, v::ClusterCovariance, ::Bool)
    minimum((length(levels(x)) for x in values(v.clusters))) - 1
end

# based on FixedEffectModels.jl
function S_hat(x::VcovData, v::ClusterCovariance)
    # Cameron, Gelbach, & Miller (2011): section 2.3
    dim = size(x.gradient, 2) * size(residuals(x), 2)
    S = zeros(dim, dim)
    G = typemax(Int)
    for c in combinations(keys(v.clusters))
        # no need for group in case of one fixed effect, since was already done in VcovMethod
        f = (length(c) == 1) ? v.clusters[c[1]] : group((v.clusters[var] for var in c)...)
        # capture length of smallest dimension of multiway clustering in G
        G = min(G, length(f.pool))
        S += (-1)^(length(c) - 1) * helper_cluster(x.gradient, f)
    end
    # i use the same finite-sample adjustment as what CGM recommend for the linear case:
    # scale total vcov estimate by ((N-1)/(N-K)) * (G/(G-1))
    # another option would be to adjust each matrix given by helper_cluster by number of its categories
    # both methods are presented in Cameron, Gelbach and Miller (2011), section 2.3
    # without this small sample correction, the Vcov matrix equals the one in alpaca
    rmul!(S, (size(x.gradient, 1) - 1) / dof_residual(x) * G / (G - 1))
end

function helper_cluster(G::Matrix, f::CategoricalVector)
    G2 = zeros(eltype(G), length(f.pool), size(G, 2) )
    index = 0
    for j in 1:size(G, 2)
        index += 1
        @inbounds @simd for i in 1:size(G, 1)
            G2[f.refs[i], index] += G[i, j]
        end
    end
    return Symmetric(G2' * G2)
end

function StatsBase.vcov(x::VcovData, v::ClusterCovariance)
    A = inv(cholesky(x.hessian))
    pinvertible(Symmetric(A * S_hat(x, v) * A))
end
