struct VcovDataGLM{T, N} <: RegressionModel
    modelmatrix::Matrix{Float64}       # X
    crossmodelmatrix::T                    # X'X in the simplest case. Can be Matrix but preferably Factorization
    residuals::Array{Float64, N}      # vector or matrix of residuals (matrix in the case of IV, residuals of Xendo on (Z, Xexo))
    dof_residual::Int
end
StatsAPI.modelmatrix(x::VcovDataGLM) = x.modelmatrix
StatsAPI.crossmodelmatrix(x::VcovDataGLM) = x.crossmodelmatrix
StatsAPI.residuals(x::VcovDataGLM) = x.residuals
StatsAPI.dof_residual(x::VcovDataGLM) = x.dof_residual

# with clusters, the standard StatsBase.vcov works

function StatsAPI.vcov(x::VcovDataGLM, ::Vcov.RobustCovariance)
    A = inv(crossmodelmatrix(x))
    C = modelmatrix(x) .* residuals(x)
    B = C' * C
    return Symmetric(A * B * A)
end

function StatsAPI.vcov(x::VcovDataGLM, ::Vcov.SimpleCovariance)
	return Symmetric(inv(crossmodelmatrix(x)))
end
