using GLFixedEffectModels

using GLM
using RDatasets, Test, Distributions
using Random

# using Alpaca

rng = MersenneTwister(1234)

df = dataset("datasets", "iris")
df.binary = zeros(Float64, size(df,1))
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesDummy = categorical(df.Species)
idx = rand(rng,1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = categorical(df.Random)

# result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
#     Binomial(),
#     fe = :SpeciesDummy
#     )
# @test StatsBase.coef(result) ≈ [-0.221486] atol = 1e-4
#
# result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
#     Binomial(),
#     fe = :SpeciesDummy,
#     start = [0.2], trace = 2
#     )
# # glm
# gm1 = fit(GeneralizedLinearModel, @formula(binary ~ SepalWidth),
#               df, Poisson())


# LOGIT ------------------------------------------------------------------

# One FE, Logit
m = GLFixedEffectModels.@formula binary ~ SepalWidth + GLFixedEffectModels.fe(SpeciesDummy)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), start = [0.2] )
@test coef(x) ≈ [8.144352] atol = 1e-4
# Two FE, Logit
m = GLFixedEffectModels.@formula binary ~ SepalWidth + GLFixedEffectModels.fe(SpeciesDummy) + GLFixedEffectModels.fe(RandomCategorical)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), start = [0.2] )
@test coef(x) ≈ [8.3927666] atol = 1e-4
# make sure that showing works
@show x

# VCov
m = GLFixedEffectModels.@formula binary ~ SepalWidth + GLFixedEffectModels.fe(SpeciesDummy)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), GLFixedEffectModels.Vcov.simple() , start = [0.2] )
@test vcov(x) ≈ [3.585929] atol = 1e-4
m = GLFixedEffectModels.@formula binary ~ SepalWidth + PetalLength + GLFixedEffectModels.fe(SpeciesDummy)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), GLFixedEffectModels.Vcov.robust() , start = [0.2, 0.2] )
@test vcov(x) ≈ [ 2.28545  0.35542; 0.35542  3.65724] atol = 1e-4
m = GLFixedEffectModels.@formula binary ~ SepalWidth + PetalLength + GLFixedEffectModels.fe(SpeciesDummy)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), GLFixedEffectModels.Vcov.cluster(:SpeciesDummy) , start = [0.2, 0.2] )
@test vcov(x) ≈ [ 1.48889   0.464914; 0.464914  3.07176 ] atol = 1e-4
m = GLFixedEffectModels.@formula binary ~ SepalWidth + PetalLength + GLFixedEffectModels.fe(SpeciesDummy)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), GLFixedEffectModels.Vcov.cluster(:SpeciesDummy,:RandomCategorical) , start = [0.2, 0.2] )
@test vcov(x) ≈ [ 1.32851  -1.11561; -1.11561  6.55242] atol = 1e-4

# Save fe
m = GLFixedEffectModels.@formula binary ~ SepalWidth + GLFixedEffectModels.fe(SpeciesDummy)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), GLM.LogitLink(), start = [0.2] , save = :fe )
fes = Float64[]
for c in levels(df.SpeciesDummy)
    push!(fes, x.augmentdf[df.SpeciesDummy .== c, :fe_SpeciesDummy][1])
end
@test fes[1] ≈ -28.3176042490 atol = 1e-4
@test fes[2] ≈ -17.507252832 atol = 1e-4
@test fes[3] ≈ -17.851658274 atol = 1e-4


result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth ),
    Binomial(),
    fe = :SpeciesDummy,
    start = [0.2], trace = 2)

# For comparison with Alpaca.jl
# result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth + PetalLength),
#     Binomial(),
#     fe = :SpeciesDummy,
#     start = [0.2, 0.2], trace = 2, vcov = :robust)
# result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth + PetalLength),
#     Binomial(),
#     fe = :SpeciesDummy,
#     start = [0.2, 0.2], trace = 2, vcov = :(cluster(SpeciesDummy)))
# result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth + PetalLength),
#     Binomial(),
#     fe = :SpeciesDummy,
#     start = [0.2, 0.2], trace = 2, vcov = :(cluster(SpeciesDummy + RandomCategorical)))

# POISSON ----------------------------------------------------------------

rng = MersenneTwister(1234)
N = 1_000_000
K = 100
id1 = rand(rng, 1:(round(Int64,N/K)), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1_noncat = id1, id2_noncat = id2, x1 = x1, x2 = x2, y = y)
df.id1 = categorical(id1)
df.id2 = categorical(id2)

# One FE, Poisson
m = GLFixedEffectModels.@formula y ~ x1 + x2 + GLFixedEffectModels.fe(id1)
x = GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )
@test coef(x) ≈ [3.008021723434; 2.01367439742] atol = 1e-4
# Two FE, Poisson
m = GLFixedEffectModels.@formula y ~ x1 + x2 + GLFixedEffectModels.fe(id1) +  GLFixedEffectModels.fe(id2)
x = GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )
@test coef(x) ≈ [3.008437743; 2.00983105012] atol = 1e-4



#
# using FixedEffectModels, CSV, DataFrames, LinearAlgebra, Test
# df = CSV.read(joinpath(dirname(pathof(FixedEffectModels)), "../dataset/Cigar.csv"))
# df.id1 = df.State
# df.id2 = df.Year
# df.pid1 = categorical(df.id1)
# df.pid2 = categorical(df.id2)
#
# df.mid1 = div.(df.id1, Ref(10))
# df.y = df.Sales
# df.x1 = df.Price
# df.z1 = df.Pimin
# df.x2 = df.NDI
# df.w = df.Pop
#
#
# cd("/Users/jboehm/Dropbox/Github/GLFixedEffectModels4.jl")
# using FixedEffects, DataFrames
# include("src/GLFixedEffectModels4.jl")
# using LinearAlgebra
#
# e = ones(Float64, 100)
# # 1.0, 0.0, 1.0, 0.0, ...
# for i = 1:100
#     if mod(i,2) == 0
#         e[i] = 0.0
#     end
# end
# e = e .- 0.5
# D_cat = ones(Int64,100)
# D_cat[51:100] .= 2
# D = zeros(Float64, 100, 2)
# D[1:50,1] .= 1.0
# D[51:100,2] .= 1.0
# alpha = [1.0;2.0]
# beta = 3.0 .* ones(Float64, 1,1)
# X = collect(0.0:0.1:9.9)
# yplain = D * alpha .+ e
# yfull = D * alpha .+ X * beta .+ e
# df = DataFrame(yplain = yplain, yfull = vec(yfull), X = X, e = e, d = D_cat)
# df.d = categorical(df.d)
#
# wt = GLFixedEffectModels4.Weights(GLFixedEffectModels4.Ones{Float64}(100))
#
# # no weights
# m = GLFixedEffectModels4.@formula yplain ~ X + GLFixedEffectModels4.fe(d)
# fes, ids, ff = GLFixedEffectModels4.parse_fixedeffect(df, m )
# #fes = FixedEffect[_subset(fe, esample) for fe in fes]
# feM = AbstractFixedEffectSolver{Float64}(fes, wt, Val{:lsmr})
# eps_est, b, c = FixedEffects.solve_residuals!(yplain, feM; maxiter = 1000, tol = 1e-8)
# beta_est, b, c = FixedEffects.solve_coefficients!(yplain, feM; maxiter = 1000, tol = 1e-8)
#
# # -------------------------
#
# e = ones(Float64, 100)
# # 1.0, 0.0, 1.0, 0.0, ...
# for i = 1:100
#     if mod(i,2) == 0
#         e[i] = 0.0
#     end
# end
# e = e .- 0.5
# D_cat = ones(Int64,100)
# D_cat[51:100] .= 2
# D = zeros(Float64, 100, 2)
# D[1:50,1] .= 1.0
# D[51:100,2] .= 1.0
# alpha = [1.0;2.0]
# beta = 3.0 .* ones(Float64, 1,1)
# X = collect(0.0:0.1:9.9)
# yplain = D * alpha .+ e
# yfull = D * alpha .+ X * beta .+ e
# df = DataFrame(yplain = yplain, yfull = vec(yfull), X = X, e = e, d = D_cat)
# df.d = categorical(df.d)
#
# # specification with W
# w = collect(0.1:0.1:10.0)
# sqrtw = sqrt.(w)
# df.sqrtw = sqrtw
# df.yw = df.sqrtw .* df.yplain
#
# wt = GLFixedEffectModels4.Weights(1.0 ./ sqrtw)
# m = GLFixedEffectModels4.@formula yw ~ X + GLFixedEffectModels4.fe(d)
# fes, ids, ff = GLFixedEffectModels4.parse_fixedeffect(df, m )
# feM = AbstractFixedEffectSolver{Float64}(fes, wt, Val{:lsmr})
# eps_est, b, c = FixedEffects.solve_residuals!(df.yw, feM; maxiter = 1000, tol = 1e-8)
# beta_est, b, c = FixedEffects.solve_coefficients!(df.yw, feM; maxiter = 1000, tol = 1e-8)
# eps_est .* sqrtw
#
# # 3
#
# e = ones(Float64, 100)
# # 1.0, 0.0, 1.0, 0.0, ...
# for i = 1:100
#     if mod(i,2) == 0
#         e[i] = 0.0
#     end
# end
# e = e .- 0.5
# D_cat = ones(Int64,100)
# D_cat[51:100] .= 2
# D = zeros(Float64, 100, 2)
# D[1:50,1] .= 1.0
# D[51:100,2] .= 1.0
# alpha = [1.0;2.0]
# beta = 3.0 .* ones(Float64, 1,1)
# X = collect(0.0:0.1:9.9)
# w = collect(0.1:0.1:10.0)
# sqrtw = sqrt.(w)
# y = D * alpha .+ e
# #yfull = D * alpha .+ X * beta .+ e
# df = DataFrame(y = vec(y), X = X, e = e, d = D_cat)
# df.d = categorical(df.d)
#
# # specification with W
# df.sqrtw = sqrtw
#
# wt = GLFixedEffectModels4.Weights(w)
# m = GLFixedEffectModels4.@formula y ~ X + GLFixedEffectModels4.fe(d)
# fes, ids, ff = GLFixedEffectModels4.parse_fixedeffect(df, m )
# feM = AbstractFixedEffectSolver{Float64}(fes, wt, Val{:lsmr})
# eps_est, b, c = FixedEffects.solve_residuals!(df.y, feM; maxiter = 1000, tol = 1e-8)
# beta_est, b, c = FixedEffects.solve_coefficients!(df.y, feM; maxiter = 1000, tol = 1e-8)
#
# # get the estimator manually
# crossx = cholesky!(Symmetric(D' * Diagonal(w) * D))
# beta_est_manual = crossx \ (D' * Diagonal(w) * y)
# eps_est_manual = y - D*beta_est_manual
#
#
# crossx = cholesky!(Symmetric(Xhat' * Xhat))
# #crossx = Symmetric(Xhat' * Xhat)
#
# beta_update = crossx \ (Xhat' * nudemean)
