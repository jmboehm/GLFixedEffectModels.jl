using GLFixedEffectModels

using Distributions, CategoricalArrays
using RDatasets, Test, Random
using StableRNGs
using GLM

# using Alpaca

rng = StableRNG(1234)

df = dataset("datasets", "iris")
df.binary = zeros(Float64, size(df,1))
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesDummy = string.(df.Species)
idx = rand(rng,1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = df.Random

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

# PROBIT ------------------------------------------------------------------
# One FE, Probit
m = @formula binary ~ SepalWidth + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), GLM.ProbitLink(), start = [0.2], save = [:fe])
@test x.coef ≈ [4.7793003788996895] atol = 1e-4

# Two FE, Probit
m = @formula binary ~ SepalWidth + fe(SpeciesDummy) + fe(RandomCategorical)
x = nlreg(df, m, Binomial(), ProbitLink(), start = [0.2] )
@test x.coef ≈ [4.734428099238226] atol = 1e-4
# test target value obtained from alpaca::feglm with `dev.tol <- 1e-10`


# LOGIT ------------------------------------------------------------------

# One FE, Logit
m = @formula binary ~ SepalWidth + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), start = [0.2] )
@test coef(x) ≈ [8.144352] atol = 1e-4

# Two FE, Logit
m = @formula binary ~ SepalWidth + fe(SpeciesDummy) + fe(RandomCategorical)
x = nlreg(df, m, Binomial(), LogitLink(), start = [0.2] )
@test coef(x) ≈ [8.05208] atol = 1e-4
# make sure that showing works
@show x



# VCov
m = @formula binary ~ SepalWidth + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), Vcov.simple() , start = [0.2] )
# result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
#     Binomial(),
#     fe = :SpeciesDummy,
#     start = [0.2], trace = 2)
@test vcov(x) ≈ [3.585929] atol = 1e-4
m = @formula binary ~ SepalWidth + PetalLength + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), Vcov.robust() , start = [0.2, 0.2] )
@test vcov(x) ≈ [ 2.28545  0.35542; 0.35542  3.65724] atol = 1e-4
m = @formula binary ~ SepalWidth + PetalLength + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), Vcov.cluster(:SpeciesDummy) , start = [0.2, 0.2] )
@test vcov(x) ≈ [ 1.48889   0.464914; 0.464914  3.07176 ] atol = 1e-4
m = @formula binary ~ SepalWidth + PetalLength + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), Vcov.cluster(:SpeciesDummy,:RandomCategorical) , start = [0.2, 0.2] )
@test vcov(x) ≈ [0.43876 0.315690; 0.315690 1.59676] atol = 1e-4

# Save fe
m = @formula binary ~ SepalWidth + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), start = [0.2] , save = [:fe] )
fes = Float64[]
for c in levels(df.SpeciesDummy)
    push!(fes, x.augmentdf[df.SpeciesDummy .== c, :fe_SpeciesDummy][1])
end
@test fes[1] ≈ -28.3176042490 atol = 1e-4
@test fes[2] ≈ -17.507252832 atol = 1e-4
@test fes[3] ≈ -17.851658274 atol = 1e-4


# loglikelihood and psuedo r2

m = @formula binary ~ SepalWidth + fe(Species)
x = nlreg(df, m, Binomial(), LogitLink(), Vcov.cluster(:Species) , start = [0.2] )
@test loglikelihood(x) ≈ -27.58750 atol = 1e-4
@test pseudo_r2(x) ≈ 0.64518 atol = 1e-4
@test pseudo_adjr2(x) ≈ 0.60660 atol = 1e-4

m = @formula binary ~ SepalWidth + PetalLength + fe(Species)
x = nlreg(df, m, Binomial(), LogitLink(), Vcov.cluster(:Species) , start = [0.2, 0.2] )
@test loglikelihood(x) ≈ -21.35416 atol = 1e-4
@test pseudo_r2(x) ≈ 0.72535 atol = 1e-4
@test pseudo_adjr2(x) ≈ 0.67391 atol = 1e-4

m = @formula binary ~ SepalWidth + fe(Species)
x = nlreg(df, m, Poisson(), LogLink(), Vcov.cluster(:Species) , start = [0.2] )
@test loglikelihood(x) ≈ -137.19554 atol = 1e-4
@test pseudo_r2(x) ≈ 0.06232 atol = 1e-4
@test pseudo_adjr2(x) ≈ 0.04182 atol = 1e-4

m = @formula binary ~ SepalWidth + PetalLength + fe(Species)
x = nlreg(df, m, Poisson(), LogLink(), Vcov.cluster(:Species) , start = [0.2, 0.2] )
@test loglikelihood(x) ≈ -137.19365 atol = 1e-4
@test pseudo_r2(x) ≈ 0.06234 atol = 1e-4
@test pseudo_adjr2(x) ≈ 0.03500 atol = 1e-4


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

rng = StableRNG(1234)
N = 1_000_000
K = 100
id1 = rand(rng, 1:(round(Int64,N/K)), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1_noncat = id1, id2_noncat = id2, x1 = x1, x2 = x2, y = y)
df.id1 = id1
df.id2 = id2

# One FE, Poisson
m = @formula y ~ x1 + x2 + fe(id1)
x = nlreg(df, m, Poisson(), LogLink() , start = [0.2;0.2] )
# result = Alpaca.feglm(df, Alpaca.@formula(y ~ x1 + x2),
#     Poisson(),
#     fe =:(id1)
#     )
@test coef(x) ≈ [2.9912251435680237; 2.002088081633829] atol = 1e-4
# Two FE, Poisson
m = @formula y ~ x1 + x2 + fe(id1) +  fe(id2)
x = nlreg(df, m, Poisson(), LogLink() , start = [0.2;0.2] )
# result = Alpaca.feglm(df, Alpaca.@formula(y ~ x1 + x2),
#     Poisson(),
#     fe =:(id1 + id2)
#     )
@test coef(x) ≈ [ 2.987722385633501; 2.0056217356569155] atol = 1e-4

# Separation: based on Sergio Correia's example (https://github.com/sergiocorreia/ppmlhdfe/blob/master/guides/separation_primer.md) but with logit (easier to generate)
rng = StableRNG(1234)
df_sep = DataFrame(y = [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0];rand(rng,[0.0,1.0],500)], x1 = [[1, 1, 0, 0, 0, 0];zeros(Float64,500)], x = collect(1.0:(6.0+500.0)))
m = @formula y ~ x + fe(x1)
try
    # this should fail
    local x = nlreg(df_sep, m, Binomial(), LogitLink() , start = [0.1], separation = Symbol[], separation_mu_lbound=1e-10, separation_mu_ubound=1.0-1e-10, verbose=true, rho_tol=1e-12 )
catch ex
    @test !isnothing(ex)
end
# with cutoff on mu, it converges
try
    # this should pass
    local x = nlreg(df_sep, m, Binomial(), LogitLink() , start = [0.1], separation = [:mu], separation_mu_lbound=1e-10, separation_mu_ubound=1.0-1e-10, verbose=true, rho_tol=1e-12 )
    @test x.coef ≈ [-0.0005504145168443688] atol = 1e-4
catch ex
    @test isnothing(ex)
end

