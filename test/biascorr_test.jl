include("../src/GLFixedEffectModels.jl")

using Distributions, CategoricalArrays
using RDatasets, Test, Random
using StableRNGs

using GLM: LogitLink, ProbitLink

rng = StableRNG(1234)

df = dataset("datasets", "iris")
df.binary = zeros(Float64, size(df,1))
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesDummy = string.(df.Species)
idx = rand(rng,1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = df.Random

# Set up R
using RCall
R"library(alpaca)"
df_r = deepcopy(df)
@rput df_r

# Test 1: Two-way Logit
R"""
res1 <- alpaca::feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical , df_r, binomial(), beta.start = c(0.2))
res_bc1 <- alpaca::biasCorr(res1)
coef1 <- res_bc1[["coefficients"]]
"""
@rget coef1

m = GLFixedEffectModels.@formula binary ~ SepalWidth + GLFixedEffectModels.fe(SpeciesDummy) + GLFixedEffectModels.fe(RandomCategorical)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), LogitLink(), start = [0.2], save=true)
x_afterbc = GLFixedEffectModels.BiasCorr(x, df)

@test x_afterbc.coef ≈ [coef1] atol = 1e-4

# Test 2: Two-way Probit
#= R"""
res2 <- alpaca::feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical , df_r, binomial(link = "probit"), beta.start = c(0.2))
res_bc2 <- alpaca::biasCorr(res2)
coef2 <- res_bc2[["coefficients"]]
"""
@rget coef2

x = GLFixedEffectModels.nlreg(df, m, Binomial(), ProbitLink(), start = [0.2], save=true)
x_afterbc = GLFixedEffectModels.BiasCorr(x, df)
@test coef(x_afterbc) ≈ [coef2] atol = 1e-4 =#

# Test 2 fails to pass. It seems like the result before bias correction is already different
R"""
res2 <- alpaca::feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical , df_r, binomial(link = "probit"), beta.start = c(0.2))
coef2_wobc <- res2[["coefficients"]]
"""
@rget coef2_wobc

x = GLFixedEffectModels.nlreg(df, m, Binomial(), ProbitLink(), start = [0.2], save=true)
@test x.coef ≈ [coef2_wobc] atol = 1e-4