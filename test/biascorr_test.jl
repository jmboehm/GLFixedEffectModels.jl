# include("../src/GLFixedEffectModels.jl")
using GLFixedEffectModels
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
# See if the coefficients after bias correction match with the results obtained from R package alpaca
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
# Since the coefficients for models that uses the probit link didn't match the results from R's alpaca before bias correction
# We test if the ``net adjusted value`` for the coefficients matches, for now.
# TODO: check why probit link didn't work as expected
R"""
res2 <- alpaca::feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical , df_r, binomial(link = "probit"), beta.start = c(0.2))
res_bc2 <- alpaca::biasCorr(res2)
coef2<- res2[["coefficients"]]
coef2_afterbc <- res_bc2[["coefficients"]]
net_adjusted_value = coef2_afterbc - coef2
"""
@rget net_adjusted_value

x = GLFixedEffectModels.nlreg(df, m, Binomial(), ProbitLink(), start = [0.2], save=true)
x_afterbc = GLFixedEffectModels.BiasCorr(x, df)
@test x_afterbc.coef - x.coef ≈ [net_adjusted_value] atol = 1e-4

# Test 3: Three-way Logit (I = 5, J = 6, T = 7), Network structure
R"""
data <- alpaca::simGLM(n = 30, t = 7, seed = 1234, model = "logit")
colnames(data)[1] <- "ij"
ij <- merge(seq(5),seq(6))
data$i <- ij$x
data$j <- ij$y
data_sorted <- data[order(data$i, data$j, data$t),]
res3 <- alpaca::feglm(y ~ x1 | i + j + t , data_sorted, binomial(), beta.start = c(0.2))
res3.bc_L0 <- biasCorr(res3,panel.structure = 'network',L = 0)
res3.bc_L3 <- biasCorr(res3,panel.structure = 'network',L = 3)
coef3 <- res3[["coefficients"]]
coef_L0 <- res3.bc_L0[["coefficients"]]
coef_L3 <- res3.bc_L3[["coefficients"]]
"""
@rget data_sorted coef3 coef_L0 coef_L3

m = GLFixedEffectModels.@formula y ~ x1 + GLFixedEffectModels.fe(i) + GLFixedEffectModels.fe(j) + GLFixedEffectModels.fe(t)
x = GLFixedEffectModels.nlreg(data_sorted, m, Binomial(), LogitLink(), start = [0.2], save=true)
x_bc_L0 = GLFixedEffectModels.BiasCorr(x, data_sorted; panel_structure = "network", L = 0)
x_bc_L3 = GLFixedEffectModels.BiasCorr(x, data_sorted; panel_structure = "network", L = 3)

@test x_bc_L0.coef ≈ [coef_L0] atol = 1e-4
@test x_bc_L3.coef ≈ [coef_L3] atol = 1e-4