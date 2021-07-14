# include("../src/GLFixedEffectModels.jl")
using GLFixedEffectModels
using Distributions, CategoricalArrays
using RDatasets, Test, Random
using StableRNGs

using GLM: LogitLink, ProbitLink, LogLink
# using RCall

rng = StableRNG(1234)

df = dataset("datasets", "iris")
df.binary = zeros(Float64, size(df,1))
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesDummy = string.(df.Species)
idx = rand(rng,1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = df.Random

# Test 1: Two-way Logit
# See if the coefficients after bias correction match with the results obtained from R package alpaca

#= R"""
res1 <- alpaca::feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical , df_r, binomial(), beta.start = c(0.2))
res_bc1 <- alpaca::biasCorr(res1)
coef1 <- res_bc1[["coefficients"]]
"""
@rget coef1
###############################
coef1 = 7.214197357443702
###############################
=#

m = GLFixedEffectModels.@formula binary ~ SepalWidth + GLFixedEffectModels.fe(SpeciesDummy) + GLFixedEffectModels.fe(RandomCategorical)
x = GLFixedEffectModels.nlreg(df, m, Binomial(), LogitLink(), start = [0.2], save=true)
x_afterbc = GLFixedEffectModels.BiasCorr(x, df; i_symb = :SpeciesDummy, j_symb = :RandomCategorical)

@test x_afterbc.coef ≈ [7.214197357443702] atol = 1e-4

# Test 2: Two-way Probit

#=R"""
control <- do.call(feglmControl,list())
control[["dev.tol"]] <- 1e-10
res2 <- alpaca::feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical , df_r, binomial(link = "probit"), beta.start = c(0.2), control = control)
res_bc2 <- alpaca::biasCorr(res2)
coef2_afterbc <- res_bc2[["coefficients"]]
"""
@rget coef2_afterbc
###############################
coef2_afterbc = 4.1962783532153605
###############################
=#

x = GLFixedEffectModels.nlreg(df, m, Binomial(), ProbitLink(), start = [0.2], save=true)
x_afterbc = GLFixedEffectModels.BiasCorr(x, df; i_symb = :SpeciesDummy, j_symb = :RandomCategorical)
@test x_afterbc.coef ≈ [4.1962783532153605] atol = 1e-4

#= Test 3: Three-way Logit (I = 5, J = 6, T = 7), Network structure, (NEED TO BE REDONE)
I, J, T = 5, 6, 7
i_index = repeat(1:I,inner = J*T)
j_index = repeat(1:J,outer = I, inner = T)
t_index = repeat(1:T,outer = I*J)

# Reset rng
rng = StableRNG(1234)
data = DataFrame(i = i_index, j = j_index, t = t_index, x = rand(rng, I*J*T), y = rand(rng,Bernoulli(), I*J*T))

#= @rput data
R"""
res3 <- alpaca::feglm(y ~ x | i + j + t , data, binomial(), beta.start = c(0.2))
res3.bc_L0 <- alpaca::biasCorr(res3,panel.structure = 'network',L = 0)
res3.bc_L3 <- alpaca::biasCorr(res3,panel.structure = 'network',L = 3)
coef_L0 <- res3.bc_L0[["coefficients"]]
coef_L3 <- res3.bc_L3[["coefficients"]]
"""
@rget coef_L0 coef_L3 
###############################
coef_L0 = -0.8958868087842135
coef_L3 = -0.8845864617809618
###############################
=#

m = GLFixedEffectModels.@formula y ~ x + GLFixedEffectModels.fe(i) + GLFixedEffectModels.fe(j) + GLFixedEffectModels.fe(t)
x = GLFixedEffectModels.nlreg(data, m, Binomial(), LogitLink(), start = [0.2], save=true)
x_bc_L0 = GLFixedEffectModels.BiasCorr(x, data; panel_structure = "network", L = 0)
x_bc_L3 = GLFixedEffectModels.BiasCorr(x, data; panel_structure = "network", L = 3)
@test x_bc_L0.coef ≈ [-0.8958868087842135] atol = 1e-4
@test x_bc_L3.coef ≈ [-0.8845864617809618] atol = 1e-4
=#
# Test 4: Three-way Poisson, Network structure
I, J, T = 9, 9, 6
i_index = repeat(1:I,inner = J*T)
j_index = repeat(1:J,outer = I, inner = T)
t_index = repeat(1:T,outer = I*J)

# Reset rng
rng = StableRNG(1234)
data = DataFrame(i = i_index, j = j_index, t = t_index, x = rand(rng, I*J*T), y = rand(rng, Poisson(), I*J*T))
using CSV
CSV.write("test4.csv",data)
m = GLFixedEffectModels.@formula y ~ x + GLFixedEffectModels.fe(i)*GLFixedEffectModels.fe(t) + GLFixedEffectModels.fe(j)*GLFixedEffectModels.fe(t) + GLFixedEffectModels.fe(i)*GLFixedEffectModels.fe(j)
x = GLFixedEffectModels.nlreg(data, m, Poisson(), LogLink(), start = [0.2], save=true)

x_afterbc = GLFixedEffectModels.BiasCorr(x, data; i_symb = :i, j_symb = :j, t_symb = :t, panel_structure="network")
@show x_afterbc










#=
using CSV
df_pois = CSV.read("PPMLFEBIAS_EXAMPLE_DATA.csv",DataFrame)
m = GLFixedEffectModels.@formula trade ~ fta + GLFixedEffectModels.fe(imp) * GLFixedEffectModels.fe(year) + GLFixedEffectModels.fe(exp) * GLFixedEffectModels.fe(year) + GLFixedEffectModels.fe(imp) * GLFixedEffectModels.fe(exp)
x = GLFixedEffectModels.nlreg(df_pois, m, Poisson(), LogLink(), start = [0.2], save=true; rho_tol = 1e-8)

link = x.link
y = df_pois[x.esample,x.yname]
residuals = x.augmentdf.residuals
η = y - residuals
μ = GLM.linkinv.(Ref(link),η) # λ equivalent in the paper and ppml_fe_bias program
μη = GLM.mueta.(Ref(link),η)
score = x.gradient
hessian = x.hessian

fes = select(x.augmentdf, Not(:residuals)) =#