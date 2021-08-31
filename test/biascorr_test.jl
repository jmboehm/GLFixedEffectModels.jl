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
x = GLFixedEffectModels.nlreg(df, m, Binomial(), LogitLink(), start = [0.2], save = [:fe,:residuals])
x_afterbc = GLFixedEffectModels.bias_correction(x, df; i_symb = :SpeciesDummy, j_symb = :RandomCategorical)

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

x = GLFixedEffectModels.nlreg(df, m, Binomial(), ProbitLink(), start = [0.2], save=[:fe,:residuals])
x_afterbc = GLFixedEffectModels.bias_correction(x, df; i_symb = :SpeciesDummy, j_symb = :RandomCategorical)
@test x_afterbc.coef ≈ [4.1962783532153605] atol = 1e-4

# Test 3: Three-way Logit (I = 5, J = 6, T = 7) Network
I, J, T = 5, 6, 7
i_index = repeat(1:I,inner = J*T)
j_index = repeat(1:J,outer = I, inner = T)
t_index = repeat(1:T,outer = I*J)

# Reset rng
rng = StableRNG(1234)
data = DataFrame(i = i_index, j = j_index, t = t_index, x = rand(rng, I*J*T), y = rand(rng,Bernoulli(), I*J*T))

#= @rput data
R"""
data <- data %>%
  mutate(it = as.factor(paste(i,t))) %>%
  mutate(jt = as.factor(paste(j,t))) %>%
  mutate(ij = as.factor(paste(i,j)))
res3 <- alpaca::feglm(y ~ x | it + jt + ij, data)
res3.bc_L0 <- alpaca::biasCorr(res3,panel.structure = 'network',L = 0)
res3.bc_L3 <- alpaca::biasCorr(res3,panel.structure = 'network',L = 3)
coef_L0 <- res3.bc_L0[["coefficients"]]
coef_L3 <- res3.bc_L3[["coefficients"]]
"""
@rget coef_L0 coef_L3 
###############################
coef_L0 = -0.5478161609879237
coef_L3 = -0.6077559017896819
###############################
=#

m = GLFixedEffectModels.@formula y ~ x + GLFixedEffectModels.fe(i)*GLFixedEffectModels.fe(t) + GLFixedEffectModels.fe(j)*GLFixedEffectModels.fe(t) + GLFixedEffectModels.fe(i)*GLFixedEffectModels.fe(j)
x = GLFixedEffectModels.nlreg(data, m, Binomial(), LogitLink(), start = [0.2], save=[:fe,:residuals])
x_bc_L0 = GLFixedEffectModels.bias_correction(x, data; i_symb = :i, j_symb = :j, t_symb = :t, panel_structure = :network, L = 0)
x_bc_L3 = GLFixedEffectModels.bias_correction(x, data; i_symb = :i, j_symb = :j, t_symb = :t, panel_structure = :network, L = 3)
@test x_bc_L0.coef ≈ [-0.5478161609879237] atol = 1e-4
@test x_bc_L3.coef ≈ [-0.5950043813922534] atol = 1e-4 
# we didn't use -0.6077559017896819 because in this network specification, alpaca will perform bandwidth on a refined dataset that deletes observation(s) due to perfect classification.
# we perform the bandwidth adjustment on the original dataset.

# Test 4: Three-way Poisson, Network structure
I, J, T = 9, 9, 6
i_index = repeat(1:I,inner = J*T)
j_index = repeat(1:J,outer = I, inner = T)
t_index = repeat(1:T,outer = I*J)

# Reset rng
rng = StableRNG(1234)
data = DataFrame(i = i_index, j = j_index, t = t_index, x = rand(rng, I*J*T), y = rand(rng, Poisson(), I*J*T).+1)
# data = sort(data, [:t,:j,:i])
# using CSV
# CSV.write("test4.csv",data)
m = GLFixedEffectModels.@formula y ~ x + GLFixedEffectModels.fe(i)*GLFixedEffectModels.fe(t) + GLFixedEffectModels.fe(j)*GLFixedEffectModels.fe(t) + GLFixedEffectModels.fe(i)*GLFixedEffectModels.fe(j)
x = GLFixedEffectModels.nlreg(data, m, Poisson(), LogLink(), start = [0.2], save=[:fe,:residuals])
x_afterbc = GLFixedEffectModels.bias_correction(x, data; i_symb = :i, j_symb = :j, t_symb = :t, panel_structure = :network)
@test x_afterbc.coef ≈ [-0.0088560] atol = 1e-4

#= Test 5, example from ppml_fe_bias
using CSV
df_pois = CSV.read("PPMLFEBIAS_EXAMPLE_DATA.csv",DataFrame)
m = GLFixedEffectModels.@formula trade ~ fta + GLFixedEffectModels.fe(imp) * GLFixedEffectModels.fe(year) + GLFixedEffectModels.fe(exp) * GLFixedEffectModels.fe(year) + GLFixedEffectModels.fe(imp) * GLFixedEffectModels.fe(exp)
x = GLFixedEffectModels.nlreg(df_pois, m, Poisson(), LogLink(), start = [0.2], save=true, GLFixedEffectModels.Vcov.cluster(:imp,:exp); rho_tol = 1e-8)
 =#
