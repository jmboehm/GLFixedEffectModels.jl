
include("src/GLFixedEffectModels.jl")

using CSV, DataFrames, LinearAlgebra, Test
using GLM, Distributions
df = CSV.read("dataset/Cigar.csv")
df[:id1] = df[:State]
df[:id2] = df[:Year]
df[:pid1] = categorical(df[:id1])
df[:ppid1] = categorical(div.(df[:id1], 10))
df[:pid2] = categorical(df[:id2])
df[:y] = df[:Sales]
df[:x1] = df[:Price]
df[:z1] = df[:Pimin]
df[:x2] = df[:NDI]
df[:w] = df[:Pop]

df[:logSales] = log.(df[:Sales])
df[:logPrice] = log.(df[:Price])

df[:Sales2] = (df[:Sales] .- mean(df[:Sales]))./std(df[:Sales])
df[:Sales2] = df[:Sales2] .- minimum(df[:Sales2])

##############################################################################
##
## coefficients
##
##############################################################################

# simple
m = GLFixedEffectModels.@model Sales2 ~ x1 fe = pid1
x = GLFixedEffectModels.reg(df, m; distribution = Poisson(), link = LogLink())

f=m.f;
fe= :pid1
link = :poisson
vcov  = :(simple())
weights = nothing
subset = nothing
maxiter = 10000
contrasts = Dict()
tol = 1e-8
df_add = 0
save = false
method = :lsmr
drop_singletons = true
link = LogLink()

m = FixedEffectModels.@model y ~ x1 fe = pid1
x = FixedEffectModels.reg(df, m)

probit = glm(@formula(y ~ x1), df, Poisson(), LogLink())

probit = lm(@formula(logSales ~ logPrice), df)
probit = glm(@formula(logSales ~ logPrice), df, Poisson(), LogLink())