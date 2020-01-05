
#include("../src/GLFixedEffectModels.jl")

using DataFrames, GLM, Random
using GLFixedEffectModels
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

rng = MersenneTwister(1234)
N = 1_000_000
K = 100
id1 = rand(rng, 1:(N/K), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1 = categorical(string.(id1)), id1_noncat = id1, id2 = categorical(string.(id2)), x1 = x1, x2 = x2, y = y)



# Benchmark GLFixedEffectModels

# One FE
m = @formula y ~ x1 + x2 + fe(id1)
@benchmark x = nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )
# BenchmarkTools.Trial:
#   memory estimate:  1.29 GiB
#   allocs estimate:  6144
#   --------------
#   minimum time:     1.953 s (6.00% GC)
#   median time:      2.255 s (7.11% GC)
#   mean time:        2.225 s (10.03% GC)
#   maximum time:     2.529 s (15.10% GC)
#   --------------
#   samples:          10
#   evals/sample:     1

# Two FE
m = GLFixedEffectModels.@formula y ~ x1 + x2 + GLFixedEffectModels.fe(id1) +  GLFixedEffectModels.fe(id2)
@benchmark x = GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )

# Set up R
using RCall
R"library(alpaca)"
R"library(fixest)"
df_r = deepcopy(df)
@rput df_r

# Benchmark Alpaca
function runme()
    R"res <- alpaca::feglm(y ~ x1 + x2  | id1 , df_r, poisson(), beta.start = c(0.2,0.2))"
end
@benchmark runme()
# BenchmarkTools.Trial:
#   memory estimate:  1.09 KiB
#   allocs estimate:  32
#   --------------
#   minimum time:     5.695 s (0.00% GC)
#   median time:      6.327 s (0.00% GC)
#   mean time:        6.552 s (0.00% GC)
#   maximum time:     8.360 s (0.00% GC)
#   --------------
#   samples:          10
#   evals/sample:     1


# Benchmark fixest
function runme()
    R"res <- fixest::feglm(y ~ x1 + x2  | id1, df_r, family = \"poisson\", nthreads = 1, start = c(0.2,0.2))"
end
@benchmark runme()
# BenchmarkTools.Trial:
#   memory estimate:  1.09 KiB
#   allocs estimate:  32
#   --------------
#   minimum time:     1.839 s (0.00% GC)
#   median time:      3.827 s (0.00% GC)
#   mean time:        3.607 s (0.00% GC)
#   maximum time:     7.438 s (0.00% GC)
#   --------------
#   samples:          10
#   evals/sample:     1
# clear everything in R
R"rm(list = ls())"
