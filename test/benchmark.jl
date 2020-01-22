
#include("../src/GLFixedEffectModels.jl")

# Benchmark on Intel i7-6700 with 4 cores @ 3.40 GHz, 64GB RAM
# running Windows 10 Enterprise, Julia 1.3.1, CUDA 10.2, R 3.6.2
# fixest 0.2.1

using DataFrames, GLM, Random
using GLFixedEffectModels
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

rng = MersenneTwister(1234)
N = 2_000_000
K = 100
id1 = rand(rng, 1:(N/K), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1 = categorical(string.(id1)), id1_noncat = id1, id2 = categorical(string.(id2)), x1 = x1, x2 = x2, y = y)



m = @formula y ~ x1 + x2 + GLFixedEffectModels.fe(id1) + GLFixedEffectModels.fe(id2)
@profile GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )
Profile.print()
Profile.clear()

@time GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )

@profile (for i = 1:5; GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] ); end)

# Benchmark GLFixedEffectModels

# Two FE
m = @formula y ~ x1 + x2 + fe(id1) + fe(id2)
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


# BENCHMARK FIXEST

# One thread
function runme()
    R"res <- fixest::feglm(y ~ x1 + x2  | id1, df_r, family = \"poisson\", nthreads = 1, start = c(0.2,0.2))"
end
function runme()
    R"res <- fixest::feglm(y ~ x1 + x2  | id1 + id2, df_r, family = \"poisson\", nthreads = 1, start = c(0.2,0.2))"
end
# Two FE's
# BenchmarkTools.Trial:
#   memory estimate:  1.09 KiB
#   allocs estimate:  32
#   --------------
#   minimum time:     3.201 s (0.00% GC)
#   median time:      3.393 s (0.00% GC)
#   mean time:        3.392 s (0.00% GC)
#   maximum time:     3.725 s (0.00% GC)
#   --------------
#   samples:          10
#   evals/sample:     1
@benchmark runme()

# As many threads as you like
function runme()
    R"res <- fixest::feglm(y ~ x1 + x2  | id1, df_r, family = \"poisson\", start = c(0.2,0.2))"
end
function runme()
    R"res <- fixest::feglm(y ~ x1 + x2  | id1 + id2, df_r, family = \"poisson\", start = c(0.2,0.2))"
end
@benchmark runme()


R"rm(list = ls())"
