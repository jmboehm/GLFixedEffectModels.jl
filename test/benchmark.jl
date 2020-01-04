
include("../src/GLFixedEffectModels.jl")
using DataFrames, GLM, Random

rng = MersenneTwister(1234)
N = 1_000_000
K = 100
id1 = rand(rng, 1:(N/K), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1 = categorical(id1), id1_noncat = id1, id2 = categorical(id2), x1 = x1, x2 = x2, y = y)

@time reg(df, @formula(y ~ x1 + x2))
@time reg(df, @formula(y ~ x1 + x2 + fe(id1)))


m = GLFixedEffectModels.@formula y ~ x1 + x2 + GLFixedEffectModels.fe(id1)
@time x = GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )

m = GLFixedEffectModels.@formula y ~ x1 + x2 + GLFixedEffectModels.fe(id1) +  GLFixedEffectModels.fe(id2)
@time x = GLFixedEffectModels.nlreg(df, m, Poisson(), GLM.LogLink() , start = [0.2;0.2] )

using Alpaca
result = Alpaca.feglm(df, Alpaca.@formula(y ~ x1 + x2),
    Poisson(),
    fe = :id1,
    start = [0.2;0.2], trace = 2
    )


rng = MersenneTwister(1234)
N = 1_000_000
K = 100
id1 = rand(rng, 1:(round(Int64,N/K)), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1_noncat = id1, id2_noncat = id2, x1 = x1, x2 = x2, y = y)

using RCall
df_r = deepcopy(df)
@rput df_r
R"df_r$id1 <- as.factor(df_r$id1_noncat)"
R"df_r$id2 <- as.factor(df_r$id2_noncat)"
R"library(alpaca)"
R"ctrl <- feglmControl(dev.tol = 1e-08,
       center.tol = 1e-08, rho.tol = 1e-08,
       conv.tol = 1e-08,
       iter.max = 100,
       trace = 1,
       drop.pc = 1)"
@time R"result <- feglm(formula =  y ~ x1 + x2 | id1 , data = df_r,
            family = poisson() , beta.start = c(0.2,0.2),
            control = ctrl)"
@time R"result <- feglm(formula =  y ~ x1 + x2 | id1 + id2 , data = df_r,
            family = poisson() , beta.start = c(0.2,0.2),
            control = ctrl)"
