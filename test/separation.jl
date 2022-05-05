using DataFrames
include("../src/GLFixedEffectModels.jl")
using .GLFixedEffectModels
using Test


# test1 for collinearity
df = DataFrame(y = rand(6), x1 = [1;0.5;0.8;1;0;0], x2 = [0;0.5;0.2;0;0;0], id = [1;1;1;1;2;2])

# y ~ x1 + x2 + fe(id), will drop x2
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(id)), Poisson(), LogLink())
@test res1.coef[1] == 0
# y ~ x1 + fe(id)

res2 = nlreg(df, @formula(y ~ x2 + fe(id)), Poisson(), LogLink())
@test res2.coef[1] != 0

# ---------------------------------------------------------------------------------------------------------------- #

# test2 for ReLU separation
df = DataFrame(y = [0.0;1;0;0;5], x = [0;0;0;0;0], id1 = [1;1;2;2;2], id2 = [1;1;1;2;2])

res1 = nlreg(df, @formula(y ~ x + fe(id1) + fe(id2)), Poisson(), LogLink() ; separation = [:ReLU])
@test res1.nobs == 4

# test3 for FE separation
df = DataFrame(y = [0.0;0;0;1;2;3], id = [1;1;2;2;3;3])

res1 = nlreg(df, @formula(y ~ fe(id)), Poisson(), LogLink() ; separation = [:fe])
@test res1.nobs == 4