using DataFrames
include("../src/GLFixedEffectModels.jl")
using .GLFixedEffectModels
using Test
using Downloads, CSV


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

# ---------------------------------------------------------------------------------------------------------------- #

#=
clear
input int(y x1 x2 x3)
 0   0   1  0
 0   0   0  0
 0   0   0  0
 0   0   0  0
 0   1   9  0
 2  21  21 21
 3   0   0  0
 5   0   0  0
 7   0   0  0
10 -18 -18  0
end
=#
url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/example1.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + x4 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/example2.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + x4 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/fe1.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(i) + fe(j)), Poisson(), LogLink() ; separation = [:ReLU])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/fe2.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ x1 + fe(i) + fe(j)), Poisson(), LogLink() ; separation = [:ReLU])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/fe3.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(i) + fe(j)), Poisson(), LogLink() ; separation = [:ReLU])