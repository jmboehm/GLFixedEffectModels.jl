using DataFrames
include("../src/GLFixedEffectModels.jl")
using .GLFixedEffectModels
using Test
using Downloads, CSV, Random

Random.seed!(1234)

# test1 for collinearity
df = DataFrame(y = rand(6), x1 = [1;0.5;0.8;1;0;0], x2 = [0;0.5;0.2;0;0;0], id = [1;1;1;1;2;2])

# y ~ x1 + x2 + fe(id), will drop x2
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(id)), Poisson(), LogLink())
@test 0 ∈ res1.coef
@show res1
# y ~ x1 + fe(id)

res2 = nlreg(df, @formula(y ~ x2 + fe(id)), Poisson(), LogLink())
@test 0 ∉ res2.coef
@show res2

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
# benchmark from sergio correia's ppmlhdfe repo
url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/example1.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
# add one fixed effect that is basically a intercept, because nlreg won't run without fe
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + x4 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/guides/csv/example2.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
# add one fixed effect that is basically a intercept, because nlreg won't run without fe
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
res1 = nlreg(df, @formula(y ~ x1 + fe(i) + fe(j)), Poisson(), LogLink() ; separation = [:ReLU])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/01.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/02.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ fe(id1) + fe(id2)), Poisson(), LogLink() ; drop_singletons = false, separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/03.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ fe(id1) + fe(id2) + fe(id3)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/04.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ fe(id1) + fe(id2)), Poisson(), LogLink() ; separation = [:ReLU])
# don't test on the last ob because it was a singleton instead of a separation
@test all(df.separated[1:end-1] .== .~res1.esample[1:end-1])

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/05.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
# add one fixed effect that is basically a intercept, because nlreg won't run without fe
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + x4 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/06.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
# add one fixed effect that is basically a intercept, because nlreg won't run without fe
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + x4 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

#= something wrong with this one
i think the reason is that ppml calls :simplex before :ReLU.
ppmlhdfe's output is:
(simplex method dropped 4 separated observations)
(dropped 1 singleton observations)
something was dropped before calling ReLU.

when setting ppmlhdfe sep as sep(ir), the output is:
(ReLU method dropped 1 separated observation in 2 iterations)
and the output gives ill results too.

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/07.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)), Poisson(), LogLink() ; drop_singletons = false, separation = [:ReLU])
@test all(df.separated .== .~res1.esample)
=# 

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/08.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(id1) + fe(id2)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/09.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
# add one fixed effect that is basically a intercept, because nlreg won't run without fe
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/10.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/11.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/12.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ fe(id1) + fe(id2)), Poisson(), LogLink() ; drop_singletons = false , separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/13.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
res1 = nlreg(df, @formula(y ~ fe(id1) + fe(id2)), Poisson(), LogLink() ; drop_singletons = false , separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/14.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/15.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/16.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/17.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)

url = "https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/master/test/separation_datasets/18.csv"
df = DataFrame(CSV.File(Downloads.download(url)))
df.id = ones(size(df,1))
res1 = nlreg(df, @formula(y ~ x1 + x2 + x3 + fe(id)), Poisson(), LogLink() ; separation = [:ReLU])
@test all(df.separated .== .~res1.esample)