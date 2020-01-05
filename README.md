# GLFixedEffectModels.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/jmboehm/GLFixedEffectModels.svg?branch=master)](https://travis-ci.org/jmboehm/GLFixedEffectModels.jl) [![Coverage Status](https://coveralls.io/repos/jmboehm/GLFixedEffectModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/jmboehm/GLFixedEffectModels.jl?branch=master)

This package estimates generalized linear models with high dimensional categorical variables. It builds on Matthieu Gomez's [FixedEffects.jl](https://github.com/FixedEffects/FixedEffects.jl) and Amrei Stammann's [Alpaca](https://github.com/amrei-stammann/alpaca).

# Installation

```
] add https://github.com/jmboehm/GLFixedEffectModels.jl.git
```

# Usage (experimental)

```julia
using GLFixedEffectModels, GLM, Distributions
using RDatasets

df = dataset("datasets", "iris")
df.binary = zeros(Float64, size(df,1))
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesDummy = categorical(df.Species)
idx = rand(1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = categorical(df.Random)

m = @formula binary ~ SepalWidth + fe(SpeciesDummy)
x = nlreg(df, m, Binomial(), LogitLink(), start = [0.2] )

m = @formula binary ~ SepalWidth + PetalLength + fe(SpeciesDummy)
nlreg(df, m, Binomial(), LogitLink(), Vcov.cluster(:SpeciesDummy,:RandomCategorical) , start = [0.2, 0.2] )
```

# Things that still need to be implemented

- Test saving fixed effects
- Better default starting values
- Bias correction
- Weights
- StatsModels interface
- Better benchmarking
- Integration with [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl)

# References

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Stammann, A. (2018) *Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects*. Mimeo, Heinrich-Heine University Düsseldorf
