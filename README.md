# GLFixedEffectModels.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/jmboehm/GLFixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/jmboehm/GLFixedEffectModels.jl) [![Coverage Status](https://coveralls.io/repos/jmboehm/GLFixedEffectModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/jmboehm/GLFixedEffectModels.jl?branch=master)

This package estimates generalized linear models with high dimensional categorical variables. It builds on Matthieu Gomez's [FixedEffects.jl](https://github.com/FixedEffects/FixedEffects.jl) and Amrei Stammann's [Alpaca](https://github.com/amrei-stammann/alpaca).

## Installation

```
] add https://github.com/jmboehm/GLFixedEffectModels.jl.git
```

## Example use

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

## Documentation

The main function is `nlreg()`, which returns a `GLFixedEffectModel <: RegressionModel`.
```julia
nlreg(df, formula::FormulaTerm,
    distribution::Distribution,
    link::GLM.Link,
    vcov::CovarianceEstimator; ...)
```
The required arguments are:
* `df`: a Table
* `formula`: A formula created using `@formula`.
* `distribution`: A `Distribution`. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid distributions.
* `link`: A `Link` function. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid link functions.
* `vcov`: A `CovarianceEstimator` to compute the variance-covariance matrix.

The optional arguments are:
* `save::Union{Bool, Symbol} = false`: Should residuals and eventual estimated fixed effects saved in a dataframe? Use `save = :residuals` to only save residuals. Use `save = :fe` to only save fixed effects.
* `method::Symbol = :lsmr`: Method to deman regressors. `:lsmr` is akin to conjugate gradient descent.  To use LSMR on multiple cores, use `:lsmr_parallel`. To use LSMR with multiple threads,  use `lsmr_threads`. To use LSMR on GPU, use `lsmr_gpu`(requires `CuArrays`. Use the option `double_precision = false` to use `Float32` on the GPU.). This option is the same as for the [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) package.
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 1000`: Maximum number of iterations in the Newton-Raphson routine.
* `maxiter_center::Integer = 10000`: Maximum number of iterations for centering procedure.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `dev_tol::Real` : Tolerance level for the first stopping condition of the maximization routine.
* `rho_tol::Real` : Tolerance level for the stephalving in the maximization routine.
* `step_tol::Real` : Tolerance level that accounts for rounding errors inside the stephalving routine
* `center_tol::Real` : Tolerance level for the stopping condition of the centering algorithm. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.

## Things that still need to be implemented

- Better default starting values
- Bias correction
- Weights
- Better StatsBase interface & prediction
- Better benchmarking
- Integration with [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl)

## References

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Stammann, A. (2018) *Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects*. Mimeo, Heinrich-Heine University Düsseldorf
