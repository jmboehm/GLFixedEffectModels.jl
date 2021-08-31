# GLFixedEffectModels.jl

<!--![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg) -->
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
![example branch parameter](https://github.com/jmboehm/GLFixedEffectModels.jl/actions/workflows/ci.yml/badge.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/jmboehm/GLFixedEffectModels.jl/badge.svg?branch=master)](https://coveralls.io/github/jmboehm/GLFixedEffectModels.jl?branch=master)

This package estimates generalized linear models with high dimensional categorical variables. It builds on Matthieu Gomez's [FixedEffects.jl](https://github.com/FixedEffects/FixedEffects.jl) and Amrei Stammann's [Alpaca](https://github.com/amrei-stammann/alpaca).

## Installation

```
] add GLFixedEffectModels
```

## Example use

```julia
using GLFixedEffectModels, GLM, Distributions
using RDatasets

df = dataset("datasets", "iris")
df.binary = zeros(Float64, size(df,1))
df[df.SepalLength .> 5.0,:binary] .= 1.0
df.SpeciesStr = string.(df.Species)
idx = rand(1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])

m = @formula binary ~ SepalWidth + fe(Species)
x = nlreg(df, m, Binomial(), LogitLink(), start = [0.2] )

m = @formula binary ~ SepalWidth + PetalLength + fe(Species)
nlreg(df, m, Binomial(), LogitLink(), Vcov.cluster(:SpeciesStr,:Random) , start = [0.2, 0.2] )
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
* `method::Symbol`: A symbol for the method. Default is `:cpu`. Alternatively, `:gpu` requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`. This option is the same as for the [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) package.
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 1000`: Maximum number of iterations in the Newton-Raphson routine.
* `maxiter_center::Integer = 10000`: Maximum number of iterations for centering procedure.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `dev_tol::Real` : Tolerance level for the first stopping condition of the maximization routine.
* `rho_tol::Real` : Tolerance level for the stephalving in the maximization routine.
* `step_tol::Real` : Tolerance level that accounts for rounding errors inside the stephalving routine
* `center_tol::Real` : Tolerance level for the stopping condition of the centering algorithm. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `separation::Symbol = :ignore` : Method to detect/deal with [separation](https://github.com/sergiocorreia/ppmlhdfe/blob/master/guides/separation_primer.md). Currently supported values are `:none`, `:ignore` and `:mu`. `:none` checks for observations that are outside `[separation_mu_lbound,separation_mu_ubound]`, and gives a warning, but does not do anything. `:ignore` does not check (and may therefore be slightly faster than the other options). `:mu` truncates mu at `separation_mu_lbound` or `separation_mu_ubound`. 
* `separation_mu_lbound::Real = -Inf` : Lower bound for the separation detection/correction heuristic (on mu). What a reasonable value would be depends on the model that you're trying to fit.
* `separation_mu_ubound::Real = Inf` : Upper bound for the separation detection/correction heuristic.

## Things that still need to be implemented

- Better default starting values
- Bias correction
- Weights
- Better StatsBase interface & prediction
- Better benchmarking
- Integration with [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl)

## Related Julia packages

- [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) estimates linear models with high dimensional categorical variables (and with or without endogeneous regressors).
- [FixedEffects.jl](https://github.com/FixedEffects/FixedEffects.jl) is a package for fast pseudo-demeaning operations using LSMR. Both this package and [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) build on this.
- [Alpaca.jl](https://github.com/jmboehm/Alpaca.jl) is a wrapper to the [Alpaca R package](https://github.com/amrei-stammann/alpaca), which solves the same tasks as this package.
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) estimates generalized linear models, but without explicit support for categorical regressors.
- [Econometrics.jl](https://github.com/Nosferican/Econometrics.jl) provides routines to estimate multinomial logit and other models.
- [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl) will, in the future, support pretty printing of results from this package.

## References

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Stammann, A. (2018) *Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects*. Mimeo, Heinrich-Heine University Düsseldorf
