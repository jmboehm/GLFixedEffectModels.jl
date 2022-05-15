# GLFixedEffectModels.jl

<!--![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
![example branch parameter](https://github.com/jmboehm/GLFixedEffectModels.jl/actions/workflows/ci.yml/badge.svg?branch=master) [![codecov.io](http://codecov.io/github/jmboehm/GLFixedEffectModels.jl/coverage.svg?branch=master)](http://codecov.io/github/jmboehm/GLFixedEffectModels.jl?branch=master) [![DOI](https://zenodo.org/badge/164128032.svg)](https://zenodo.org/badge/latestdoi/164128032)

This package estimates generalized linear models with high dimensional categorical variables. It builds on Matthieu Gomez's [FixedEffects.jl](https://github.com/FixedEffects/FixedEffects.jl), Amrei Stammann's [Alpaca](https://github.com/amrei-stammann/alpaca), and Sergio Correia's [ppmlhdfe](https://github.com/sergiocorreia/ppmlhdfe).

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
* `link`: A `GLM.Link` function. See the documentation of [GLM.jl](https://juliastats.org/GLM.jl/stable/manual/#Fitting-GLM-models-1) for valid link functions.
* `vcov`: A `CovarianceEstimator` to compute the variance-covariance matrix.

The optional arguments are:
* `save::Union{Bool, Symbol} = false`: Should residuals and eventual estimated fixed effects saved in a dataframe? Use `save = :residuals` to only save residuals. Use `save = :fe` to only save fixed effects.
* `method::Symbol`: A symbol for the method. Default is `:cpu`. Alternatively, `:gpu` requires `CuArrays`. In this case, use the option `double_precision = false` to use `Float32`. This option is the same as for the [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) package.
* `double_precision::Bool = true`: Uses 64-bit floats if `true`, otherwise 32-bit.
* `drop_singletons = true` : drop observations that are perfectly classified.
* `contrasts::Dict = Dict()` An optional Dict of contrast codings for each categorical variable in the `formula`.  Any unspecified variables will have `DummyCoding`.
* `maxiter::Integer = 1000`: Maximum number of iterations in the Newton-Raphson routine.
* `maxiter_center::Integer = 10000`: Maximum number of iterations for centering procedure.
* `double_precision::Bool`: Should the demeaning operation use Float64 rather than Float32? Default to true.
* `dev_tol::Real` : Tolerance level for the first stopping condition of the maximization routine.
* `rho_tol::Real` : Tolerance level for the stephalving in the maximization routine.
* `step_tol::Real` : Tolerance level that accounts for rounding errors inside the stephalving routine
* `center_tol::Real` : Tolerance level for the stopping condition of the centering algorithm. Default to 1e-8 if `double_precision = true`, 1e-6 otherwise.
* `separation::Vector{Symbol} = Symbol[]` : Method to detect/deal with [separation](https://github.com/sergiocorreia/ppmlhdfe/blob/master/guides/separation_primer.md). Supported elements are `:mu`, `:fe`, `:ReLU`, and in the future, `:simplex`. `:mu` truncates mu at `separation_mu_lbound` or `separation_mu_ubound`. `:fe` finds categories of the fixed effects that only exist when y is at the separation point. `ReLU` detects separation using ReLU, with the maxiter being `separation_ReLU_maxiter` and tolerance being `separation_ReLU_tol`.
* `separation_mu_lbound::Real = -Inf` : Lower bound for the separation detection/correction heuristic (on mu). What a reasonable value would be depends on the model that you're trying to fit.
* `separation_mu_ubound::Real = Inf` : Upper bound for the separation detection/correction heuristic.
* `separation_ReLU_tol::Real = 1e-4` : Tolerance level for the ReLU algorithm.
* `separation_ReLU_maxiter::Integer = 1000` : Maximal number of iterations for the ReLU algorithm.
* `verbose::Bool = false` : If `true`, prints output on each iteration.

The function returns a `GLFixedEffectModel` object which supports the `StatsBase.RegressionModel` abstraction. It can be displayed in table form by using [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl).

## Bias correction methods

The package experimentally supports bias correction methods for the following models:
- Binomial regression, Logit link, Two-way, Classic (Fernández-Val and Weidner (2016, 2018))
- Binomial regression, Probit link, Two-way, Classic (Fernández-Val and Weidner (2016, 2018))
- Binomial regression, Logit link, Two-way, Network (Hinz, Stammann and Wanner (2020) & Fernández-Val and Weidner (2016))
- Binomial regression, Probit link, Two-way, Network (Hinz, Stammann and Wanner (2020) & Fernández-Val and Weidner (2016))
- Binomial regression, Logit link, Three-way, Network (Hinz, Stammann and Wanner (2020))
- Binomial regression, Probit link, Three-way, Network (Hinz, Stammann and Wanner (2020))
- Poisson regression, Log link, Three-way, Network (Weidner and Zylkin (2021))
- Poisson regression, Log link, Two-way, Network (Weidner and Zylkin (2021))

## Things that still need to be implemented

- Better default starting values
- Weights
- Better StatsBase interface & prediction
- Better benchmarking

## Related Julia packages

- [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) estimates linear models with high dimensional categorical variables (and with or without endogeneous regressors).
- [FixedEffects.jl](https://github.com/FixedEffects/FixedEffects.jl) is a package for fast pseudo-demeaning operations using LSMR. Both this package and [FixedEffectModels.jl](https://github.com/FixedEffects/FixedEffectModels.jl) build on this.
- [Alpaca.jl](https://github.com/jmboehm/Alpaca.jl) is a wrapper to the [Alpaca R package](https://github.com/amrei-stammann/alpaca), which solves the same tasks as this package.
- [GLM.jl](https://github.com/JuliaStats/GLM.jl) estimates generalized linear models, but without explicit support for categorical regressors.
- [Econometrics.jl](https://github.com/Nosferican/Econometrics.jl) provides routines to estimate multinomial logit and other models.
- [RegressionTables.jl](https://github.com/jmboehm/RegressionTables.jl) supports pretty printing of results from this package.

## References

Correia, S. and Guimarães, P, and Zylkin, T., 2019. Verifying the existence of maximum likelihood estimates for generalized linear models. Working paper, https://arxiv.org/abs/1903.01633

Fernández-Val, I. and Weidner, M., 2016. Individual and time effects in nonlinear panel models with large N, T. Journal of Econometrics, 192(1), pp.291-312.

Fernández-Val, I. and Weidner, M., 2018. Fixed effects estimation of large-T panel data models. Annual Review of Economics, 10, pp.109-138.

Fong, DC. and Saunders, M. (2011) *LSMR: An Iterative Algorithm for Sparse Least-Squares Problems*.  SIAM Journal on Scientific Computing

Hinz, J., Stammann, A. and Wanner, J., 2021. State dependence and unobserved heterogeneity in the extensive margin of trade. 

Stammann, A. (2018) *Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects*. Mimeo, Heinrich-Heine University Düsseldorf

Weidner, M. and Zylkin, T., 2021. Bias and consistency in three-way gravity models. Journal of International Economics, 132, p.103513.
