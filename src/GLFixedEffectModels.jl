
module GLFixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Base
using LinearAlgebra
using Statistics
using Printf
using FillArrays
using DataFrames
using CategoricalArrays
using Distributions
using Reexport
using GLM
using Combinatorics

@reexport using StatsBase
@reexport using GLM

using FixedEffects

##############################################################################
##
## Exported methods and types
##
##############################################################################

export nlreg,
fe,
GLFixedEffectModel,
has_fe,
Vcov,
VcovData,
responsename

##############################################################################
##
## Load files
##
##############################################################################
include("GLFixedEffectModel.jl")

include("utils/tss.jl")
include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/formula.jl")
include("utils/biascorr.jl")

include("vcov/Vcov.jl")

include("fit.jl")

# precompile script
df = DataFrame(y = rand(10), x = 1:10, id = repeat([1, 2], 5))
nlreg(df, @formula(y ~ x + fe(id)), Binomial(), GLM.LogitLink())

end
