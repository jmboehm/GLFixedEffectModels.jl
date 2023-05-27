
module GLFixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################

using LinearAlgebra
using Statistics
using Printf
using FillArrays
using DataFrames
using Distributions
using Reexport
using FixedEffects
using LoopVectorization
using Vcov
using StatsBase
using StatsModels
using StatsAPI

@reexport using GLM
@reexport using FixedEffectModels
# not necessary to reexport StatsModels since it is reexported by FixedEffectModels

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
responsename,
bias_correction

##############################################################################
##
## Load files
##
##############################################################################
include("GLFixedEffectModel.jl")

include("utils/vcov.jl")
include("utils/fixedeffects.jl")
include("utils/basecol.jl")
include("utils/biascorr.jl")


include("fit.jl")

include("presolve.jl")

# precompile script
df = DataFrame(y = rand(10), x = 1:10, id = repeat([1, 2], 5))
nlreg(df, @formula(y ~ x + fe(id)), Binomial(), GLM.LogitLink())

end
