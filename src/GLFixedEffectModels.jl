
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
using Distributions
using Reexport
using GLM
using Combinatorics

@reexport using StatsBase
@reexport using StatsModels

using FixedEffects

##############################################################################
##
## Exported methods and types
##
##############################################################################

export reg, nlreg,
partial_out,
fe,
GLFixedEffectModel,
has_fe,
Vcov,
VcovData,

#deprecated
@model,
fes

##############################################################################
##
## Load files
##
##############################################################################
include("GLFixedEffectModel.jl")
include("utils/tss.jl")
include("utils/fixedeffects.jl")
include("utils/basecol.jl")

include("utils/ranktest.jl")
include("utils/formula.jl")

include("vcov/Vcov.jl")

include("fit.jl")

# precompile script
df = DataFrame(y = [1, 1], x =[1, 2], id = [1, 1])
nlreg(df, @formula(y ~ x + fe(id)), Binomial(), GLM.LogitLink())
#nlreg(df, @formula(y ~ x), Vcov.cluster(:id))

end
