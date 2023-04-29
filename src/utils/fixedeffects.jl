

##############################################################################
##
## Subet and Make sure Interaction if Vector{Float64} (instead of missing)
##
##############################################################################

# index and convert interaction Vector{Float64,, Missing} to Vector{Missing}
function _subset(fe::FixedEffect, esample)
    interaction = convert(AbstractVector{Float64}, fe.interaction[esample])
    FixedEffect{typeof(fe.refs), typeof(interaction)}(fe.refs[esample], interaction, fe.n)
end
