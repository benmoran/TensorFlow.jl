module ExtraSyntax
using ..CoreTypes
using ..API
import ..API: get_variable
export FLAGS

const FLAGS = API.TfFlags.tf_flags.FLAGS

get_variable(name::AbstractString, dt::Vector) = get_variable(name, NegDimsType(dt))

end
