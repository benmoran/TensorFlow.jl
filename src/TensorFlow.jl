module TensorFlow
import Base: (*), (+), (-), (.*), (==), size, log, exp, mean, squeeze, linspace, randn, run, close, isequal
export AbstractTensor, Tensor, Session, Variable, dtype, constant, cast, argmax, Placeholder, FeedDict, Optimizer, minimize, Operation
export all_variables, initialize_all_variables, tfeval, DimsType

include("CoreTypes.jl")
using .CoreTypes

include("InputData.jl")
import .InputData

include("API.jl")
using .API

include("Idiomatic.jl")
using .Idiomatic

include("Sessions.jl")
using .Sessions

include("Train.jl")
using .Train

end # module
