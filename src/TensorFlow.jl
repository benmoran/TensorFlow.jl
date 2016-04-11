module TensorFlow
import Base: (*), (+), (-), (.*), (==), size, log, exp, mean, squeeze, linspace, randn, run, close, isequal
export AbstractTensor, Tensor, Session, Variable, dtype, constant, cast, argmax, Placeholder, FeedDict, Optimizer, minimize, Operation, batch_matmul, var_scope, var_op_scopeo
export all_variables, initialize_all_variables, tfeval, DimsType, FLAGS

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

include("Scopes.jl")
using .Scopes

include("ExtraSyntax.jl")
using .ExtraSyntax

end # module
