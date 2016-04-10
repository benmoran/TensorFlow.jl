# N.B. This file is maintained in TensorFlowBuilder source
module CoreTypes
export DimsType
import PyCall: PyObject, @pyimport
@pyimport tensorflow as tf

include("types.jl")
include("dtypes.jl")

end
