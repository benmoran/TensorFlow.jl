# N.B. This file is maintained in TensorFlowBuilder source
using PyCall
import Base: convert
import PyCall: PyObject

export SessionRunnable, AbstractTensor, DimsType, NoneDimsType, NegDimsType, PyVectorType

macro pywrapper(typename, supertype)
  quote
    immutable $typename <: $supertype
      x::PyCall.PyObject
    end
    PyCall.PyObject(o::$typename) = o.x
    isequal(x::$typename, y::$typename) = x.x[__eq__](y.x)
    export $typename
  end
end

abstract SessionRunnable
abstract AbstractTensor <: SessionRunnable


@pywrapper Dtype Any

@pywrapper Tensor AbstractTensor
@pywrapper Variable AbstractTensor
@pywrapper SparseTensor AbstractTensor
@pywrapper Placeholder AbstractTensor

@pywrapper Operation SessionRunnable

@pywrapper Session Any

@pywrapper Optimizer Any

convert(::Type{Tensor}, a::Array) = Tensor(a)
convert(::Type{AbstractTensor}, a::Array) = Tensor(a)

## DimsType ##
# TensorFlow shape arguments sometimes want None to mean "wild" (e.g. tf.placeholder)
# and sometimes use negatives to mean wildcard (e.g.reshape)

abstract DimsType

"A literal used for TensorFlow shape arguments.
  Negative indices are passed as Python `None` values."
type NoneDimsType <: DimsType
  arr::Vector{Int}
end
NoneDimsType(a::Tuple{Vararg{Int}}) = NoneDimsType(collect(a))
PyCall.PyObject(o::NoneDimsType) = PyCall.PyObject([a < 0 ? nothing : a for a in o.arr])

"A literal used for TensorFlow shape arguments.
  Negative indices are passed through to Python."
type NegDimsType <: DimsType
  arr::Vector{Int}
end
NegDimsType(a::Tuple{Vararg{Int}}) = NegDimsType(collect(a))
PyCall.PyObject(o::NegDimsType) = PyCall.PyObject(o.arr)


"A wrapper type for things that TensorFlow insists must be Python
  lists, like conv2d's strides argument."
type PyVectorType <: DimsType
  pyvec::PyVector
end
PyVectorType(a::Vector) = PyVectorType(PyVector(a))
PyVectorType(a::Tuple) = PyVectorType(collect(a))
PyCall.PyObject(o::PyVectorType) = PyCall.PyObject(o.pyvec)
