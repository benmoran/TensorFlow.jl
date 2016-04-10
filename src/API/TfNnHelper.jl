"Extra methods for TfNN functions, providing data type conversion"
module TfNnHelper
using TensorFlow.CoreTypes
import TensorFlow.API.TfNn: conv2d

# TODO When a function wants a PyVector argument, we should emit this extra method automatically

conv2d(input::AbstractTensor, filter_::AbstractTensor, strides_::Vector{Int}, padding::AbstractString, use_cudnn_on_gpu::Union{Void,Bool}=nothing, name::Union{AbstractString,Void}=nothing) = conv2d(input, filter_, PyVectorType(strides_), padding, use_cudnn_on_gpu,name)


end
