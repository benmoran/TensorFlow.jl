"Make Tensors appear like Julia Arrays"
module Idiomatic
using ..CoreTypes
using ..API.Tf
using ..API.TfNn
import ..API.Tf: constant

import Base: (*), (+), (-), (.*), (.+), (/), (==), size, log, exp, mean, squeeze, linspace, randn, length, reshape, isequal, sum, mean, zeros, getindex, split

export constant, dtype, cast
export relu, batch_matmul

zeroindexed(d::DimsType) = map(x->x-1, d)
dtype(t::AbstractTensor) = Dtype(t.x[:dtype])

# Functions that accept arrays as well
constant{N<:Number}(n::N, dt::Dtype) = constant(Tensor(n), dt)
constant(a::Array, dt::Dtype) = constant(Tensor(a), dt)

constant(a::Array) = constant(Tensor(a))
constant{N<:Number}(n::N) = constant(Tensor(n))


(.*){N<:Number}(n::N, t::AbstractTensor) = Tf.mul(n,t)
(.*){N<:Number}(t::AbstractTensor, n::N) = Tf.mul(t,n)
(.*)(t::AbstractTensor, s::AbstractTensor) = Tf.mul(t,s)
(*){N<:Number}(n::N, t::AbstractTensor) = Tf.mul(constant(n, dtype(t)) ,t)
(*){N<:Number}(t::AbstractTensor, n::N) = Tf.mul(t, constant(n, dtype(t)))
(/){N<:Number}(n::N, t::AbstractTensor) = Tf.div_(constant(n, dtype(t)) ,t)
(/){N<:Number}(t::AbstractTensor, n::N) = Tf.div_(t, constant(n, dtype(t)))
(*)(t::AbstractTensor, s::AbstractTensor) = Tf.matmul(t,s)
(+){N<:Number}(n::N, t::AbstractTensor) = Tf.add(constant(n, dtype(t)), t)
(+){N<:Number}(t::AbstractTensor, n::N) = Tf.add(t, constant(n, dtype(t)))
(+)(t::AbstractTensor, s::AbstractTensor) = Tf.add(t,s)
(.+)(t::AbstractTensor, s::AbstractTensor) = Tf.add(t,s)
(-)(t::AbstractTensor, s::AbstractTensor) = Tf.sub_(t,s)
(-){N<:Number}(t::AbstractTensor, n::N) = Tf.sub_(t, constant(n, dtype(t)))
(-){N<:Number}(n::N, t::AbstractTensor) = Tf.sub_(constant(n, dtype(t)), t)
(==)(t::AbstractTensor, s::AbstractTensor) = Tf.equal(t,s)

# We need to use Placeholders as dictionary keys in a FeedDict. So let's
# have isequal() compare by Python object identity.
isequal(t::AbstractTensor, s::AbstractTensor) = isequal(t.x.o, s.x.o)

batch_matmul(x::AbstractTensor, y::AbstractTensor) = Tf._batch_mat_mul(x, y)

log(t::AbstractTensor) = Tf.log_(t)
exp(t::AbstractTensor) = Tf.exp_(t)

# Julia has findmax(A, dims) but that returns (max, index) rather than just index
argmax{T<:Integer}(t::AbstractTensor, n::T) = Tf.arg_max(t, n-1) # name


# TODO sum over axis
sum(t::AbstractTensor) = Tf.reduce_sum(t)

mean(t::AbstractTensor, region) = Tf.reduce_mean(t, zeroindexed(region), keep_dims=true)
mean(t::AbstractTensor) = Tf.reduce_mean(t)
squeeze(t::AbstractTensor, dims) = Tf.squeeze(t, zeroindexed(dims))
cast(t::AbstractTensor, dt::Dtype) = Tf.cast(t, dt)
cast{R<:Range}(r::R, dt::Dtype) = constant(collect(r), dt)


#size(t::AbstractTensor) = Tf.shape(t) # N.B. not tf.size_
size(t::AbstractTensor) = tuple(t.x[:get_shape]()[:as_list]())
size(t::AbstractTensor, dim::Int) = size(t)[dim]

length(t::AbstractTensor) = Tensor(Tf.size_(t))


#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
randn(::Type{Tensor}, shape::NegDimsType, dtype=DT_FLOAT32, name=nothing) = Tf.random_normal(shape, constant(0.0), constant(1.0), dtype, nothing, name)
randn(T::Type{Tensor}, shape::Vector{Int}, dtype=DT_FLOAT32, name=nothing)  = randn(T, NegDimsType(shape), dtype, name)

reshape(t::AbstractTensor, shape::NegDimsType) = Tf.reshape_(t, shape)
reshape(t::AbstractTensor, shape::Vector{Int}) = reshape(t, NegDimsType(shape))

zeros{T<:AbstractTensor}(::Type{T}, dims::NegDimsType, dtype=DT_FLOAT32) = Tf.zeros_(dims, dtype)
zeros{T<:AbstractTensor}(::Type{T}, dims::Vector{Int}, dtype=DT_FLOAT32) = zeros(T, NegDimsType(dims), dtype)

getindex(t::AbstractTensor, ix::Int) = Tensor(t.x[ix-1])

split(split_dim::Int, num_split::Int, value::Tensor, name::AbstractString) = Tf.split_(constant(split_dim-1), num_split, value, name)
split(split_dim::Int, num_split::Int, value::Tensor) = split(split_dim, num_split, value, "split")
end
