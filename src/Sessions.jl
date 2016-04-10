module Sessions
using PyCall
using ..CoreTypes
using ..API
import ..CoreTypes: Session, tf, Variable, Placeholder
import Base: run, close

export initialize_all_variables, FeedDict, all_variables

typealias FeedDict Dict{Placeholder, Array}
PyObject(fd::FeedDict) = PyObject(Dict([k.x => v for (k,v) in fd]))

Placeholder(dtype::Dtype, shape::NoneDimsType, name=nothing) = API.Tf.placeholder(dtype, shape, name)
Placeholder(dtype::Dtype, shape::Vector{Int}, name=nothing) = Placeholder(dtype, NoneDimsType(shape), name)

initialize_all_variables() = Operation(tf.initialize_all_variables())

#all_variables() = Variable[Variable(v) for v in tf.all_variables()]

Variable(value::AbstractTensor,
         trainable::Bool=true,
         collections::Union{Void, Array{AbstractString}}=nothing,
         validate_shape::Bool=true,
         name::AbstractString="") = Variable(tf.Variable(convert(PyCall.PyObject,value),
                                                         trainable,
                                                         collections,
                                                         validate_shape,
                                                         isempty(name) ? nothing : name))


### Session  ###

Session() = Session(tf.Session())


run{T<:SessionRunnable}(sess::Session, g::T, fd::FeedDict=FeedDict()) = sess.x[:run](g.x, fd)
close(sess::Session) = sess.x[:close]()
tfeval(t::AbstractTensor, fd::FeedDict) = t.x[:eval](fd)

# TODO Get eval'ing in a "do" block to work, analagous to open() in Base
# function InteractiveSession(f::Function, fd::FeedDict=FeedDict())
#   sess = InteractiveSession()
#   try:
#     run(sess, f(sess), fd)
#   finally
#     #close(sess)
#   end
# end

end
