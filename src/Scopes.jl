module Scopes
export var_scope, var_op_scope

using PyCall
@pyimport tensorflow as tf

# VariableScope - TODO fix the API generation in Tf.jl
var_scope(name::AbstractString, reuse::Bool=false, initializer::Any=nothing) = tf.variable_scope(name, reuse ? true : nothing, initializer)

function var_scope(f::Function, name::AbstractString, reuse::Bool=false, initializer::Any=nothing)
  vs = var_scope(name, reuse, initializer)
  try
    vs[:__enter__]()
    return f(vs)
  finally
    exc_type, exc_value, traceback = nothing, nothing, nothing
    vs[:__exit__](exc_type, exc_value, traceback)
  end
end

# TODO var_op_scope


end # module
