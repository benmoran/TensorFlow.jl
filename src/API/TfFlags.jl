"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.8.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module TfFlags
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.platform.flags as tf_flags
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Defines a flag of type 'boolean'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a boolean.
    docstring: A helpful message explaining the use of the flag.
  """
DEFINE_boolean(flag_name::Any, default_value::Any, docstring::Any) = tf_flags.DEFINE_boolean(;Dict(:flag_name=>flag_name, :default_value=>default_value, :docstring=>docstring)...)
export DEFINE_boolean
          

"""
Defines a flag of type 'boolean'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a boolean.
    docstring: A helpful message explaining the use of the flag.
  """
DEFINE_boolean(flag_name::Any, default_value::Any, docstring::Any) = tf_flags.DEFINE_boolean(;Dict(:flag_name=>flag_name, :default_value=>default_value, :docstring=>docstring)...)
export DEFINE_boolean
          

"""
Defines a flag of type 'float'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a float.
    docstring: A helpful message explaining the use of the flag.
  """
DEFINE_float(flag_name::Any, default_value::Any, docstring::Any) = tf_flags.DEFINE_float(;Dict(:flag_name=>flag_name, :default_value=>default_value, :docstring=>docstring)...)
export DEFINE_float
          

"""
Defines a flag of type 'int'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as an int.
    docstring: A helpful message explaining the use of the flag.
  """
DEFINE_integer(flag_name::Any, default_value::Union{Int64,Void}, docstring::Any) = tf_flags.DEFINE_integer(;Dict(:flag_name=>flag_name, :default_value=>default_value, :docstring=>docstring)...)
export DEFINE_integer
          

"""
Defines a flag of type 'string'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a string.
    docstring: A helpful message explaining the use of the flag.
  """
DEFINE_string(flag_name::Any, default_value::Any, docstring::Any) = tf_flags.DEFINE_string(;Dict(:flag_name=>flag_name, :default_value=>default_value, :docstring=>docstring)...)
export DEFINE_string
          end
