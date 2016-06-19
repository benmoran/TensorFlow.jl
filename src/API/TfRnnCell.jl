"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.6.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module TfRnnCell
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.ops.rnn_cell as tf_rnn_cell
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Create a cell with added input and/or output dropout.

    Dropout is never used on the state.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
DropoutWrapper(cell_::Any, input_keep_prob::Any=1.0, output_keep_prob::Any=1.0, seed::Union{Int64,Void}=nothing) = tf_rnn_cell.DropoutWrapper(;Dict(:cell=>cell_, :input_keep_prob=>input_keep_prob, :output_keep_prob=>output_keep_prob, :seed=>seed)...)
export DropoutWrapper
          

"""
Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding: Variable, the embedding to use; if None, a new embedding
        will be created; if set, then embedding_classes is not required.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
EmbeddingWrapper(cell_::Any, embedding_classes::Any=0, embedding::Any=nothing, initializer::Any=nothing) = tf_rnn_cell.EmbeddingWrapper(;Dict(:cell=>cell_, :embedding_classes=>embedding_classes, :embedding=>embedding, :initializer=>initializer)...)
export EmbeddingWrapper
          

"""
Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      input_size: integer, the size of the inputs before projection.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if input_size is not positive.
    """
InputProjectionWrapper(cell_::Any, input_size::Union{Int64,Void}) = tf_rnn_cell.InputProjectionWrapper(;Dict(:cell=>cell_, :input_size=>input_size)...)
export InputProjectionWrapper
          

"""
Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: int, The dimensionality of the inputs into the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
    """
LSTMCell(num_units::Union{Int64,Void}, input_size::Union{Int64,Void}, use_peepholes::Bool=false, cell_clip::Any=nothing, initializer::Any=nothing, num_proj::Union{Int64,Void}=nothing, num_unit_shards::Int64=1, num_proj_shards::Int64=1) = tf_rnn_cell.LSTMCell(;Dict(:num_units=>num_units, :input_size=>input_size, :use_peepholes=>use_peepholes, :cell_clip=>cell_clip, :initializer=>initializer, :num_proj=>num_proj, :num_unit_shards=>num_unit_shards, :num_proj_shards=>num_proj_shards)...)
export LSTMCell
          

"""
Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.

    Raises:
      ValueError: if cells is empty (not allowed) or if their sizes don't match.
    """
MultiRNNCell(cells::Any) = tf_rnn_cell.MultiRNNCell(;Dict(:cells=>cells)...)
export MultiRNNCell
          

"""
Create a cell with output projection.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      output_size: integer, the size of the output after projection.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if output_size is not positive.
    """
OutputProjectionWrapper(cell_::Any, output_size::Union{Int64,Void}) = tf_rnn_cell.OutputProjectionWrapper(;Dict(:cell=>cell_, :output_size=>output_size)...)
export OutputProjectionWrapper
          

"""
Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
linear(args::Union{AbstractTensor,Void}, output_size::Union{Int64,Void}, bias::Any, bias_start::Any=0.0, scope::Any=nothing) = Tensor(tf_rnn_cell.linear(;Dict(:args=>args, :output_size=>output_size, :bias=>bias, :bias_start=>bias_start, :scope=>scope)...))
export linear
          

"""
Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32`
      otherwise the return type is `quint8`.
  """
sigmoid(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_rnn_cell.sigmoid(;Dict(:x=>x, :name=>name)...))
export sigmoid
          

"""
Computes hyperbolic tangent of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
tanh_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_rnn_cell.tanh(;Dict(:x=>x, :name=>name)...))
export tanh_
          end
