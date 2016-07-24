"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.9.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module TfRnnCell
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.ops.rnn_cell as tf_rnn_cell
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
BasicLSTMCell(num_units::Union{Int64,Void}, forget_bias::Any=1.0, input_size::Union{Int64,Void}=nothing, state_is_tuple::Any=false, activation::Function=tanh_) = tf_rnn_cell.BasicLSTMCell(;Dict(:num_units=>num_units, :forget_bias=>forget_bias, :input_size=>input_size, :state_is_tuple=>state_is_tuple, :activation=>activation)...)
export BasicLSTMCell
          

"""
"""
BasicRNNCell(num_units::Union{Int64,Void}, input_size::Union{Int64,Void}=nothing, activation::Function=tanh_) = tf_rnn_cell.BasicRNNCell(;Dict(:num_units=>num_units, :input_size=>input_size, :activation=>activation)...)
export BasicRNNCell
          

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
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
EmbeddingWrapper(cell_::Any, embedding_classes::Any, embedding_size::Union{Int64,Void}, initializer::Any=nothing) = tf_rnn_cell.EmbeddingWrapper(;Dict(:cell=>cell_, :embedding_classes=>embedding_classes, :embedding_size=>embedding_size, :initializer=>initializer)...)
export EmbeddingWrapper
          

"""
"""
GRUCell(num_units::Union{Int64,Void}, input_size::Union{Int64,Void}=nothing, activation::Function=tanh_) = tf_rnn_cell.GRUCell(;Dict(:num_units=>num_units, :input_size=>input_size, :activation=>activation)...)
export GRUCell
          

"""
Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      num_proj: Python integer.  The dimension to project to.
      input_size: Deprecated and unused.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
InputProjectionWrapper(cell_::Any, num_proj::Union{Int64,Void}, input_size::Union{Int64,Void}=nothing) = tf_rnn_cell.InputProjectionWrapper(;Dict(:cell=>cell_, :num_proj=>num_proj, :input_size=>input_size)...)
export InputProjectionWrapper
          

"""
Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: Deprecated and unused.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
      provided, then the projected values are clipped elementwise to within
      `[-proj_clip, proj_clip]`.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
LSTMCell(num_units::Union{Int64,Void}, input_size::Union{Int64,Void}=nothing, use_peepholes::Bool=false, cell_clip::Any=nothing, initializer::Any=nothing, num_proj::Union{Int64,Void}=nothing, proj_clip::Any=nothing, num_unit_shards::Int64=1, num_proj_shards::Int64=1, forget_bias::Any=1.0, state_is_tuple::Any=false, activation::Function=tanh_) = tf_rnn_cell.LSTMCell(;Dict(:num_units=>num_units, :input_size=>input_size, :use_peepholes=>use_peepholes, :cell_clip=>cell_clip, :initializer=>initializer, :num_proj=>num_proj, :proj_clip=>proj_clip, :num_unit_shards=>num_unit_shards, :num_proj_shards=>num_proj_shards, :forget_bias=>forget_bias, :state_is_tuple=>state_is_tuple, :activation=>activation)...)
export LSTMCell
          

"""
Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  By default (False), the states are all
        concatenated along the column axis.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
MultiRNNCell(cells::Any, state_is_tuple::Any=false) = tf_rnn_cell.MultiRNNCell(;Dict(:cells=>cells, :state_is_tuple=>state_is_tuple)...)
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
Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A Tensor with type `float32`, `float64`, `int32`, `complex64`, `int64`,
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
    x: A Tensor or SparseTensor with type `float`, `double`, `int32`,
      `complex64`, `int64`, or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor or SparseTensor respectively with the same type as `x` if
    `x.dtype != qint32` otherwise the return type is `quint8`.
  """
tanh_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_rnn_cell.tanh(;Dict(:x=>x, :name=>name)...))
export tanh_
          end
