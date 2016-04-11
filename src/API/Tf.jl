"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.6.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module Tf
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow as tf
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Creates a new `DataType`.

    NOTE(mrry): In normal circumstances, you should not need to
    construct a `DataType` object directly. Instead, use the
    `tf.as_dtype()` function.

    Args:
      type_enum: A `types_pb2.DataType` enum value.

    Raises:
      TypeError: If `type_enum` is not a value `types_pb2.DataType`.

    """
DType(type_enum::Any) = tf.DType(;Dict(:type_enum=>type_enum)...)
export DType
          

"""
Creates a new Dimension with the given value."""
Dimension(value::Any) = tf.Dimension(;Dict(:value=>value)...)
export Dimension
          

"""
Creates a queue that dequeues elements in a first-in first-out order.

    A `FIFOQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `FIFOQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `dtypes`, and whose shapes are optionally described
    by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects,
        with the same length as `dtypes` or `None`.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
FIFOQueue(capacity::Any, dtypes::Any, shapes::Any=nothing, shared_name::Any=nothing, name::AbstractString="fifo_queue") = tf.FIFOQueue(;Dict(:capacity=>capacity, :dtypes=>dtypes, :shapes=>shapes, :shared_name=>shared_name, :name=>name)...)
export FIFOQueue
          

"""
Create a FixedLengthRecordReader.

    Args:
      record_bytes: An int.
      header_bytes: An optional int. Defaults to 0.
      footer_bytes: An optional int. Defaults to 0.
      name: A name for the operation (optional).
    """
FixedLengthRecordReader(record_bytes::Any, header_bytes::Any=nothing, footer_bytes::Any=nothing, name::Union{AbstractString,Void}=nothing) = tf.FixedLengthRecordReader(;Dict(:record_bytes=>record_bytes, :header_bytes=>header_bytes, :footer_bytes=>footer_bytes, :name=>name)...)
export FixedLengthRecordReader
          

"""
Creates a new, empty Graph."""
Graph() = tf.Graph(;Dict()...)
export Graph
          

"""
Create a IdentityReader.

    Args:
      name: A name for the operation (optional).
    """
IdentityReader(name::Union{AbstractString,Void}=nothing) = tf.IdentityReader(;Dict(:name=>name)...)
export IdentityReader
          

"""
Creates an `IndexedSlices`."""
IndexedSlices(values_::Any, indices::Any, dense_shape::Any=nothing) = tf.IndexedSlices(;Dict(:values=>values_, :indices=>indices, :dense_shape=>dense_shape)...)
export IndexedSlices
          

"""
Creates a new interactive TensorFlow session.

    If no `graph` argument is specified when constructing the session,
    the default graph will be launched in the session. If you are
    using more than one graph (created with `tf.Graph()` in the same
    process, you will have to use different sessions for each graph,
    but each graph can be used in multiple sessions. In this case, it
    is often clearer to pass the graph to be launched explicitly to
    the session constructor.

    Args:
      target: (Optional.) The execution engine to connect to.
        Defaults to using an in-process engine. At present, no value
        other than the empty string is supported.
      graph: (Optional.) The `Graph` to be launched (described above).
      config: (Optional) `ConfigProto` proto used to configure the session.
    """
InteractiveSession(target::Any="", graph::Any=nothing, config::Any=nothing) = Session(tf.InteractiveSession(;Dict(:target=>target, :graph=>graph, :config=>config)...))
export InteractiveSession
          

"""
Creates a new `OpError` indicating that a particular op failed.

    Args:
      node_def: The `graph_pb2.NodeDef` proto representing the op that failed.
      op: The `ops.Operation` that failed, if known; otherwise None.
      message: The message string describing the failure.
      error_code: The `error_codes_pb2.Code` describing the error.
    """
OpError(node_def::Any, op::Any, message::Any, error_code::Any) = tf.OpError(;Dict(:node_def=>node_def, :op=>op, :message=>message, :error_code=>error_code)...)
export OpError
          

"""
Constructs a queue object from a queue reference.

    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: Constraints on the shapes of tensors in an element:
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      queue_ref: The queue reference, i.e. the output of the queue op.
    """
QueueBase(dtypes::Any, shapes::Any, queue_ref::Any) = tf.QueueBase(;Dict(:dtypes=>dtypes, :shapes=>shapes, :queue_ref=>queue_ref)...)
export QueueBase
          

"""
Create a queue that dequeues elements in a random order.

    A `RandomShuffleQueue` has bounded capacity; supports multiple
    concurrent producers and consumers; and provides exactly-once
    delivery.

    A `RandomShuffleQueue` holds a list of up to `capacity`
    elements. Each element is a fixed-length tuple of tensors whose
    dtypes are described by `dtypes`, and whose shapes are optionally
    described by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    The `min_after_dequeue` argument allows the caller to specify a
    minimum number of elements that will remain in the queue after a
    `dequeue` or `dequeue_many` operation completes, to ensure a
    minimum level of mixing of elements. This invariant is maintained
    by blocking those operations until sufficient elements have been
    enqueued. The `min_after_dequeue` argument is ignored after the
    queue has been closed.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      min_after_dequeue: An integer (described above).
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects,
        with the same length as `dtypes` or `None`.
      seed: A Python integer. Used to create a random seed. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
RandomShuffleQueue(capacity::Any, min_after_dequeue::Any, dtypes::Any, shapes::Any=nothing, seed::Union{Int64,Void}=nothing, shared_name::Any=nothing, name::AbstractString="random_shuffle_queue") = tf.RandomShuffleQueue(;Dict(:capacity=>capacity, :min_after_dequeue=>min_after_dequeue, :dtypes=>dtypes, :shapes=>shapes, :seed=>seed, :shared_name=>shared_name, :name=>name)...)
export RandomShuffleQueue
          

"""
Creates a new ReaderBase.

    Args:
      reader_ref: The operation that implements the reader.
      supports_serialize: True if the reader implementation can
        serialize its state.
    """
ReaderBase(reader_ref::Any, supports_serialize::Any=false) = tf.ReaderBase(;Dict(:reader_ref=>reader_ref, :supports_serialize=>supports_serialize)...)
export ReaderBase
          

"""
Creates a new decorator with `op_type` as the Operation type.

    Args:
      op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
    """
RegisterGradient(op_type::Any) = tf.RegisterGradient(;Dict(:op_type=>op_type)...)
export RegisterGradient
          

"""
Saves the `op_type` as the `Operation` type."""
RegisterShape(op_type::Any) = tf.RegisterShape(;Dict(:op_type=>op_type)...)
export RegisterShape
          

"""
Create a TFRecordReader.

    Args:
      name: A name for the operation (optional).
    """
TFRecordReader(name::Union{AbstractString,Void}=nothing) = tf.TFRecordReader(;Dict(:name=>name)...)
export TFRecordReader
          

"""
Creates a new TensorShape with the given dimensions.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.
        DEPRECATED: A single integer is treated as a singleton list.
    """
TensorShape(dims::Any) = tf.TensorShape(;Dict(:dims=>dims)...)
export TensorShape
          

"""
Create a TextLineReader.

    Args:
      skip_header_lines: An optional int. Defaults to 0.  Number of lines
        to skip from the beginning of every file.
      name: A name for the operation (optional).
    """
TextLineReader(skip_header_lines::Any=nothing, name::Union{AbstractString,Void}=nothing) = tf.TextLineReader(;Dict(:skip_header_lines=>skip_header_lines, :name=>name)...)
export TextLineReader
          

"""
Create a WholeFileReader.

    Args:
      name: A name for the operation (optional).
    """
WholeFileReader(name::Union{AbstractString,Void}=nothing) = tf.WholeFileReader(;Dict(:name=>name)...)
export WholeFileReader
          

"""
Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
  """
Assert(condition::Any, data::Union{AbstractTensor,Void}, summarize::Union{AbstractTensor,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.Assert(;Dict(:condition=>condition, :data=>data, :summarize=>summarize, :name=>name)...)
export Assert
          

"""
Specifies that ops of type `op_type` do not have a defined gradient.

  This function is only used when defining a new op type. It may be
  used for ops such as `tf.size()` that are not differentiable.  For
  example:

  ```python
  tf.NoGradient("Size")
  ```

  Args:
    op_type: The string type of an operation. This corresponds to the
      `OpDef.name` field for the proto that defines the operation.

  Raises:
    TypeError: If `op_type` is not a string.

  """
NoGradient(op_type::Union{Dtype,Void}) = tf.NoGradient(;Dict(:op_type=>op_type)...)
export NoGradient
          

"""
Prints a list of tensors.

  This is an identity op with the side effect of printing `data` when
  evaluating.

  Args:
    input_: A tensor passed through this op.
    data: A list of tensors to print out when op is evaluated.
    message: A string, prefix of the error message.
    first_n: Only log `first_n` number of times. Negative numbers log always;
             this is the default.
    summarize: Only print this many entries of each tensor. If None, then a
               maximum of 3 elements are printed per input tensor.
    name: A name for the operation (optional).

  Returns:
    Same tensor as `input_`.
  """
Print(input_::Union{AbstractTensor,Void}, data::Union{AbstractTensor,Void}, message::Any=nothing, first_n::Any=nothing, summarize::Union{AbstractTensor,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.Print(;Dict(:input_=>input_, :data=>data, :message=>message, :first_n=>first_n, :summarize=>summarize, :name=>name)...))
export Print
          

"""
Computes the absolute value of a tensor.

  Given a tensor of real numbers `x`, this operation returns a tensor
  containing the absolute value of each element in `x`. For example, if x is
  an input element and y is an output element, this operation computes
  \\(y = |x|\\).

  See [`tf.complex_abs()`](#tf_complex_abs) to compute the absolute value of a complex
  number.

  Args:
    x: A `Tensor` of type `float`, `double`, `int32`, or `int64`.
    name: A name for the operation (optional).

  Returns:
     A `Tensor` the same size and type as `x` with absolute values.
  """
abs_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.abs(;Dict(:x=>x, :name=>name)...))
export abs_
          

"""
Returns the element-wise sum of a list of tensors.

  Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
  otherwise, these are inferred.

  For example:

  ```python
  # tensor 'a' is [[1, 2], [3, 4]
  # tensor `b` is [[5, 0], [0, 6]]
  tf.accumulate_n([a, b, a]) ==> [[7, 4], [6, 14]]

  # Explicitly pass shape and type
  tf.accumulate_n([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)
    ==> [[7, 4], [6, 14]]
  ```

  Args:
    inputs: A list of `Tensor` objects, each with same shape and type.
    shape: Shape of elements of `inputs`.
    tensor_dtype: The type of `inputs`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  """
accumulate_n(inputs::Union{AbstractTensor,Void}, shape::Union{AbstractTensor,DimsType,Void}=nothing, tensor_dtype::Union{Dtype,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.accumulate_n(;Dict(:inputs=>inputs, :shape=>shape, :tensor_dtype=>tensor_dtype, :name=>name)...))
export accumulate_n
          

"""
Returns x + y element-wise.

  *NOTE*: Add supports broadcasting. AddN does not.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `string`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
add(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.add(;Dict(:x=>x, :y=>y, :name=>name)...))
export add
          

"""
Connect a `check_numerics` to every floating point tensor.

  `check_numerics` operations themselves are added for each `float` or `double`
  tensor in the graph. For all ops in the graph, the `check_numerics` op for
  all of its (`float` or `double`) inputs is guaranteed to run before the
  `check_numerics` op on any of its outputs.

  Returns:
    A `group` op depending on all `check_numerics` ops added.
  """
add_check_numerics_ops() = tf.add_check_numerics_ops(;Dict()...)
export add_check_numerics_ops
          

"""
Add all input tensors element wise.

  Args:
    inputs: A list of at least 1 `Tensor` objects of the same type in: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Must all be the same size and shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
add_n(inputs::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.add_n(;Dict(:inputs=>inputs, :name=>name)...))
export add_n
          

"""
Wrapper for `Graph.add_to_collection()` using the default graph.

  See [`Graph.add_to_collection()`](../../api_docs/python/framework.md#Graph.add_to_collection)
  for more details.

  Args:
    name: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collection.
  """
add_to_collection(name::Union{AbstractString,Void}, value::Any) = tf.add_to_collection(;Dict(:name=>name, :value=>value)...)
export add_to_collection
          

"""
Returns all variables collected in the graph.

  The `Variable()` constructor automatically adds new variables to the graph
  collection `GraphKeys.VARIABLES`. This convenience function returns the
  contents of that collection.

  Returns:
    A list of `Variable` objects.
  """
all_variables() = tf.all_variables(;Dict()...)
export all_variables
          

"""
Returns the index with the largest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
    dimension: A `Tensor` of type `int32`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
arg_max(input::Union{AbstractTensor,Void}, dimension::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.arg_max(;Dict(:input=>input, :dimension=>dimension, :name=>name)...))
export arg_max
          

"""
Returns the index with the smallest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
    dimension: A `Tensor` of type `int32`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
arg_min(input::Union{AbstractTensor,Void}, dimension::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.arg_min(;Dict(:input=>input, :dimension=>dimension, :name=>name)...))
export arg_min
          

"""
Returns the index with the largest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
    dimension: A `Tensor` of type `int32`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
arg_max(input::Union{AbstractTensor,Void}, dimension::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.arg_max(;Dict(:input=>input, :dimension=>dimension, :name=>name)...))
export arg_max
          

"""
Returns the index with the smallest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
    dimension: A `Tensor` of type `int32`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
arg_min(input::Union{AbstractTensor,Void}, dimension::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.arg_min(;Dict(:input=>input, :dimension=>dimension, :name=>name)...))
export arg_min
          

"""
Converts the given `type_value` to a `DType`.

  Args:
    type_value: A value that can be converted to a `tf.DType`
      object. This may currently be a `tf.DType` object, a
      [`DataType` enum](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/types.proto),
      a string type name, or a `numpy.dtype`.

  Returns:
    A `DType` corresponding to `type_value`.

  Raises:
    TypeError: If `type_value` cannot be converted to a `DType`.
  """
as_dtype(type_value::Any) = Dtype(tf.as_dtype(;Dict(:type_value=>type_value)...))
export as_dtype
          

"""
Returns an Op to check if variables are initialized.

  When run, the returned Op will raise the exception `FailedPreconditionError`
  if any of the variables has not yet been initialized.

  Note: This function is implemented by trying to fetch the values of the
  variables. If one of the variables is not initialized a message may be
  logged by the C++ runtime. This is expected.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the
      value of `all_variables().`

  Returns:
    An Op, or None if there are no variables.
  """
assert_variables_initialized(var_list::Any=nothing) = tf.assert_variables_initialized(;Dict(:var_list=>var_list)...)
export assert_variables_initialized
          

"""
Update 'ref' by assigning 'value' to it.

  This operation outputs "ref" after the assignment is done.
  This makes it easier to chain operations that need to use the reset value.

  Args:
    ref: A mutable `Tensor`.
      Should be from a `Variable` node. May be uninitialized.
    value: A `Tensor`. Must have the same type as `ref`.
      The value to be assigned to the variable.
    validate_shape: An optional `bool`. Defaults to `True`.
      If true, the operation will validate that the shape
      of 'value' matches the shape of the Tensor being assigned to.  If false,
      'ref' will take on the shape of 'value'.
    use_locking: An optional `bool`. Defaults to `True`.
      If True, the assignment will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".  Returned as a convenience for operations that want
    to use the new value after the variable has been reset.
  """
assign(ref::Union{AbstractTensor,Void}, value::Union{AbstractTensor,Void}, validate_shape::Union{Bool,Void}=nothing, use_locking::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.assign(;Dict(:ref=>ref, :value=>value, :validate_shape=>validate_shape, :use_locking=>use_locking, :name=>name)...)
export assign
          

"""
Update 'ref' by adding 'value' to it.

  This operation outputs "ref" after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a `Variable` node.
    value: A `Tensor`. Must have the same type as `ref`.
      The value to be added to the variable.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the addition will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".  Returned as a convenience for operations that want
    to use the new value after the variable has been updated.
  """
assign_add(ref::Union{AbstractTensor,Void}, value::Union{AbstractTensor,Void}, use_locking::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.assign_add(;Dict(:ref=>ref, :value=>value, :use_locking=>use_locking, :name=>name)...)
export assign_add
          

"""
Update 'ref' by subtracting 'value' from it.

  This operation outputs "ref" after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a `Variable` node.
    value: A `Tensor`. Must have the same type as `ref`.
      The value to be subtracted to the variable.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".  Returned as a convenience for operations that want
    to use the new value after the variable has been updated.
  """
assign_sub(ref::Union{AbstractTensor,Void}, value::Union{AbstractTensor,Void}, use_locking::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.assign_sub(;Dict(:ref=>ref, :value=>value, :use_locking=>use_locking, :name=>name)...)
export assign_sub
          

"""
Calculates the Cholesky decomposition of a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix Cholesky
  decomposition above. The output is a tensor of the same shape as the input
  containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
batch_cholesky(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.batch_cholesky(;Dict(:input=>input, :name=>name)...))
export batch_cholesky
          

"""
Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`.
      3-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      3-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
    3-D or higher with shape `[..., r_o, c_o]`
  """
_batch_mat_mul(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, adj_x::Union{Bool,Void}=nothing, adj_y::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf._batch_mat_mul(;Dict(:x=>x, :y=>y, :adj_x=>adj_x, :adj_y=>adj_y, :name=>name)...))
export _batch_mat_mul
          

"""
Calculates the determinants for a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a 1-D tensor containing the determinants
  for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[...]`.
  """
batch_matrix_determinant(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.batch_matrix_determinant(;Dict(:input=>input, :name=>name)...))
export batch_matrix_determinant
          

"""
Calculates the inverse of square invertible matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the inverse for all input submatrices `[..., :, :]`.

  The op uses the Cholesky decomposition if the matrices are symmetric positive
  definite and LU decomposition with partial pivoting otherwise.

  If a matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
batch_matrix_inverse(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.batch_matrix_inverse(;Dict(:input=>input, :name=>name)...))
export batch_matrix_inverse
          

"""
Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix
  SelfAdjointEig.

  The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
  eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.
  """
batch_self_adjoint_eig(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.batch_self_adjoint_eig(;Dict(:input=>input, :name=>name)...))
export batch_self_adjoint_eig
          

"""
Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor`) to `dtype`.

  For example:

  ```python
  # tensor `a` is [1.8, 2.2], dtype=tf.float
  tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
  ```

  Args:
    x: A `Tensor` or `SparseTensor`.
    dtype: The destination type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x`.

  Raises:
    TypeError: If `x` cannot be cast to the `dtype`.
  """
cast(x::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.cast(;Dict(:x=>x, :dtype=>dtype, :name=>name)...))
export cast
          

"""
Returns element-wise smallest integer in not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
ceil_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.ceil(;Dict(:x=>x, :name=>name)...))
export ceil_
          

"""
Checks a tensor for NaN and Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

  Args:
    tensor: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
check_numerics(tensor::Union{AbstractTensor,Void}, message::Union{AbstractString,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.check_numerics(;Dict(:tensor=>tensor, :message=>message, :name=>name)...))
export check_numerics
          

"""
Calculates the Cholesky decomposition of a square matrix.

  The input has to be symmetric and positive definite. Only the lower-triangular
  part of the input will be used for this operation. The upper-triangular part
  will not be read.

  The result is the lower-triangular matrix of the Cholesky decomposition of the
  input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M, M]`.
  """
cholesky(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.cholesky(;Dict(:input=>input, :name=>name)...))
export cholesky
          

"""
Clips tensor values to a maximum average L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its average L2-norm is less than or equal to
  `clip_norm`. Specifically, if the average L2-norm is already less than or
  equal to `clip_norm`, then `t` is not modified. If the average L2-norm is
  greater than `clip_norm`, then this operation returns a tensor of the same
  type and shape as `t` with its values set to:

  `t * clip_norm / l2norm_avg(t)`

  In this case, the average L2-norm of the output tensor is `clip_norm`.

  This operation is typically used to clip gradients before applying them with
  an optimizer.

  Args:
    t: A `Tensor`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
clip_by_average_norm(t::Union{AbstractTensor,Void}, clip_norm::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.clip_by_average_norm(;Dict(:t=>t, :clip_norm=>clip_norm, :name=>name)...))
export clip_by_average_norm
          

"""
Clips values of multiple tensors by the ratio of the sum of their norms.

  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
  if you've already computed the global norm for `t_list`, you can specify
  the global norm with `use_norm`.

  To perform the clipping, the values `t_list[i]` are set to:

      t_list[i] * clip_norm / max(global_norm, clip_norm)

  where:

      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.

  Any of the entries of `t_list` that are of type `None` are ignored.

  This is the correct way to perform gradient clipping (for example, see
  R. Pascanu, T. Mikolov, and Y. Bengio, "On the difficulty of training
  Recurrent Neural Networks".  http://arxiv.org/abs/1211.5063)

  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).

  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.

  Raises:
    TypeError: If `t_list` is not a sequence.
  """
clip_by_global_norm(t_list::Union{AbstractTensor,Void}, clip_norm::Union{AbstractTensor,Void}, use_norm::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.clip_by_global_norm(;Dict(:t_list=>t_list, :clip_norm=>clip_norm, :use_norm=>use_norm, :name=>name)...))
export clip_by_global_norm
          

"""
Clips tensor values to a maximum L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`.
  Specifically, if the L2-norm is already less than or equal to `clip_norm`,
  then `t` is not modified. If the L2-norm is greater than `clip_norm`, then
  this operation returns a tensor of the same type and shape as `t` with its
  values set to:

  `t * clip_norm / l2norm(t)`

  In this case, the L2-norm of the output tensor is `clip_norm`.

  This operation is typically used to clip gradients before applying them with
  an optimizer.

  Args:
    t: A `Tensor`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
clip_by_norm(t::Union{AbstractTensor,Void}, clip_norm::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.clip_by_norm(;Dict(:t=>t, :clip_norm=>clip_norm, :name=>name)...))
export clip_by_norm
          

"""
Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Args:
    t: A `Tensor`.
    clip_value_min: A 0-D (scalar) `Tensor`. The minimum value to clip by.
    clip_value_max: A 0-D (scalar) `Tensor`. The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor`.
  """
clip_by_value(t::Union{AbstractTensor,Void}, clip_value_min::Union{AbstractTensor,Void}, clip_value_max::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.clip_by_value(;Dict(:t=>t, :clip_value_min=>clip_value_min, :clip_value_max=>clip_value_max, :name=>name)...))
export clip_by_value
          

"""
Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation computes complex numbers elementwise of the form \\(a + bj\\),
  where *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must be the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.74j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor` of type `float`.
    imag: A `Tensor` of type `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
complex_(real_::Union{AbstractTensor,Void}, imag_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.complex(;Dict(:real=>real_, :imag=>imag_, :name=>name)...))
export complex_
          

"""
Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` that is the absolute value of each element in `x`. All elements in `x`
  must be complex numbers of the form \\(a + bj\\). The absolute value is
  computed as \\( \sqrt{a^2 + b^2}\\).

  For example:

  ```
  # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
  tf.complex_abs(x) ==> [5.25594902, 6.60492229]
  ```

  Args:
    x: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
complex_abs(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.complex_abs(;Dict(:x=>x, :name=>name)...))
export complex_abs
          

"""
Concatenates tensors along one dimension.

  Concatenates the list of tensors `values` along dimension `concat_dim`.  If
  `values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]`, the concatenated
  result has shape

      [D0, D1, ... Rconcat_dim, ...Dn]

  where

      Rconcat_dim = sum(Dconcat_dim(i))

  That is, the data from the input tensors is joined along the `concat_dim`
  dimension.

  The number of dimensions of the input tensors must match, and all dimensions
  except `concat_dim` must be equal.

  For example:

  ```python
  t1 = [[1, 2, 3], [4, 5, 6]]
  t2 = [[7, 8, 9], [10, 11, 12]]
  tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

  # tensor t3 with shape [2, 3]
  # tensor t4 with shape [2, 3]
  tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
  tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
  ```

  Args:
    concat_dim: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
    values: A list of `Tensor` objects or a single `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` resulting from concatenation of the input tensors.
  """
concat(concat_dim::Union{AbstractTensor,Void}, values_::Union{AbstractTensor,Void}, name::AbstractString="concat") = Tensor(tf.concat(;Dict(:concat_dim=>concat_dim, :values=>values_, :name=>name)...))
export concat
          

"""
Returns the complex conjugate of a complex number.

  Given a tensor `in` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `in`. The
  complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
  part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  ```
  # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
  ```

  Args:
    in_: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
conj_(in_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.conj(;Dict(:in_=>in_, :name=>name)...))
export conj_
          

"""
Creates a constant tensor.

   The resulting tensor is populated with values of type `dtype`, as
   specified by arguments `value` and (optionally) `shape` (see examples
   below).

   The argument `value` can be a constant value, or a list of values of type
   `dtype`. If `value` is a list, then the length of the list must be less
   than or equal to the number of elements implied by the `shape` argument (if
   specified). In the case where the list length is less than the number of
   elements specified by `shape`, the last element in the list will be used
   to fill the remaining entries.

   The argument `shape` is optional. If present, it specifies the dimensions
   of the resulting tensor. If not present, then the tensor is a scalar (0-D)
   if `value` is a scalar, or 1-D otherwise.

   If the argument `dtype` is not specified, then the type is inferred from
   the type of `value`.

   For example:

   ```python
   # Constant 1-D Tensor populated with value list.
   tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

   # Constant 2-D tensor populated with scalar value -1.
   tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                [-1. -1. -1.]]
   ```

  Args:
    value:     A constant value (or list) of output type `dtype`.

    dtype:     The type of the elements of the resulting tensor.

    shape:     Optional dimensions of resulting tensor.

    name:      Optional name for the tensor.

  Returns:
    A Constant Tensor.
  """
constant(value::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}=nothing, shape::Union{AbstractTensor,DimsType,Void}=nothing, name::AbstractString="Const") = Tensor(tf.constant(;Dict(:value=>value, :dtype=>dtype, :shape=>shape, :name=>name)...))
export constant
          

"""
Returns an initializer that generates tensors with a single value.

  Args:
    value: A Python scalar. All elements of the initialized variable
      will be set to this value.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a single value.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
constant_initializer(value::Any=0.0, dtype::Dtype=DT_FLOAT32) = Tensor(tf.constant_initializer(;Dict(:value=>value, :dtype=>dtype)...))
export constant_initializer
          

"""
Wrapper for `Graph.control_dependencies()` using the default graph.

  See [`Graph.control_dependencies()`](../../api_docs/python/framework.md#Graph.control_dependencies)
  for more details.

  Args:
    control_inputs: A list of `Operation` or `Tensor` objects which
      must be executed or computed before running the operations
      defined in the context.  Can also be `None` to clear the control
      dependencies.

  Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
  """
control_dependencies(control_inputs::Union{AbstractTensor,Void}) = tf.control_dependencies(;Dict(:control_inputs=>control_inputs)...)
export control_dependencies
          

"""
Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars. For example:

  ```python
  import numpy as np
  array = np.random.rand(32, 100, 100)

  def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

  # The following calls are equivalent.
  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  ```

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: True if we want the result as a ref tensor.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value`.
    RuntimeError: If a registered conversion function returns an invalid value.

  """
convert_to_tensor(value::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}=nothing, name::Union{AbstractString,Void}=nothing, as_ref::AbstractTensor=false) = Tensor(tf.convert_to_tensor(;Dict(:value=>value, :dtype=>dtype, :name=>name, :as_ref=>as_ref)...))
export convert_to_tensor
          

"""
Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices` or an object that can be consumed by
      `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    An `Tensor` or an `IndexedSlices` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
convert_to_tensor_or_indexed_slices(value::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}=nothing, name::Union{AbstractString,Void}=nothing, as_ref::AbstractTensor=false) = Tensor(tf.convert_to_tensor_or_indexed_slices(;Dict(:value=>value, :dtype=>dtype, :name=>name, :as_ref=>as_ref)...))
export convert_to_tensor_or_indexed_slices
          

"""
Computes cos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
cos_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.cos(;Dict(:x=>x, :name=>name)...))
export cos_
          

"""
Increments 'ref' until it reaches 'limit'.

  This operation outputs "ref" after the update is done.  This makes it
  easier to chain operations that need to use the updated value.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should be from a scalar `Variable` node.
    limit: An `int`.
      If incrementing ref would bring it above limit, instead generates an
      'OutOfRange' error.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `ref`.
    A copy of the input before increment. If nothing else modifies the
    input, the values produced will all be distinct.
  """
count_up_to(ref::Union{AbstractTensor,Void}, limit::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf.count_up_to(;Dict(:ref=>ref, :limit=>limit, :name=>name)...))
export count_up_to
          

"""
Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or empty if the column is required.
    field_delim: An optional `string`. Defaults to `","`.
      delimiter to separate fields in a record.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.
  """
decode_csv(records::Union{AbstractTensor,Void}, record_defaults::Union{AbstractTensor,Void}, field_delim::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.decode_csv(;Dict(:records=>records, :record_defaults=>record_defaults, :field_delim=>field_delim, :name=>name)...))
export decode_csv
          

"""
Reinterpret the bytes of a string as a vector of numbers.

  Args:
    bytes: A `Tensor` of type `string`.
      All the elements must have the same length.
    out_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`.
    little_endian: An optional `bool`. Defaults to `True`.
      Whether the input `bytes` are in little-endian order.
      Ignored for `out_type` values that are stored in a single byte like
      `uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
    A Tensor with one more dimension than the input `bytes`.  The
    added dimension will have size equal to the length of the elements
    of `bytes` divided by the number of bytes to represent `out_type`.
  """
decode_raw(bytes::Union{AbstractTensor,Void}, out_type::Any, little_endian::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.decode_raw(;Dict(:bytes=>bytes, :out_type=>out_type, :little_endian=>little_endian, :name=>name)...))
export decode_raw
          

"""
Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `serialize_sparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects (they have been
  concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `sparse_reorder` to restore index ordering.

  For example, if the serialized input is a `[2, 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: 2-D `Tensor` of type `string` of shape `[N, 3]`.
      The serialized and packed `SparseTensor' objects.
    dtype: The `dtype` of the serialized `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor`s,
    concatenated along the `SparseTensor`s' first dimension.

    All of the serialized `SparseTensor`s must have had the same rank and type.
  """
deserialize_many_sparse(serialized_sparse::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.deserialize_many_sparse(;Dict(:serialized_sparse=>serialized_sparse, :dtype=>dtype, :name=>name)...))
export deserialize_many_sparse
          

"""
Wrapper for `Graph.device()` using the default graph.

  See
  [`Graph.device()`](../../api_docs/python/framework.md#Graph.device)
  for more details.

  Args:
    device_name_or_function: The device name or function to use in
      the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.
  """
device(dev::Any) = tf.device(;Dict(:dev=>dev)...)
export device
          

"""
Returns a diagonal tensor with a given diagonal values.

  Given a `diagonal`, this operation returns a tensor with the `diagonal` and
  everything else padded with zeros. The diagonal is computed as follows:

  Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
  rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

  `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

  For example:

  ```prettyprint
  # 'diagonal' is [1, 2, 3, 4]
  tf.diag(diagonal) ==> [[1, 0, 0, 0]
                         [0, 2, 0, 0]
                         [0, 0, 3, 0]
                         [0, 0, 0, 4]]
  ```

  Args:
    diagonal: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Rank k tensor where k is at most 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
diag_(diagonal::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.diag(;Dict(:diagonal=>diagonal, :name=>name)...))
export diag_
          

"""
Returns x / y element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
div_(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.div(;Dict(:x=>x, :y=>y, :name=>name)...))
export div_
          

"""
Partitions `data` into `num_partitions` tensors using indices from `partitions`.

  For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
  becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
  are placed in `outputs[i]` in lexicographic order of `js`, and the first
  dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
  In detail,

      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

  `data.shape` must start with `partitions.shape`.

  For example:

      # Scalar partitions
      partitions = 1
      num_partitions = 2
      data = [10, 20]
      outputs[0] = []  # Empty with shape [0, 2]
      outputs[1] = [[10, 20]]

      # Vector partitions
      partitions = [0, 0, 1, 1, 0]
      num_partitions = 2
      data = [10, 20, 30, 40, 50]
      outputs[0] = [10, 20, 50]
      outputs[1] = [30, 40]

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/DynamicPartition.png" alt>
  </div>

  Args:
    data: A `Tensor`.
    partitions: A `Tensor` of type `int32`.
      Any shape.  Indices in the range `[0, num_partitions)`.
    num_partitions: An `int` that is `>= 1`.
      The number of partitions to output.
    name: A name for the operation (optional).

  Returns:
    A list of `num_partitions` `Tensor` objects of the same type as data.
  """
dynamic_partition(data::Union{AbstractTensor,Void}, partitions_::Union{AbstractTensor,Void}, num_partitions::Union{Int64,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.dynamic_partition(;Dict(:data=>data, :partitions=>partitions_, :num_partitions=>num_partitions, :name=>name)...))
export dynamic_partition
          

"""
Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

  For example, if each `indices[m]` is scalar or vector, we have

      # Scalar indices
      merged[indices[m], ...] = data[m][...]

      # Vector indices
      merged[indices[m][i], ...] = data[m][i, ...]

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result.

  For example:

      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 2 `Tensor` objects of type `int32`.
    data: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
dynamic_stitch(indices::Union{AbstractTensor,Void}, data::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.dynamic_stitch(;Dict(:indices=>indices, :data=>data, :name=>name)...))
export dynamic_stitch
          

"""
Computes the Levenshtein distance between sequences.

  This operation takes variable-length sequences (`hypothesis` and `truth`),
  each provided as a `SparseTensor`, and computes the Levenshtein distance.
  You can normalize the edit distance by length of `truth` by setting
  `normalize` to true.

  For example, given the following input:

  ```python
  # 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
  #   (0,0) = ["a"]
  #   (1,0) = ["b"]
  hypothesis = tf.SparseTensor(
      [[0, 0, 0],
       [1, 0, 0]],
      ["a", "b"]
      (2, 1, 1))

  # 'truth' is a tensor of shape `[2, 2]` with variable-length values:
  #   (0,0) = []
  #   (0,1) = ["a"]
  #   (1,0) = ["b", "c"]
  #   (1,1) = ["a"]
  truth = tf.SparseTensor(
      [[0, 1, 0],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0]]
      ["a", "b", "c", "a"],
      (2, 2, 2))

  normalize = True
  ```

  This operation would return the following:

  ```python
  # 'output' is a tensor of shape `[2, 2]` with edit distances normalized
  # by 'truth' lengths.
  output ==> [[inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
             [0.5, 1.0]]  # (1,0): addition, (1,1): no hypothesis
  ```

  Args:
    hypothesis: A `SparseTensor` containing hypothesis sequences.
    truth: A `SparseTensor` containing truth sequences.
    normalize: A `bool`. If `True`, normalizes the Levenshtein distance by
      length of `truth.`
    name: A name for the operation (optional).

  Returns:
    A dense `Tensor` with rank `R - 1`, where R is the rank of the
    `SparseTensor` inputs `hypothesis` and `truth`.

  Raises:
    TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
  """
edit_distance(hypothesis::Union{AbstractTensor,Void}, truth::Union{AbstractTensor,Void}, normalize::Any=true, name::AbstractString="edit_distance") = Tensor(tf.edit_distance(;Dict(:hypothesis=>hypothesis, :truth=>truth, :normalize=>normalize, :name=>name)...))
export edit_distance
          

"""
Returns the truth value of (x == y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
equal(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.equal(;Dict(:x=>x, :y=>y, :name=>name)...))
export equal
          

"""
Computes Gauss error function of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
erf_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.erf(;Dict(:x=>x, :name=>name)...))
export erf_
          

"""
Computes complementary error function of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
erfc_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.erfc(;Dict(:x=>x, :name=>name)...))
export erfc_
          

"""
Computes exponential of x element-wise.  \\(y = e^x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
exp_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.exp(;Dict(:x=>x, :name=>name)...))
export exp_
          

"""
Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
  zero; if you specify a negative number for `dim` it is counted backward from
  the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```prettyprint
  # 't' is a tensor of shape [2]
  shape(expand_dims(t, 0)) ==> [1, 2]
  shape(expand_dims(t, 1)) ==> [2, 1]
  shape(expand_dims(t, -1)) ==> [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
  shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
  shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    dim: A `Tensor` of type `int32`.
      0-D (scalar). Specifies the dimension index at which to
      expand the shape of `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but its shape has an additional
    dimension of size 1 added.
  """
expand_dims(input::Union{AbstractTensor,Void}, dim::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.expand_dims(;Dict(:input=>input, :dim=>dim, :name=>name)...))
export expand_dims
          

"""
Compute the 2-dimensional discrete Fourier Transform.

  Args:
    in_: A `Tensor` of type `complex64`. A complex64 matrix.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`. The 2D Fourier Transform of `in`.
  """
fft2d(in_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.fft2d(;Dict(:in_=>in_, :name=>name)...))
export fft2d
          

"""
Creates a tensor filled with a scalar value.

  This operation creates a tensor of shape `dims` and fills it with `value`.

  For example:

  ```prettyprint
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
  ```

  Args:
    dims: A `Tensor` of type `int32`.
      1-D. Represents the shape of the output tensor.
    value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
fill_(dims::Union{AbstractTensor,Void}, value::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.fill(;Dict(:dims=>dims, :value=>value, :name=>name)...))
export fill_
          

"""
Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
floor_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.floor(;Dict(:x=>x, :name=>name)...))
export floor_
          

"""
Divides `x / y` elementwise, rounding down for floating point.

  The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
  floating point arguments so that the result is always an integer (though
  possibly an integer represented as floating point).  This op is generated by
  `x // y` floor division in Python 3 and in Python 2.7 with
  `from __future__ import division`.

  Note that for efficiency, `floordiv` uses C semantics for negative numbers
  (unlike Python and Numpy).

  `x` and `y` must have the same type, and the result will have the same type
  as well.

  Args:
    x: `Tensor` numerator of real numeric type.
    y: `Tensor` denominator of real numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` rounded down (except possibly towards zero for negative integers).

  Raises:
    TypeError: If the inputs are complex.
  """
floordiv(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = tf.floordiv(;Dict(:x=>x, :y=>y, :name=>name)...)
export floordiv
          

"""
Gather slices from `params` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

      # Scalar indices
      output[:, ..., :] = params[indices, :, ... :]

      # Vector indices
      output[i, :, ..., :] = params[indices[i], :, ... :]

      # Higher rank indices
      output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

  If `indices` is a permutation and `len(indices) == params.shape[0]` then
  this operation will permute `params` accordingly.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/Gather.png" alt>
  </div>

  Args:
    params: A `Tensor`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
gather(params::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.gather(;Dict(:params=>params, :indices=>indices, :name=>name)...))
export gather
          

"""
Wrapper for `Graph.get_collection()` using the default graph.

  See [`Graph.get_collection()`](../../api_docs/python/framework.md#Graph.get_collection)
  for more details.

  Args:
    key: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    scope: (Optional.) If supplied, the resulting list is filtered to include
      only items whose name begins with this string.

  Returns:
    The list of values in the collection with the given `name`, or
    an empty list if no value has been added to that collection. The
    list contains the values in the order under which they were
    collected.
  """
get_collection(key::Any, scope::Any=nothing) = AbstractString(tf.get_collection(;Dict(:key=>key, :scope=>scope)...))
export get_collection
          

"""
Returns the default graph for the current thread.

  The returned graph will be the innermost graph on which a
  `Graph.as_default()` context has been entered, or a global default
  graph if none has been explicitly created.

  NOTE: The default graph is a property of the current thread. If you
  create a new thread, and wish to use the default graph in that
  thread, you must explicitly add a `with g.as_default():` in that
  thread's function.

  Returns:
    The default `Graph` being used in the current thread.
  """
get_default_graph() = tf.get_default_graph(;Dict()...)
export get_default_graph
          

"""
Returns the default session for the current thread.

  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.

  NOTE: The default session is a property of the current thread. If you
  create a new thread, and wish to use the default session in that
  thread, you must explicitly add a `with sess.as_default():` in that
  thread's function.

  Returns:
    The default `Session` being used in the current thread.
  """
get_default_session() = tf.get_default_session(;Dict()...)
export get_default_session
          

"""
Returns the local seeds an operation should use given an op-specific seed.

  Given operation-specific seed, `op_seed`, this helper function returns two
  seeds derived from graph-level and op-level seeds. Many random operations
  internally use the two seeds to allow user to change the seed globally for a
  graph, or for only specific operations.

  For details on how the graph-level seed interacts with op seeds, see
  [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed).

  Args:
    op_seed: integer.

  Returns:
    A tuple of two integers that should be used for the local seed of this
    operation.
  """
get_seed(op_seed::Any) = tf.get_seed(;Dict(:op_seed=>op_seed)...)
export get_seed
          

"""
Gets an existing variable with these parameters or create a new one.

  This function prefixes the name with the current variable scope
  and performs reuse checks. See the
  [Variable Scope How To](../../how_tos/variable_scope/index.md)
  for an extensive description of how reusing works. Here is a basic example:

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
      w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
  with tf.variable_scope("foo", reuse=True)
      v1 = tf.get_variable("v")  # The same as v above.
  ```

  If initializer is `None` (the default), the default initializer passed in
  the constructor is used. If that one is `None` too, a
  `UniformUnitScalingInitializer` will be used.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: List of graph collections keys to add the Variable to.
      Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).

  Returns:
    The created or existing variable.

  Raises:
    ValueError: when creating a new variable and shape is not declared,
      or when violating reuse during variable creation. Reuse is set inside
      `variable_scope`.
  """
get_variable(name::Union{AbstractString,Void}, shape::Union{AbstractTensor,DimsType,Void}=nothing, dtype::Dtype=DT_FLOAT32, initializer::Any=nothing, trainable::Bool=true, collections::Any=nothing) = tf.get_variable(;Dict(:name=>name, :shape=>shape, :dtype=>dtype, :initializer=>initializer, :trainable=>trainable, :collections=>collections)...)
export get_variable
          

"""
Returns the current variable scope."""
get_variable_scope() = tf.get_variable_scope(;Dict()...)
export get_variable_scope
          

"""
Computes the global norm of multiple tensors.

  Given a tuple or list of tensors `t_list`, this operation returns the
  global norm of the elements in all tensors in `t_list`. The global norm is
  computed as:

  `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`

  Any entries in `t_list` that are of type None are ignored.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    name: A name for the operation (optional).

  Returns:
    A 0-D (scalar) `Tensor` of type `float`.

  Raises:
    TypeError: If `t_list` is not a sequence.
  """
global_norm(t_list::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.global_norm(;Dict(:t_list=>t_list, :name=>name)...))
export global_norm
          

"""
Constructs symbolic partial derivatives of `ys` w.r.t. x in `xs`.

  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.

  `gradients()` adds ops to the graph to output the partial
  derivatives of `ys` with respect to `xs`.  It returns a list of
  `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
  for y in `ys`.

  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
      `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'gradients'.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.
    gate_gradients: If True, add a tuple around the gradients returned
      for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.

  Returns:
    A list of `sum(dy/dx)` for each x in `xs`.

  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.

  """
gradients(ys::Union{AbstractTensor,Void}, xs::Union{AbstractTensor,Void}, grad_ys::Any=nothing, name::AbstractString="gradients", colocate_gradients_with_ops::Bool=false, gate_gradients::Bool=false, aggregation_method::Any=nothing) = tf.gradients(;Dict(:ys=>ys, :xs=>xs, :grad_ys=>grad_ys, :name=>name, :colocate_gradients_with_ops=>colocate_gradients_with_ops, :gate_gradients=>gate_gradients, :aggregation_method=>aggregation_method)...)
export gradients
          

"""
Returns the truth value of (x > y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
greater(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.greater(;Dict(:x=>x, :y=>y, :name=>name)...))
export greater
          

"""
Returns the truth value of (x >= y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
greater_equal(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.greater_equal(;Dict(:x=>x, :y=>y, :name=>name)...))
export greater_equal
          

"""
Create an op that groups multiple operations.

  When this op finishes, all ops in `input` have finished. This op has no
  output.

  See also `tuple` and `with_dependencies`.

  Args:
    *inputs: One or more tensors to group.
    **kwargs: Optional parameters to pass when constructing the NodeDef.
    name: A name for this operation (optional).

  Returns:
    An Operation that executes all its inputs.

  Raises:
    ValueError: If an unknown keyword argument is provided, or if there are
                no inputs.
  """
group() = tf.group(;Dict()...)
export group
          

"""
Outputs a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `OutOfRange` error if any value is not finite.

  Args:
    tag: A `string` `Tensor`. 0-D.  Tag to use for the summary value.
    values: A real numeric `Tensor`. Any shape. Values to use to
      build the histogram.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
histogram_summary(tag::Union{AbstractTensor,Void}, values_::Union{AbstractTensor,Void}, collections::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.histogram_summary(;Dict(:tag=>tag, :values=>values_, :collections=>collections, :name=>name)...))
export histogram_summary
          

"""
Return a tensor with the same shape and contents as the input tensor or value.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
identity_(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.identity(;Dict(:input=>input, :name=>name)...))
export identity_
          

"""
Compute the inverse 2-dimensional discrete Fourier Transform.

  Args:
    in_: A `Tensor` of type `complex64`. A complex64 matrix.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`. The inverse 2D Fourier Transform of `in`.
  """
ifft2d(in_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.ifft2d(;Dict(:in_=>in_, :name=>name)...))
export ifft2d
          

"""
Returns the imaginary part of a complex number.

  Given a tensor `in` of complex numbers, this operation returns a tensor of type
  `float` that is the imaginary part of each element in `in`. All elements in `in`
  must be complex numbers of the form \\(a + bj\\), where *a* is the real part
  and *b* is the imaginary part returned by this operation.

  For example:

  ```
  # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(in) ==> [4.75, 5.75]
  ```

  Args:
    in_: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
imag_(in_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.imag(;Dict(:in_=>in_, :name=>name)...))
export imag_
          

"""
Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_images` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. For float
  input, the values are normalized one image at a time to fit in the range
  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
  normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_images` is 1, the summary value tag is '*tag*/image'.
  *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

  Args:
    tag: A scalar `Tensor` of type `string`. Used to build the `tag`
      of the summary values.
    tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
      width, channels]` where `channels` is 1, 3, or 4.
    max_images: Max number of batch elements to generate images for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
image_summary(tag::Union{AbstractTensor,Void}, tensor::Union{AbstractTensor,Void}, max_images::Any=3, collections::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.image_summary(;Dict(:tag=>tag, :tensor=>tensor, :max_images=>max_images, :collections=>collections, :name=>name)...))
export image_summary
          

"""
Imports the TensorFlow graph in `graph_def` into the Python `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  [`Tensor`](#Tensor) and [`Operation`](#Operation) objects. See
  [`Graph.as_graph_def()`](#Graph.as_graph_def) for a way to create a
  `GraphDef` proto.

  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into
      the default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def`
      to `Tensor` objects. The values of the named input tensors in the
      imported graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in
      `graph_def` that will be returned as `Operation` objects; and/or
      tensor names in `graph_def` that will be returned as `Tensor` objects.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Defaults to `"import"`.
    op_dict: (Optional.) A dictionary mapping op type names to `OpDef` protos.
      Must contain an `OpDef` proto for each op type named in `graph_def`.
      If omitted, uses the `OpDef` protos registered in the global registry.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
import_graph_def(graph_def::Any, input_map::Union{AbstractTensor,Void}=nothing, return_elements::Union{AbstractTensor,Void}=nothing, name::Union{AbstractString,Void}=nothing, op_dict::Union{Dtype,Void}=nothing) = Tensor(tf.import_graph_def(;Dict(:graph_def=>graph_def, :input_map=>input_map, :return_elements=>return_elements, :name=>name, :op_dict=>op_dict)...))
export import_graph_def
          

"""
Returns an Op that initializes all tables of the default graph.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.
  """
initialize_all_tables(name::AbstractString="init_all_tables") = tf.initialize_all_tables(;Dict(:name=>name)...)
export initialize_all_tables
          

"""
Returns an Op that initializes all variables.

  This is just a shortcut for `initialize_variables(all_variables())`

  Returns:
    An Op that initializes all variables in the graph.
  """
initialize_all_variables() = tf.initialize_all_variables(;Dict()...)
export initialize_all_variables
          

"""
Returns an Op that initializes a list of variables.

  After you launch the graph in a session, you can run the returned Op to
  initialize all the variables in `var_list`. This Op runs all the
  initializers of the variables in `var_list` in parallel.

  Calling `initialize_variables()` is equivalent to passing the list of
  initializers to `Group()`.

  If `var_list` is empty, however, the function still returns an Op that can
  be run. That Op just has no effect.

  Args:
    var_list: List of `Variable` objects to initialize.
    name: Optional name for the returned operation.

  Returns:
    An Op that run the initializers of all the specified variables.
  """
initialize_variables(var_list::Any, name::AbstractString="init") = tf.initialize_variables(;Dict(:var_list=>var_list, :name=>name)...)
export initialize_variables
          

"""
Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
inv_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.inv(;Dict(:x=>x, :name=>name)...))
export inv_
          

"""
Computes the inverse permutation of a tensor.

  This operation computes the inverse of an index permutation. It takes a 1-D
  integer tensor `x`, which represents the indices of a zero-based array, and
  swaps each value with its index position. In other words, for an output tensor
  `y` and an input tensor `x`, this operation computes the following:

  `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

  The values must include 0. There can be no duplicate values or negative values.

  For example:

  ```prettyprint
  # tensor `x` is [3, 4, 0, 2, 1]
  invert_permutation(x) ==> [2, 4, 3, 0, 1]
  ```

  Args:
    x: A `Tensor` of type `int32`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. 1-D.
  """
invert_permutation(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.invert_permutation(;Dict(:x=>x, :name=>name)...))
export invert_permutation
          

"""
Returns which elements of x are finite.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
is_finite(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.is_finite(;Dict(:x=>x, :name=>name)...))
export is_finite
          

"""
Returns which elements of x are Inf.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
is_inf(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.is_inf(;Dict(:x=>x, :name=>name)...))
export is_inf
          

"""
Returns which elements of x are NaN.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
is_nan(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.is_nan(;Dict(:x=>x, :name=>name)...))
export is_nan
          

"""
Returns the truth value of (x < y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
less_(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.less(;Dict(:x=>x, :y=>y, :name=>name)...))
export less_
          

"""
Returns the truth value of (x <= y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
less_equal(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.less_equal(;Dict(:x=>x, :y=>y, :name=>name)...))
export less_equal
          

"""
Computes `ln(|gamma(x)|)` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`, `int64`,
      or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
      the return type is `quint8`.
  """
lgamma_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.lgamma(;Dict(:x=>x, :name=>name)...))
export lgamma_
          

"""
Generates values in an interval.

  A sequence of `num` evenly-spaced values are generated beginning at `start`.
  If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
  so that the last one is exactly `stop`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      First entry in the range.
    stop: A `Tensor`. Must have the same type as `start`.
      Last entry in the range.
    num: A `Tensor` of type `int32`. Number of values to generate.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`. 1-D. The generated values.
  """
lin_space(start_::Union{AbstractTensor,Void}, stop::Union{AbstractTensor,Void}, num_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.lin_space(;Dict(:start=>start_, :stop=>stop, :num=>num_, :name=>name)...))
export lin_space
          

"""
Generates values in an interval.

  A sequence of `num` evenly-spaced values are generated beginning at `start`.
  If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
  so that the last one is exactly `stop`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      First entry in the range.
    stop: A `Tensor`. Must have the same type as `start`.
      Last entry in the range.
    num: A `Tensor` of type `int32`. Number of values to generate.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`. 1-D. The generated values.
  """
lin_space(start_::Union{AbstractTensor,Void}, stop::Union{AbstractTensor,Void}, num_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.lin_space(;Dict(:start=>start_, :stop=>stop, :num=>num_, :name=>name)...))
export lin_space
          

"""
Computes the difference between two lists of numbers or strings.

  Given a list `x` and a list `y`, this operation returns a list `out` that
  represents all values that are in `x` but not in `y`. The returned list `out`
  is sorted in the same order that the numbers appear in `x` (duplicates are
  preserved). This operation also returns a list `idx` that represents the
  position of each `out` element in `x`. In other words:

  `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

  For example, given this input:

  ```prettyprint
  x = [1, 2, 3, 4, 5, 6]
  y = [1, 3, 5]
  ```

  This operation would return:

  ```prettyprint
  out ==> [2, 4, 6]
  idx ==> [1, 3, 5]
  ```

  Args:
    x: A `Tensor`. 1-D. Values to keep.
    y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, idx).
    out: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
    idx: A `Tensor` of type `int32`. 1-D. Positions of `x` values preserved in `out`.
  """
list_diff(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.list_diff(;Dict(:x=>x, :y=>y, :name=>name)...))
export list_diff
          

"""
Computes the difference between two lists of numbers or strings.

  Given a list `x` and a list `y`, this operation returns a list `out` that
  represents all values that are in `x` but not in `y`. The returned list `out`
  is sorted in the same order that the numbers appear in `x` (duplicates are
  preserved). This operation also returns a list `idx` that represents the
  position of each `out` element in `x`. In other words:

  `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

  For example, given this input:

  ```prettyprint
  x = [1, 2, 3, 4, 5, 6]
  y = [1, 3, 5]
  ```

  This operation would return:

  ```prettyprint
  out ==> [2, 4, 6]
  idx ==> [1, 3, 5]
  ```

  Args:
    x: A `Tensor`. 1-D. Values to keep.
    y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, idx).
    out: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
    idx: A `Tensor` of type `int32`. 1-D. Positions of `x` values preserved in `out`.
  """
list_diff(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.list_diff(;Dict(:x=>x, :y=>y, :name=>name)...))
export list_diff
          

"""
Loads a TensorFlow plugin, containing custom ops and kernels.

  Pass "library_filename" to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here.
  Expects the symbols "RegisterOps", "RegisterKernels", and "GetOpList", to be
  defined in the library.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    A python module containing the Python wrappers for Ops defined in
    the plugin.

  Raises:
    RuntimeError: when unable to load the library or get the python wrappers.
  """
load_op_library(library_filename::Any) = tf.load_op_library(;Dict(:library_filename=>library_filename)...)
export load_op_library
          

"""
Computes natural logarithm of x element-wise.

  I.e., \\(y = \log_e x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
log_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.log(;Dict(:x=>x, :name=>name)...))
export log_
          

"""
Returns the truth value of x AND y element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
logical_and(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.logical_and(;Dict(:x=>x, :y=>y, :name=>name)...))
export logical_and
          

"""
Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
logical_not(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.logical_not(;Dict(:x=>x, :name=>name)...))
export logical_not
          

"""
Returns the truth value of x OR y element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
logical_or(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.logical_or(;Dict(:x=>x, :y=>y, :name=>name)...))
export logical_or
          

"""
x ^ y = (x | y) & ~(x & y)."""
logical_xor(x::Any, y::Any, name::AbstractString="LogicalXor") = tf.logical_xor(;Dict(:x=>x, :y=>y, :name=>name)...)
export logical_xor
          

"""
Given an arbitrary function, wrap it so that it does variable sharing.

  This wraps `func_` in a Template and partially evaluates it. Templates are
  functions that create variables the first time they are called and reuse them
  thereafter. In order for `func_` to be compatible with a `Template` it must
  have the following properties:

  * The function should create all trainable variables and any variables that
     should be reused by calling `tf.get_variable`. If a trainable variable is
     created using `tf.Variable`, then a ValueError will be thrown. Variables
     that are intended to be locals can be created by specifying
     `tf.Variable(..., trainable=false)`.
  * The function may use variable scopes and other templates internally to
      create and reuse variables, but it shouldn't use `tf.get_variables` to
      capture variables that are defined outside of the scope of the function.
  * Internal scopes and variable names should not depend on any arguments that
      are not supplied to `make_template`. In general you will get a ValueError
      telling you that you are trying to reuse a variable that doesn't exist
      if you make a mistake.

  In the following example, both `z` and `w` will be scaled by the same `y`. It
  is important to note that if we didn't assign `scalar_name` and used a
  different name for z and w that a `ValueError` would be thrown because it
  couldn't reuse the variable.

  ```python
  def my_op(x, scalar_name):
    var1 = tf.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.constant_initializer(1))
    return x * var1

  scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')

  z = scale_by_y(input1)
  w = scale_by_y(input2)
  ```

  As a safe-guard, the returned function will raise a `ValueError` after the
  first call if trainable variables are created by calling `tf.Variable`.

  If all of these are true, then 2 properties are enforced by the template:

  1. Calling the same template multiple times will share all non-local
      variables.
  2. Two different templates are guaranteed to be unique, unless you reenter the
      same variable scope as the initial definition of a template and redefine
      it. An examples of this exception:

  ```python
  def my_op(x, scalar_name):
    var1 = tf.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.constant_initializer(1))
    return x * var1

  with tf.variable_scope('scope') as vs:
    scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')
    z = scale_by_y(input1)
    w = scale_by_y(input2)

  # Creates a template that reuses the variables above.
  with tf.variable_scope(vs, reuse=True):
    scale_by_y2 = tf.make_template('scale_by_y', my_op, scalar_name='y')
    z2 = scale_by_y2(input1)
    w2 = scale_by_y2(input2)
  ```

  Note: The full variable scope is captured at the time of the first call.

  Note: `name_` and `func_` have a following underscore to reduce the likelihood
  of collisions with kwargs.

  Args:
    name_: A name for the scope created by this template. If necessary, the name
      will be made unique by appending `_N` to the name.
    func_: The function to wrap.
    **kwargs: Keyword arguments to apply to `func_`.

  Returns:
    A function that will enter a `variable_scope` before calling `func_`. The
    first time it is called, it will create a non-reusing scope so that the
    variables will be unique.  On each subsequent call, it will reuse those
    variables.

  Raises:
    ValueError: if the name is None.
  """
make_template(name_::Any, func_::Any) = tf.make_template(;Dict(:name_=>name_, :func_=>func_)...)
export make_template
          

"""
Returns the set of files matching a pattern.

  Note that this routine only supports wildcard characters in the
  basename portion of the pattern, not in the directory portion.

  Args:
    pattern: A `Tensor` of type `string`. A (scalar) shell wildcard pattern.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. A vector of matching filenames.
  """
matching_files(pattern::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.matching_files(;Dict(:pattern=>pattern, :name=>name)...))
export matching_files
          

"""
Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

  The inputs must be two-dimensional matrices, with matching inner dimensions,
  possibly after transposition.

  Both matrices must be of the same type. The supported types are:
  `float`, `double`, `int32`, `complex64`.

  Either matrix can be transposed on the fly by setting the corresponding flag
  to `True`. This is `False` by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.

  For example:

  ```python
  # 2-D tensor `a`
  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                        [4. 5. 6.]]
  # 2-D tensor `b`
  b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                           [9. 10.]
                                                           [11. 12.]]
  c = tf.matmul(a, b) => [[58 64]
                          [139 154]]
  ```

  Args:
    a: `Tensor` of type `float`, `double`, `int32` or `complex64`.
    b: `Tensor` with same type as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a`.
  """
matmul(a::Union{AbstractTensor,Void}, b::Union{AbstractTensor,Void}, transpose_a::Bool=false, transpose_b::Bool=false, a_is_sparse::Bool=false, b_is_sparse::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.matmul(;Dict(:a=>a, :b=>b, :transpose_a=>transpose_a, :transpose_b=>transpose_b, :a_is_sparse=>a_is_sparse, :b_is_sparse=>b_is_sparse, :name=>name)...))
export matmul
          

"""
Calculates the determinant of a square matrix.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A tensor of shape `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    A scalar, equal to the determinant of the input.
  """
matrix_determinant(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.matrix_determinant(;Dict(:input=>input, :name=>name)...))
export matrix_determinant
          

"""
Calculates the inverse of a square invertible matrix.

  The op uses the Cholesky decomposition if the matrix is symmetric positive
  definite and LU decomposition with partial pivoting otherwise.

  If the matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Shape is `[M, M]` containing the matrix inverse of the input.
  """
matrix_inverse(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.matrix_inverse(;Dict(:input=>input, :name=>name)...))
export matrix_inverse
          

"""
Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
maximum_(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.maximum(;Dict(:x=>x, :y=>y, :name=>name)...))
export maximum_
          

"""
Merges all summaries collected in the default graph.

  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.

  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type`string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
merge_all_summaries(key::Any="summaries") = Tensor(tf.merge_all_summaries(;Dict(:key=>key)...))
export merge_all_summaries
          

"""
Merges summaries.

  This op creates a
  [`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.

  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.

  Args:
    inputs: A list of `string` `Tensor` objects containing serialized `Summary`
      protocol buffers.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer resulting from the merging.
  """
merge_summary(inputs::Union{AbstractTensor,Void}, collections::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.merge_summary(;Dict(:inputs=>inputs, :collections=>collections, :name=>name)...))
export merge_summary
          

"""
Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
minimum_(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.minimum(;Dict(:x=>x, :y=>y, :name=>name)...))
export minimum_
          

"""
Returns element-wise remainder of division.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
mod_(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.mod(;Dict(:x=>x, :y=>y, :name=>name)...))
export mod_
          

"""
Returns all variables that maintain their moving averages.

  If an `ExponentialMovingAverage` object is created and the `apply()`
  method is called on a list of variables, these variables will
  be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
  This convenience function returns the contents of that collection.

  Returns:
    A list of Variable objects.
  """
moving_average_variables() = tf.moving_average_variables(;Dict()...)
export moving_average_variables
          

"""
Returns x * y element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
mul(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.mul(;Dict(:x=>x, :y=>y, :name=>name)...))
export mul
          

"""
Wrapper for `Graph.name_scope()` using the default graph.

  See
  [`Graph.name_scope()`](../../api_docs/python/framework.md#Graph.name_scope)
  for more details.

  Args:
    name: A name for the scope.

  Returns:
    A context manager that installs `name` as a new name scope in the
    default graph.
  """
name_scope(name::Union{AbstractString,Void}) = AbstractString(tf.name_scope(;Dict(:name=>name)...))
export name_scope
          

"""
Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
neg(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.neg(;Dict(:x=>x, :name=>name)...))
export neg
          

"""
Does nothing. Only useful as a placeholder for control edges.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
no_op(name::Union{AbstractString,Void}=nothing) = tf.no_op(;Dict(:name=>name)...)
export no_op
          

"""
Returns the truth value of (x != y) element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
not_equal(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.not_equal(;Dict(:x=>x, :y=>y, :name=>name)...))
export not_equal
          

"""
Creates a tensor with all elements set to 1.

  This operation returns a tensor of type `dtype` with shape `shape` and all
  elements set to 1.

  For example:

  ```python
  tf.ones([2, 3], int32) ==> [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to 1.
  """
ones_(shape::Union{AbstractTensor,DimsType,Void}, dtype::Dtype=DT_FLOAT32, name::Union{AbstractString,Void}=nothing) = Tensor(tf.ones(;Dict(:shape=>shape, :dtype=>dtype, :name=>name)...))
export ones_
          

"""
Creates a tensor with all elements set to 1.

  Given a single tensor (`tensor`), this operation returns a tensor of the same
  type and shape as `tensor` with all elements set to 1. Optionally, you can
  specify a new type (`dtype`) for the returned tensor.

  For example:

  ```python
  # 'tensor' is [[1, 2, 3], [4, 5, 6]]
  tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
    `int8`, `int16`, `int32`, `int64`, `uint8`, or `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to 1.
  """
ones_like(tensor::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.ones_like(;Dict(:tensor=>tensor, :dtype=>dtype, :name=>name)...))
export ones_like
          

"""
Returns a context manager for use when defining a Python op.

  This context manager validates that the given `values` are from the
  same graph, ensures that that graph is the default graph, and pushes a
  name scope.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.op_scope([a, b, c], name, "MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  Args:
    values: The list of `Tensor` arguments that are passed to the op function.
    name: The name argument that is passed to the op function.
    default_name: The default name to use if the `name` argument is `None`.

  Returns:
    A context manager for use in defining Python ops. Yields the name scope.

  Raises:
    ValueError: if neither `name` nor `default_name` is provided.
  """
op_scope() = AbstractString(tf.op_scope(;Dict()...))
export op_scope
          

"""
Packs a list of rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs tensors in `values` into a tensor with rank one higher than each tensor
  in `values` and shape `[len(values)] + values[0].shape`. The output satisfies
  `output[i, ...] = values[i][...]`.

  This is the opposite of unpack.  The numpy equivalent is

      tf.pack([x, y, z]) = np.asarray([x, y, z])

  Args:
    values: A list of `Tensor` objects with the same shape and type.
    name: A name for this operation (optional).

  Returns:
    output: A packed `Tensor` with the same type as `values`.
  """
pack(values_::Union{AbstractTensor,Void}, name::AbstractString="pack") = Tensor(tf.pack(;Dict(:values=>values_, :name=>name)...))
export pack
          

"""
Pads a tensor with zeros.

  This operation pads a `input` with zeros according to the `paddings` you
  specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
  rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many zeros to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
  in that dimension.

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```prettyprint
  # 't' is [[1, 1], [2, 2]]
  # 'paddings' is [[1, 1], [2, 2]]
  # rank of 't' is 2
  pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                        [0, 0, 1, 1, 0, 0]
                        [0, 0, 2, 2, 0, 0]
                        [0, 0, 0, 0, 0, 0]]
  ```

  Args:
    input: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
pad(input::Union{AbstractTensor,Void}, paddings::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.pad(;Dict(:input=>input, :paddings=>paddings, :name=>name)...))
export pad
          

"""
Parses `Example` protos.

  Parses a number of serialized [`Example`]
  (https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/example/example.proto)
  protos given in `serialized`.

  `names` may contain descriptive names for the corresponding serialized protos.
  These may be useful for debugging purposes, but they have no effect on the
  output. If not `None`, `names` must be the same length as `serialized`.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  and `SparseTensor` objects respectively, depending on whether the keys appear
  in `sparse_keys` or `dense_keys`.

  The key `dense_keys[j]` is mapped to a `Tensor` of type `dense_types[j]` and
  of shape `(serialized.size(),) + dense_shapes[j]`.

  `dense_defaults` provides defaults for values referenced using `dense_keys`.
  If a key is not present in this dictionary, the corresponding dense `Feature`
  is required in all elements of `serialized`.

  `dense_shapes[j]` provides the shape of each `Feature` entry referenced by
  `dense_keys[j]`. The number of elements in the `Feature` corresponding to
  `dense_key[j]` must always have `np.prod(dense_shapes[j])` entries. The
  returned `Tensor` for `dense_key[j]` has shape `[N] + dense_shape[j]`, where
  `N` is the number of `Example`s in `serialized`.

  The key `sparse_keys[j]` is mapped to a `SparseTensor` of type
  `sparse_types[j]`. The `SparseTensor` represents a ragged matrix.
  Its indices are `[batch, index]` where `batch` is the batch entry the value
  is from, and `index` is the value's index in the list of values associated
  with that feature and example.

  Examples:

  For example, if one expects a `tf.float32` sparse feature `ft` and three
  serialized `Example`s are provided:

  ```
  serialized = [
    features
      { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
    features
      { feature []},
    features
      { feature { key: "ft" value { float_list { value: [3.0] } } }
  ]
  ```

  then the output will look like:

  ```
  {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                      values=[1.0, 2.0, 3.0],
                      shape=(3, 2)) }
  ```

  Given two `Example` input protos in `serialized`:

  ```
  [
    features {
      feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
      feature { key: "gps" value { float_list { value: [] } } }
    },
    features {
      feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
      feature { key: "dank" value { int64_list { value: [ 42 ] } } }
      feature { key: "gps" value { } }
    }
  ]
  ```

  And arguments

  ```
    names: ["input0", "input1"],
    sparse_keys: ["kw", "dank", "gps"]
    sparse_types: [DT_STRING, DT_INT64, DT_FLOAT]
  ```

  Then the output is a dictionary:

  ```python
  {
    "kw": SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=["knit", "big", "emmy"]
        shape=[2, 2]),
    "dank": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        shape=[2, 1]),
    "gps": SparseTensor(
        indices=[],
        values=[],
        shape=[2, 0]),
  }
  ```

  For dense results in two serialized `Example`s:

  ```
  [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
     },
     features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  names: ["input0", "input1"],
  dense_keys: np.array(["age", "gender"]),
  dense_types: [tf.int64, tf.string],
  dense_defaults: {
    "age": -1  # "age" defaults to -1 if missing
               # "gender" has no specified default so it's required
  }
  dense_shapes: [(1,), (1,)],  # age, gender, label, weight
  ```

  And the expected output is:

  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
  }
  ```

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos.
    sparse_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `SparseTensor` objects.
    sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `Tensor`s
    dense_types: A list of DTypes of the same length as `dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    dense_shapes: A list of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys` whose shapes are
      anything other than `[]` or `[1]`.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping keys to `Tensor`s and `SparseTensor`s.

  Raises:
    ValueError: If sparse and dense key sets intersect, or input lengths do not
      match up.
  """
parse_example(serialized::Union{AbstractTensor,Void}, names_::Any=nothing, sparse_keys::Any=nothing, sparse_types::Any=nothing, dense_keys::Any=nothing, dense_types::Any=nothing, dense_defaults::Union{AbstractTensor,Void}=nothing, dense_shapes::Any=nothing, name::AbstractString="ParseExample") = Tensor(tf.parse_example(;Dict(:serialized=>serialized, :names=>names_, :sparse_keys=>sparse_keys, :sparse_types=>sparse_types, :dense_keys=>dense_keys, :dense_types=>dense_types, :dense_defaults=>dense_defaults, :dense_shapes=>dense_shapes, :name=>name)...))
export parse_example
          

"""
Parses a single `Example` proto.

  Similar to `parse_example`, except:

  For dense tensors, the returned `Tensor` is identical to the output of
  `parse_example`, except there is no batch dimension, the output shape is the
  same as the shape given in `dense_shape`.

  For `SparseTensor`s, the first (batch) column of the indices matrix is removed
  (the indices matrix is a column vector), the values vector is unchanged, and
  the first (`batch_size`) entry of the shape vector is removed (it is now a
  single element vector).

  See also `parse_example`.

  Args:
    serialized: A scalar string Tensor, a single serialized Example.
      See `parse_example` documentation for more details.
    names: (Optional) A scalar string Tensor, the associated name.
      See `parse_example` documentation for more details.
    sparse_keys: See `parse_example` documentation for more details.
    sparse_types: See `parse_example` documentation for more details.
    dense_keys: See `parse_example` documentation for more details.
    dense_types: See `parse_example` documentation for more details.
    dense_defaults: See `parse_example` documentation for more details.
    dense_shapes: See `parse_example` documentation for more details.
    name: A name for this operation (optional).

  Returns:
    A dictionary mapping keys to Tensors and SparseTensors.

  Raises:
    ValueError: if "scalar" or "names" have known shapes, and are not scalars.
  """
parse_single_example(serialized::Union{AbstractTensor,Void}, names_::Union{AbstractTensor,Void}=nothing, sparse_keys::Any=nothing, sparse_types::Any=nothing, dense_keys::Any=nothing, dense_types::Any=nothing, dense_defaults::Any=nothing, dense_shapes::Any=nothing, name::AbstractString="ParseSingleExample") = Tensor(tf.parse_single_example(;Dict(:serialized=>serialized, :names=>names_, :sparse_keys=>sparse_keys, :sparse_types=>sparse_types, :dense_keys=>dense_keys, :dense_types=>dense_types, :dense_defaults=>dense_defaults, :dense_shapes=>dense_shapes, :name=>name)...))
export parse_single_example
          

"""
Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`]
  (https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialize sequence example into a tuple of dictionaries
  mapping keys to `Tensor` and `SparseTensor` objects respectively.
  The first dictionary contains mappings for keys appearing in
  `context_sparse_keys` or `context_dense_keys`, and the second dictionary
  contains mappings for keys appearing in `feature_list_dense_keys`.

  The `context` keys are associated with a `SequenceExample` as a whole,
  independent of time / frame.  In contrast, the `feature_list` keys provide
  a way to access variable-length data within the `FeatureList` section of the
  `SequenceExample` proto.  While the shapes of `context` values are fixed
  with respect to frame, the frame dimension (the first dimension)
  of `feature_list` values may vary from `SequenceExample` to `SequenceExample`
  and even between `feature_list` keys within the same `SequenceExample`.

  The key `context_dense_keys[j]` is mapped to a `Tensor` of type
  `context_dense_types[j]` and of shape `context_dense_shapes[j]`.

  `context_dense_defaults` provides defaults for values referenced using
  `context_dense_keys`.  If a key is not present in this dictionary, the
  corresponding context_dense `Feature` is required in `serialized`.

  `context_dense_shapes[j]` provides the shape of each context `Feature` entry
  referenced by `context_dense_keys[j]`. The number of elements in the
  `Feature` corresponding to `context_dense_key[j]` must always have
  `np.prod(context_dense_shapes[j])` entries. The returned `Tensor` for
  `context_dense_key[j]` has shape `context_dense_shape[j]`.

  The key `context_sparse_keys[j]` is mapped to a `SparseTensor` of type
  `context_sparse_types[j]`. This `SparseTensor` represents a ragged vector.
  Its indices are `[index]`, where `index` is the value's index in the list of
  values associated with that feature and example.

  The key `feature_list_dense_keys[j]` is mapped to a `Tensor` of type
  `feature_list_dense_types[j]` and of shape
  `(T,) + feature_list_dense_shapes[j]`, where `T` is the length of the
  associated `FeatureList` in the `SequenceExample`.

  Note: every key declared in `feature_list_dense_keys` **must** be
  provided in the `SequenceExample`'s `FeatureLists`, even if just empty.
  Exceptions are allowed by adding the given key to the map
  `feature_list_dense_defaults` with value None.  Any key with value None
  map will be  treated as empty (zero length) if not found in the
  `FeatureList` map.

  The key `feature_list_sparse_keys[j]` is mapped to a `SparseTensor` of type
  `feature_list_sparse_types[j]`. This `SparseTensor` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the FeatureList
  entry `index` is the value's index in the list of values associated with that
  time.

  `debug_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `debug_name` must be a scalar.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_sparse_keys: A list of string keys in the `SequenceExample`'s
      features.  The results for these keys will be returned as
      `SparseTensor` objects.
    context_sparse_types: A list of `DTypes`, the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    context_dense_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `Tensor`s
    context_dense_types: A list of DTypes, same length as `context_dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    context_dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the context_dense_keys of the feature.
    context_dense_shapes: A list of tuples, same length as `context_dense_keys`.
      The shape of the data for each context_dense feature referenced by
      `context_dense_keys`.  Required for any input tensors identified by
      `context_dense_keys` whose shapes are anything other than `[]` or `[1]`.
    feature_list_sparse_keys: A list of string keys in the `SequenceExample`'s
      feature_lists.  The results for these keys will be returned as
      `SparseTensor` objects.
    feature_list_sparse_types: A list of `DTypes`, same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    feature_list_dense_keys: A list of string keys in the `SequenceExample`'s
      features_lists. The results for these keys will be returned as `Tensor`s.
    feature_list_dense_types: A list of `DTypes`, same length as
      `feature_list_dense_keys`.  Only `tf.float32` (`FloatList`),
      `tf.int64` (`Int64List`), and `tf.string` (`BytesList`) are supported.
    feature_list_dense_shapes: A list of tuples, same length as
      `feature_list_dense_keys`.  The shape of the data for each
      `FeatureList` feature referenced by `feature_list_dense_keys`.
    feature_list_dense_defaults: A dict mapping key strings to values.
      The only currently allowed value is `None`.  Any key appearing
      in this dict with value `None` is allowed  to be missing from the
      `SequenceExample`.  If missing, the key is treated as zero-length.
    debug_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    ValueError: If context_sparse and context_dense key sets intersect,
      if input lengths do not match up, or if a value in
      feature_list_dense_defaults is not None.
    TypeError: if feature_list_dense_defaults is not either None or a dict.
  """
parse_single_sequence_example(serialized::Union{AbstractTensor,Void}, context_sparse_keys::Any=nothing, context_sparse_types::Any=nothing, context_dense_keys::Any=nothing, context_dense_types::Any=nothing, context_dense_defaults::Union{AbstractTensor,Void}=nothing, context_dense_shapes::Any=nothing, feature_list_sparse_keys::Any=nothing, feature_list_sparse_types::Any=nothing, feature_list_dense_keys::Any=nothing, feature_list_dense_types::Any=nothing, feature_list_dense_shapes::Any=nothing, feature_list_dense_defaults::Any=nothing, debug_name::Union{AbstractTensor,Void}=nothing, name::AbstractString="ParseSingleSequenceExample") = Tensor(tf.parse_single_sequence_example(;Dict(:serialized=>serialized, :context_sparse_keys=>context_sparse_keys, :context_sparse_types=>context_sparse_types, :context_dense_keys=>context_dense_keys, :context_dense_types=>context_dense_types, :context_dense_defaults=>context_dense_defaults, :context_dense_shapes=>context_dense_shapes, :feature_list_sparse_keys=>feature_list_sparse_keys, :feature_list_sparse_types=>feature_list_sparse_types, :feature_list_dense_keys=>feature_list_dense_keys, :feature_list_dense_types=>feature_list_dense_types, :feature_list_dense_shapes=>feature_list_dense_shapes, :feature_list_dense_defaults=>feature_list_dense_defaults, :debug_name=>debug_name, :name=>name)...))
export parse_single_sequence_example
          

"""
Inserts a placeholder for a tensor that will be always fed.

  **Important**: This tensor will produce an error if evaluated. Its value must
  be fed using the `feed_dict` optional argument to `Session.run()`,
  `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.placeholder(tf.float32, shape=(1024, 1024))
  y = tf.matmul(x, x)

  with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
  ```

  Args:
    dtype: The type of elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a tensor of any shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that may be used as a handle for feeding a value, but not
    evaluated directly.
  """
placeholder(dtype::Union{Dtype,Void}, shape::Union{AbstractTensor,DimsType,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Placeholder(tf.placeholder(;Dict(:dtype=>dtype, :shape=>shape, :name=>name)...))
export placeholder
          

"""
Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```
  # tensor 'x' is [[2, 2]], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor` of type `float`, `double`, `int32`, `complex64`, or `int64`.
    y: A `Tensor` of type `float`, `double`, `int32`, `complex64`, or `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.
  """
pow(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.pow(;Dict(:x=>x, :y=>y, :name=>name)...))
export pow
          

"""
Wraps a python function and uses it as a tensorflow op.

  Given a python function `func`, which takes numpy arrays as its
  inputs and returns numpy arrays as its outputs. E.g.,

    def my_func(x):
      return np.sinh(x)
    inp = tf.placeholder(..., tf.float32)
    y = py_func(my_func, [inp], [tf.float32])

  The above snippet constructs a tf graph which invokes a numpy
  sinh(x) as an op in the graph.

  Args:
    func: A python function.
    inp: A list of `Tensor`.
    Tout: A list of tensorflow data types indicating what `func`
          returns.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` which `func` computes.
  """
py_func(func::Any, inp::Union{AbstractTensor,Void}, Tout::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.py_func(;Dict(:func=>func, :inp=>inp, :Tout=>Tout, :name=>name)...))
export py_func
          

"""
Outputs random values from a normal distribution.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  """
random_normal(shape::Union{AbstractTensor,DimsType,Void}, mean_::AbstractTensor=0.0, stddev::AbstractTensor=1.0, dtype::Dtype=DT_FLOAT32, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.random_normal(;Dict(:shape=>shape, :mean=>mean_, :stddev=>stddev, :dtype=>dtype, :seed=>seed, :name=>name)...))
export random_normal
          

"""
Returns an initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a normal distribution.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
random_normal_initializer(mean_::AbstractTensor=0.0, stddev::AbstractTensor=1.0, seed::Union{Int64,Void}=nothing, dtype::Dtype=DT_FLOAT32) = Tensor(tf.random_normal_initializer(;Dict(:mean=>mean_, :stddev=>stddev, :seed=>seed, :dtype=>dtype)...))
export random_normal_initializer
          

"""
Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

  ```python
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```

  Args:
    value: A Tensor to be shuffled.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of same shape and type as `value`, shuffled along its first
    dimension.
  """
random_shuffle(value::Union{AbstractTensor,Void}, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.random_shuffle(;Dict(:value=>value, :seed=>seed, :name=>name)...))
export random_shuffle
          

"""
Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.

  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.

  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
      range of random values to generate.  Defaults to 0.
    maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
      the range of random values to generate.  Defaults to 1 if `dtype` is
      floating point.
    dtype: The type of the output: `float32`, `float64`, `int32`, or `int64`.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.

  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  """
random_uniform(shape::Union{AbstractTensor,DimsType,Void}, minval::AbstractTensor=0, maxval::Union{AbstractTensor,Void}=nothing, dtype::Dtype=DT_FLOAT32, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.random_uniform(;Dict(:shape=>shape, :minval=>minval, :maxval=>maxval, :dtype=>dtype, :seed=>seed, :name=>name)...))
export random_uniform
          

"""
Returns an initializer that generates tensors with a uniform distribution.

  Args:
    minval: a python scalar or a scalar tensor. lower bound of the range
      of random values to generate.
    maxval: a python scalar or a scalar tensor. upper bound of the range
      of random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a uniform distribution.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
random_uniform_initializer(minval::AbstractTensor=0.0, maxval::AbstractTensor=1.0, seed::Union{Int64,Void}=nothing, dtype::Dtype=DT_FLOAT32) = Tensor(tf.random_uniform_initializer(;Dict(:minval=>minval, :maxval=>maxval, :seed=>seed, :dtype=>dtype)...))
export random_uniform_initializer
          

"""
Creates a sequence of integers.

  Creates a sequence of integers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.

  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.

  For example:

  ```
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

  # 'limit' is 5
  tf.range(limit) ==> [0, 1, 2, 3, 4]
  ```

  Args:
    start: A 0-D (scalar) of type `int32`. First entry in sequence.
      Defaults to 0.
    limit: A 0-D (scalar) of type `int32`. Upper limit of sequence,
      exclusive.
    delta: A 0-D `Tensor` (scalar) of type `int32`. Optional. Default is 1.
      Number that increments `start`.
    name: A name for the operation (optional).

  Returns:
    An 1-D `int32` `Tensor`.
  """
range_(start_::Union{Dtype,Void}, limit::Union{Dtype,Void}=nothing, delta::AbstractTensor=1, name::AbstractString="range") = Tensor(tf.range(;Dict(:start=>start_, :limit=>limit, :delta=>delta, :name=>name)...))
export range_
          

"""
Returns the rank of a tensor.

  This operation returns an integer representing the rank of `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  # shape of tensor 't' is [2, 2, 3]
  rank(t) ==> 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
  of a tensor is the number of indices required to uniquely select each element
  of the tensor. Rank is also known as "order", "degree", or "ndims."

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
rank_(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.rank(;Dict(:input=>input, :name=>name)...))
export rank_
          

"""
Reads and outputs the entire contents of the input filename.

  Args:
    filename: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
read_file(filename::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.read_file(;Dict(:filename=>filename, :name=>name)...))
export read_file
          

"""
Returns the real part of a complex number.

  Given a tensor `in` of complex numbers, this operation returns a tensor of type
  `float` that is the real part of each element in `in`. All elements in `in`
  must be complex numbers of the form \\(a + bj\\), where *a* is the real part
  returned by this operation and *b* is the imaginary part.

  For example:

  ```
  # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(in) ==> [-2.25, 3.25]
  ```

  Args:
    in_: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
real_(in_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.real(;Dict(:in_=>in_, :name=>name)...))
export real_
          

"""
Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[True,  True]
  #         [False, False]]
  tf.reduce_all(x) ==> False
  tf.reduce_all(x, 0) ==> [False, False]
  tf.reduce_all(x, 1) ==> [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_all(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_all(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_all
          

"""
Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[True,  True]
  #         [False, False]]
  tf.reduce_any(x) ==> True
  tf.reduce_any(x, 0) ==> [True, True]
  tf.reduce_any(x, 1) ==> [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_any(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_any(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_any
          

"""
Computes the maximum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_max(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_max(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_max
          

"""
Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[1., 1.]
  #         [2., 2.]]
  tf.reduce_mean(x) ==> 1.5
  tf.reduce_mean(x, 0) ==> [1.5, 1.5]
  tf.reduce_mean(x, 1) ==> [1.,  2.]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_mean(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_mean(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_mean
          

"""
Computes the minimum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_min(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_min(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_min
          

"""
Computes the product of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_prod(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_prod(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_prod
          

"""
Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `reduction_indices`.
  Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
  entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
  are retained with length 1.

  If `reduction_indices` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  # 'x' is [[1, 1, 1]
  #         [1, 1, 1]]
  tf.reduce_sum(x) ==> 6
  tf.reduce_sum(x, 0) ==> [2, 2, 2]
  tf.reduce_sum(x, 1) ==> [3, 3]
  tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
  tf.reduce_sum(x, [0, 1]) ==> 6
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. If `None` (the defaut),
      reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  """
reduce_sum(input_tensor::Union{AbstractTensor,Void}, reduction_indices::Any=nothing, keep_dims::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reduce_sum(;Dict(:input_tensor=>input_tensor, :reduction_indices=>reduction_indices, :keep_dims=>keep_dims, :name=>name)...))
export reduce_sum
          

"""
Registers a function for converting objects of `base_type` to `Tensor`.

  The conversion function must have the following signature:

      def conversion_func(value, dtype=None, name=None, as_ref=False):
        # ...

  It must return a `Tensor` with the given `dtype` if specified. If the
  conversion function creates a new `Tensor`, it should use the given
  `name` if specified. All exceptions will be propagated to the caller.

  If `as_ref` is true, the function must return a `Tensor` reference,
  such as a `Variable`.

  NOTE: The conversion functions will execute in order of priority,
  followed by order of registration. To ensure that a conversion function
  `F` runs before another conversion function `G`, ensure that `F` is
  registered with a smaller priority than `G`.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values
      run earlier than conversion functions with larger priority values.
      Defaults to 100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.

  """
register_tensor_conversion_function(base_type::Union{Dtype,Void}, conversion_func::Union{AbstractTensor,Void}, priority::Any=100) = tf.register_tensor_conversion_function(;Dict(:base_type=>base_type, :conversion_func=>conversion_func, :priority=>priority)...)
export register_tensor_conversion_function
          

"""
Clears the default graph stack and resets the global default graph.

  NOTE: The default graph is a property of the current thread. This
  function applies only to the current thread.  Calling this function while
  a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
  behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
  after calling this function will result in undefined behavior.
  """
reset_default_graph() = tf.reset_default_graph(;Dict()...)
export reset_default_graph
          

"""
Reshapes a tensor.

  Given `tensor`, this operation returns a tensor that has the same values
  as `tensor` with shape `shape`.

  If one component of `shape` is the special value -1, the size of that dimension
  is computed so that the total size remains constant.  In particular, a `shape`
  of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

  If `shape` is 1-D or higher, then the operation returns a tensor with shape
  `shape` filled with the values of `tensor`. In this case, the number of elements
  implied by `shape` must be the same as the number of elements in `tensor`.

  For example:

  ```prettyprint
  # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # tensor 't' has shape [9]
  reshape(t, [3, 3]) ==> [[1, 2, 3]
                          [4, 5, 6]
                          [7, 8, 9]]

  # tensor 't' is [[[1, 1], [2, 2]]
  #                [[3, 3], [4, 4]]]
  # tensor 't' has shape [2, 2, 2]
  reshape(t, [2, 4]) ==> [[1, 1, 2, 2]
                          [3, 3, 4, 4]]

  # tensor 't' is [[[1, 1, 1],
  #                 [2, 2, 2]],
  #                [[3, 3, 3],
  #                 [4, 4, 4]],
  #                [[5, 5, 5],
  #                 [6, 6, 6]]]
  # tensor 't' has shape [3, 2, 3]
  # pass '[-1]' to flatten 't'
  reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
  # -1 can also be used with higher dimensional shapes
  reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]

  # tensor 't' is [7]
  # shape `[]` reshapes to a scalar
  reshape(t, []) ==> 7
  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
reshape_(tensor::Union{AbstractTensor,Void}, shape::Union{AbstractTensor,DimsType,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reshape(;Dict(:tensor=>tensor, :shape=>shape, :name=>name)...))
export reshape_
          

"""
Reverses specific dimensions of a tensor.

  Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
  of `tensor`, this operation reverses each dimension i of `tensor` where
  `dims[i]` is `True`.

  `tensor` can have up to 8 dimensions. The number of dimensions
  of `tensor` must equal the number of elements in `dims`. In other words:

  `rank(tensor) = size(dims)`

  For example:

  ```prettyprint
  # tensor 't' is [[[[ 0,  1,  2,  3],
  #                  [ 4,  5,  6,  7],
  #                  [ 8,  9, 10, 11]],
  #                 [[12, 13, 14, 15],
  #                  [16, 17, 18, 19],
  #                  [20, 21, 22, 23]]]]
  # tensor 't' shape is [1, 2, 3, 4]

  # 'dims' is [False, False, False, True]
  reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                          [ 7,  6,  5,  4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

  # 'dims' is [False, True, False, False]
  reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]]]]

  # 'dims' is [False, False, True, False]
  reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `bool`, `float32`, `float64`.
      Up to 8-D.
    dims: A `Tensor` of type `bool`. 1-D. The dimensions to reverse.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.
  """
reverse_(tensor::Union{AbstractTensor,Void}, dims::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reverse(;Dict(:tensor=>tensor, :dims=>dims, :name=>name)...))
export reverse_
          

"""
Reverses variable length slices.

  This op first slices `input` along the dimension `batch_dim`, and for each
  slice `i`, reverses the first `seq_lengths[i]` elements along
  the dimension `seq_dim`.

  The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
  and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

  The output slice `i` along dimension `batch_dim` is then given by input
  slice `i`, with the first `seq_lengths[i]` slices along dimension
  `seq_dim` reversed.

  For example:

  ```prettyprint
  # Given this:
  batch_dim = 0
  seq_dim = 1
  input.dims = (4, 8, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
  output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
  output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
  output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

  # while entries past seq_lens are copied through:
  output[0, 7:, :, ...] = input[0, 7:, :, ...]
  output[1, 2:, :, ...] = input[1, 2:, :, ...]
  output[2, 3:, :, ...] = input[2, 3:, :, ...]
  output[3, 2:, :, ...] = input[3, 2:, :, ...]
  ```

  In contrast, if:

  ```prettyprint
  # Given this:
  batch_dim = 2
  seq_dim = 0
  input.dims = (8, ?, 4, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
  output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
  output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
  output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

  # while entries past seq_lens are copied through:
  output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
  output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
  output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
  output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
  ```

  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor` of type `int64`.
      1-D with length `input.dims(0)` and
      `max(seq_lengths) < input.dims(seq_dim)`
    seq_dim: An `int`. The dimension which is partially reversed.
    batch_dim: An optional `int`. Defaults to `0`.
      The dimension along which reversal is performed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The partially reversed input. It has the same shape as `input`.
  """
reverse_sequence(input::Union{AbstractTensor,Void}, seq_lengths::Union{AbstractTensor,Void}, seq_dim::Any, batch_dim::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.reverse_sequence(;Dict(:input=>input, :seq_lengths=>seq_lengths, :seq_dim=>seq_dim, :batch_dim=>batch_dim, :name=>name)...))
export reverse_sequence
          

"""
Rounds the values of a tensor to the nearest integer, element-wise.

  For example:

  ```python
  # 'a' is [0.9, 2.5, 2.3, -4.4]
  tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]
  ```

  Args:
    x: A `Tensor` of type `float` or `double`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.
  """
round_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.round(;Dict(:x=>x, :name=>name)...))
export round_
          

"""
Computes reciprocal of square root of x element-wise.

  I.e., \\(y = 1 / \sqrt{x}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
rsqrt(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.rsqrt(;Dict(:x=>x, :name=>name)...))
export rsqrt
          

"""
Outputs a `Summary` protocol buffer with scalar values.

  The input `tags` and `values` must have the same shape.  The generated
  summary has a summary value for each tag-value pair in `tags` and `values`.

  Args:
    tags: A `string` `Tensor`.  Tags for the summaries.
    values: A real numeric Tensor.  Values for the summaries.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
scalar_summary(tags::Union{AbstractTensor,Void}, values_::Union{AbstractTensor,Void}, collections::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.scalar_summary(;Dict(:tags=>tags, :values=>values_, :collections=>collections, :name=>name)...))
export scalar_summary
          

"""
Adds sparse updates to a variable reference.

  This operation computes

      # Scalar indices
      ref[indices, ...] += updates[...]

      # Vector indices (for each i)
      ref[indices[i], ...] += updates[i, ...]

      # High rank indices (for each i, ..., j)
      ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

  This operation outputs `ref` after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  Duplicate entries are handled correctly: if multiple `indices` reference
  the same location, their contributions add.

  Requires `updates.shape = indices.shape + ref.shape[1:]`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/ScatterAdd.png" alt>
  </div>

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a `Variable` node.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into the first dimension of `ref`.
    updates: A `Tensor`. Must have the same type as `ref`.
      A tensor of updated values to add to `ref`.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the addition will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as `ref`.  Returned as a convenience for operations that want
    to use the updated values after the update is done.
  """
scatter_add(ref::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, updates::Union{AbstractTensor,Void}, use_locking::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.scatter_add(;Dict(:ref=>ref, :indices=>indices, :updates=>updates, :use_locking=>use_locking, :name=>name)...)
export scatter_add
          

"""
Subtracts sparse updates to a variable reference.

      # Scalar indices
      ref[indices, ...] -= updates[...]

      # Vector indices (for each i)
      ref[indices[i], ...] -= updates[i, ...]

      # High rank indices (for each i, ..., j)
      ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]

  This operation outputs `ref` after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  Duplicate entries are handled correctly: if multiple `indices` reference
  the same location, their (negated) contributions add.

  Requires `updates.shape = indices.shape + ref.shape[1:]`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/ScatterSub.png" alt>
  </div>

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Should be from a `Variable` node.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into the first dimension of `ref`.
    updates: A `Tensor`. Must have the same type as `ref`.
      A tensor of updated values to subtract from `ref`.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as `ref`.  Returned as a convenience for operations that want
    to use the updated values after the update is done.
  """
scatter_sub(ref::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, updates::Union{AbstractTensor,Void}, use_locking::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.scatter_sub(;Dict(:ref=>ref, :indices=>indices, :updates=>updates, :use_locking=>use_locking, :name=>name)...)
export scatter_sub
          

"""
Applies sparse updates to a variable reference.

  This operation computes

      # Scalar indices
      ref[indices, ...] = updates[...]

      # Vector indices (for each i)
      ref[indices[i], ...] = updates[i, ...]

      # High rank indices (for each i, ..., j)
      ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]

  This operation outputs `ref` after the update is done.
  This makes it easier to chain operations that need to use the reset value.

  If `indices` contains duplicate entries, lexicographically later entries
  override earlier entries.

  Requires `updates.shape = indices.shape + ref.shape[1:]`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/ScatterUpdate.png" alt>
  </div>

  Args:
    ref: A mutable `Tensor`. Should be from a `Variable` node.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into the first dimension of `ref`.
    updates: A `Tensor`. Must have the same type as `ref`.
      A tensor of updated values to store in `ref`.
    use_locking: An optional `bool`. Defaults to `True`.
      If True, the assignment will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    Same as `ref`.  Returned as a convenience for operations that want
    to use the updated values after the update is done.
  """
scatter_update(ref::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, updates::Union{AbstractTensor,Void}, use_locking::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = tf.scatter_update(;Dict(:ref=>ref, :indices=>indices, :updates=>updates, :use_locking=>use_locking, :name=>name)...)
export scatter_update
          

"""
Computes the maximum along segments of a tensor.

  Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentMax.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
segment_max(data::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.segment_max(;Dict(:data=>data, :segment_ids=>segment_ids, :name=>name)...))
export segment_max
          

"""
Computes the mean along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
  over `j` such that `segment_ids[j] == i` and `N` is the total number of
  values summed.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentMean.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
segment_mean(data::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.segment_mean(;Dict(:data=>data, :segment_ids=>segment_ids, :name=>name)...))
export segment_mean
          

"""
Computes the minimum along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentMin.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
segment_min(data::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.segment_min(;Dict(:data=>data, :segment_ids=>segment_ids, :name=>name)...))
export segment_min
          

"""
Computes the product along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentProd.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
segment_prod(data::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.segment_prod(;Dict(:data=>data, :segment_ids=>segment_ids, :name=>name)...))
export segment_prod
          

"""
Computes the sum along segments of a tensor.

  Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentSum.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
segment_sum(data::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.segment_sum(;Dict(:data=>data, :segment_ids=>segment_ids, :name=>name)...))
export segment_sum
          

"""
Selects elements from `t` or `e`, depending on `condition`.

  The `condition`, `t`, and `e` tensors must all have the same shape,
  and the output will also have that shape. The `condition` tensor acts
  as an element-wise mask that chooses, based on the value at each
  element, whether the corresponding element in the output should be
  taken from `t` (if true) or `e` (if false). For example:

  For example:

  ```prettyprint
  # 'condition' tensor is [[True, False]
  #                        [True, False]]
  # 't' is [[1, 1],
  #         [1, 1]]
  # 'e' is [[2, 2],
  #         [2, 2]]
  select(condition, t, e) ==> [[1, 2],
                               [1, 2]]
  ```

  Args:
    condition: A `Tensor` of type `bool`.
    t:  A `Tensor` with the same shape as `condition`.
    e:  A `Tensor` with the same type and shape as `t`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type and shape as `t` and `e`.
  """
select_(condition::Union{AbstractTensor,Void}, t::Union{AbstractTensor,Void}, e_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.select(;Dict(:condition=>condition, :t=>t, :e=>e_, :name=>name)...))
export select_
          

"""
Calculates the Eigen Decomposition of a square Self-Adjoint matrix.

  Only the lower-triangular part of the input will be used in this case. The
  upper-triangular part will not be read.

  The result is a M+1 x M matrix whose first row is the eigenvalues, and
  subsequent rows are eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M+1, M]`.
  """
self_adjoint_eig(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.self_adjoint_eig(;Dict(:input=>input, :name=>name)...))
export self_adjoint_eig
          

"""
Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sp_input: The input rank `R` `SparseTensor`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string matrix (2-D `Tensor`) with `N` rows and `3` columns.
    Each column represents serialized `SparseTensor`'s indices, values, and
    shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
serialize_many_sparse(sp_input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.serialize_many_sparse(;Dict(:sp_input=>sp_input, :name=>name)...))
export serialize_many_sparse
          

"""
Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string 3-vector (1D `Tensor`), with each column representing the
    serialized `SparseTensor`'s indices, values, and shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
serialize_sparse(sp_input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.serialize_sparse(;Dict(:sp_input=>sp_input, :name=>name)...))
export serialize_sparse
          

"""
Sets the graph-level random seed.

  Operations that rely on a random seed actually derive it from two seeds:
  the graph-level and operation-level seeds. This sets the graph-level seed.

  Its interactions with operation-level seeds is as follows:

    1. If neither the graph-level nor the operation seed is set:
      A random seed is used for this op.
    2. If the graph-level seed is set, but the operation seed is not:
      The system deterministically picks an operation seed in conjunction
      with the graph-level seed so that it gets a unique random sequence.
    3. If the graph-level seed is not set, but the operation seed is set:
      A default graph-level seed and the specified operation seed are used to
      determine the random sequence.
    4. If both the graph-level and the operation seed are set:
      Both seeds are used in conjunction to determine the random sequence.

  To illustrate the user-visible effects, consider these examples:

  To generate different sequences across sessions, set neither
  graph-level nor op-level seeds:

  ```python
  a = tf.random_uniform([1])
  b = tf.random_normal([1])

  print("Session 1")
  with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A3'
    print(sess2.run(a))  # generates 'A4'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'
  ```

  To generate the same repeatable sequence for an op across sessions, set the
  seed for the op:

  ```python
  a = tf.random_uniform([1], seed=1)
  b = tf.random_normal([1])

  # Repeatedly running this block with the same graph will generate the same
  # sequence of values for 'a', but different sequences of values for 'b'.
  print("Session 1")
  with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'
  ```

  To make the random sequences generated by all ops be repeatable across
  sessions, set a graph-level seed:

  ```python
  tf.set_random_seed(1234)
  a = tf.random_uniform([1])
  b = tf.random_normal([1])

  # Repeatedly running this block with the same graph will generate different
  # sequences of 'a' and 'b'.
  print("Session 1")
  with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

  print("Session 2")
  with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'
  ```

  Args:
    seed: integer.
  """
set_random_seed(seed::Union{Int64,Void}) = tf.set_random_seed(;Dict(:seed=>seed)...)
export set_random_seed
          

"""
Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
shape(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.shape(;Dict(:input=>input, :name=>name)...))
export shape
          

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
sigmoid(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sigmoid(;Dict(:x=>x, :name=>name)...))
export sigmoid
          

"""
Returns an element-wise indication of the sign of a number.

  y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
sign_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sign(;Dict(:x=>x, :name=>name)...))
export sign_
          

"""
Computes sin of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
sin_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sin(;Dict(:x=>x, :name=>name)...))
export sin_
          

"""
Returns the size of a tensor.

  This operation returns an integer representing the number of elements in
  `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  size(t) ==> 12
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
size_(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.size(;Dict(:input=>input, :name=>name)...))
export size_
          

"""
Extracts a slice from a tensor.

  This operation extracts a slice of size `size` from a tensor `input` starting
  at the location specified by `begin`. The slice `size` is represented as a
  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
  of `input` that you want to slice. The starting location (`begin`) for the
  slice is represented as an offset in each dimension of `input`. In other
  words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
  want to slice from.

  `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
  all remaining elements in dimension i are included in the
  slice. In other words, this is equivalent to setting:

  `size[i] = input.dim_size(i) - begin[i]`

  This operation requires that:

  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

  For example:

  ```
  # 'input' is [[[1, 1, 1], [2, 2, 2]],
  #             [[3, 3, 3], [4, 4, 4]],
  #             [[5, 5, 5], [6, 6, 6]]]
  tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
  tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                              [4, 4, 4]]]
  tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                             [[5, 5, 5]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    size: An `int32` or `int64` `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input`.
  """
slice_(input_::Union{AbstractTensor,Void}, begin_::Union{AbstractTensor,Void}, size_::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.slice(;Dict(:input_=>input_, :begin=>begin_, :size=>size_, :name=>name)...))
export slice_
          

"""
Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of each sparse input.
  It is assumed that each inputs is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  All inputs' shapes must match, except for the concat dimension.  The
  `indices`, `values`, and `shapes` lists must have the same length.

  The output shape is identical to the inputs', except along the concat
  dimension, where it is the sum of the inputs' sizes along that dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `concat_dim = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Args:
    concat_dim: Dimension to concatenate along.
    sp_inputs: List of `SparseTensor` to concatenate.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `SparseTensor` with the concatenated output.

  Raises:
    TypeError: If `sp_inputs` is not a list of `SparseTensor`.
  """
sparse_concat(concat_dim::Any, sp_inputs::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_concat(;Dict(:concat_dim=>concat_dim, :sp_inputs=>sp_inputs, :name=>name)...))
export sparse_concat
          

"""
Fills empty rows in the input 2-D `SparseTensor` with a default value.

  This op adds entries with the specified `default_value` at index
  `[row, 0]` for any row in the input that does not already have a value.

  For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

      [0, 1]: a
      [0, 3]: b
      [1, 0]: default_value
      [2, 0]: c
      [3, 1]: d
      [4, 0]: default_value

  Note that the input may have empty columns at the end, with no effect on
  this op.

  The output `SparseTensor` will be in row-major order and will have the
  same shape as the input.

  This op also returns an indicator vector such that

      empty_row_indicator[i] = True iff row i was an empty row.

  Args:
    sp_input: A `SparseTensor` with shape `[N, M]`.
    default_value: The value to fill for empty rows, with the same type as
      `sp_input.`
    name: A name prefix for the returned tensors (optional)

  Returns:
    sp_ordered_output: A `SparseTensor` with shape `[N, M]`, and with all empty
      rows filled in with `default_value`.
    empty_row_indicator: A bool vector of length `N` indicating whether each
      input row was empty.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
sparse_fill_empty_rows(sp_input::Union{AbstractTensor,Void}, default_value::Union{Dtype,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_fill_empty_rows(;Dict(:sp_input=>sp_input, :default_value=>default_value, :name=>name)...))
export sparse_fill_empty_rows
          

"""
Masks elements of `IndexedSlices`.

  Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
  contains a subset of the slices of `a`. Only the slices at indices specified
  in `mask_indices` are returned.

  This is useful when you need to extract a subset of slices in an
  `IndexedSlices` object.

  For example:

  ```python
  # `a` contains slices at indices [12, 26, 37, 45] from a large tensor
  # with shape [1000, 10]
  a.indices => [12, 26, 37, 45]
  tf.shape(a.values) => [4, 10]

  # `b` will be the subset of `a` slices at its second and third indices, so
  # we want to mask of its first and last indices (which are at absolute
  # indices 12, 45)
  b = tf.sparse_mask(a, [12, 45])

  b.indices => [26, 37]
  tf.shape(b.values) => [2, 10]

  ```

  Args:
    * `a`: An `IndexedSlices` instance.
    * `mask_indices`: Indices of elements to mask.
    * `name`: A name for the operation (optional).

  Returns:
    The masked `IndexedSlices` instance.
  """
sparse_mask(a::Any, mask_indices::Any, name::Union{AbstractString,Void}=nothing) = tf.sparse_mask(;Dict(:a=>a, :mask_indices=>mask_indices, :name=>name)...)
export sparse_mask
          

"""
Multiply matrix "a" by matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of "a" must
  match the outer dimension of "b". This op is optimized for the case where at
  least one of "a" or "b" is sparse. The breakeven for using this versus a dense
  matrix multiply on one platform was 30% zero values in the sparse matrix.

  Args:
    a: A `Tensor` of type `float32`.
    b: A `Tensor` of type `float32`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
_sparse_mat_mul(a::Union{AbstractTensor,Void}, b::Union{AbstractTensor,Void}, transpose_a::Union{Bool,Void}=nothing, transpose_b::Union{Bool,Void}=nothing, a_is_sparse::Union{Bool,Void}=nothing, b_is_sparse::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf._sparse_mat_mul(;Dict(:a=>a, :b=>b, :transpose_a=>transpose_a, :transpose_b=>transpose_b, :a_is_sparse=>a_is_sparse, :b_is_sparse=>b_is_sparse, :name=>name)...))
export _sparse_mat_mul
          

"""
Reorders a `SparseTensor` into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering
  along increasing dimension number. The only time ordering can be violated
  is during manual manipulation of the indices and values to add entries.

  Reordering does not affect the shape of the `SparseTensor`.

  For example, if `sp_input` has shape `[4, 5]` and `indices` / `values`:

      [0, 3]: b
      [0, 1]: a
      [3, 1]: d
      [2, 0]: c

  then the output will be a `SparseTensor` of shape `[4, 5]` and
  `indices` / `values`:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` with the same shape and non-empty values, but in
    canonical ordering.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
sparse_reorder(sp_input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_reorder(;Dict(:sp_input=>sp_input, :name=>name)...))
export sparse_reorder
          

"""
Retains specified non-empty values within a `SparseTensor`.

  For example, if `sp_input` has shape `[4, 5]` and 4 non-empty string values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  and `to_retain = [True, False, False, True]`, then the output will
  be a `SparseTensor` of shape `[4, 5]` with 2 non-empty values:

      [0, 1]: a
      [3, 1]: d

  Args:
    sp_input: The input `SparseTensor` with `N` non-empty elements.
    to_retain: A bool vector of length `N` with `M` true values.

  Returns:
    A `SparseTensor` with the same shape as the input and `M` non-empty
    elements corresponding to the true positions in `to_retain`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
sparse_retain(sp_input::Union{AbstractTensor,Void}, to_retain::Any) = Tensor(tf.sparse_retain(;Dict(:sp_input=>sp_input, :to_retain=>to_retain)...))
export sparse_retain
          

"""
Computes the mean along sparse segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor` of type `int32`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
sparse_segment_mean(data::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_segment_mean(;Dict(:data=>data, :indices=>indices, :segment_ids=>segment_ids, :name=>name)...))
export sparse_segment_mean
          

"""
Computes gradients for SparseSegmentMean.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor` of type `int32`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
sparse_segment_mean_grad(grad::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, output_dim0::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_segment_mean_grad(;Dict(:grad=>grad, :indices=>indices, :segment_ids=>segment_ids, :output_dim0=>output_dim0, :name=>name)...))
export sparse_segment_mean_grad
          

"""
Computes the sum along sparse segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  For example:

  ```prettyprint
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
    ==> [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
    ==> [[ 1  2  3  4]
         [-1 -2 -3 -4]]

  # Select all rows, two segments.
  tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
    ==> [[0 0 0 0]
         [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    indices: A `Tensor` of type `int32`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
sparse_segment_sum(data::Union{AbstractTensor,Void}, indices::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_segment_sum(;Dict(:data=>data, :indices=>indices, :segment_ids=>segment_ids, :name=>name)...))
export sparse_segment_sum
          

"""
Split a `SparseTensor` into `num_split` tensors along `split_dim`.

  If the `sp_input.shape[split_dim]` is not an integer multiple of `num_split`
  each slice starting from 0:`shape[split_dim] % num_split` gets extra one
  dimension. For example, if `split_dim = 1` and `num_split = 2` and the
  input is:

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] =
      [    a ]
      [b c   ]

      output_tensor[1] =
      [ d e  ]
      [      ]

  Args:
    split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
    num_split: A Python integer. The number of ways to split.
    sp_input: The `SparseTensor` to split.
    name: A name for the operation (optional).

  Returns:
    `num_split` `SparseTensor` objects resulting from splitting `value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
sparse_split(split_dim::Union{AbstractTensor,Void}, num_split::Union{Int64,Void}, sp_input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_split(;Dict(:split_dim=>split_dim, :num_split=>num_split, :sp_input=>sp_input, :name=>name)...))
export sparse_split
          

"""
Converts a `SparseTensor` into a dense tensor.

  This op is a convenience wrapper around `sparse_to_dense` for `SparseTensor`s.

  For example, if `sp_input` has shape `[3, 5]` and non-empty string values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c

  and `default_value` is `x`, then the output will be a dense `[3, 5]`
  string tensor with values:

      [[x a x b x]
       [x x x x x]
       [c x x x x]]

  Indices must be without repeats.  This is only
  tested if validate_indices is True.

  Args:
    sp_input: The input `SparseTensor`.
    default_value: Scalar value to set for indices not specified in
      `sp_input`.  Defaults to zero.
    validate_indices: A boolean value.  If `True`, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A dense tensor with shape `sp_input.shape` and values specified by
    the non-empty values in `sp_input`. Indices not in `sp_input` are assigned
    `default_value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
sparse_tensor_to_dense(sp_input::Union{AbstractTensor,Void}, default_value::Any=0, validate_indices::Any=true, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_tensor_to_dense(;Dict(:sp_input=>sp_input, :default_value=>default_value, :validate_indices=>validate_indices, :name=>name)...))
export sparse_tensor_to_dense
          

"""
Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```python
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values`
  is a scalar, all sparse indices are set to this single value.

  Indices should be sorted in lexicographic order, and indices must not
  contain any repeats. If `validate_indices` is True, these properties
  are checked during execution.

  Args:
    sparse_indices: A 0-D, 1-D, or 2-D `Tensor` of type `int32` or `int64`.
      `sparse_indices[i]` contains the complete index where `sparse_values[i]`
      will be placed.
    output_shape: A 1-D `Tensor` of the same type as `sparse_indices`.  Shape
      of the dense output tensor.
    sparse_values: A 0-D or 1-D `Tensor`.  Values corresponding to each row of
      `sparse_indices`, or a scalar value to be used for all sparse indices.
    default_value: A 0-D `Tensor` of the same type as `sparse_values`.  Value
      to set for indices not specified in `sparse_indices`.  Defaults to zero.
    validate_indices: A boolean value.  If True, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name for the operation (optional).

  Returns:
    Dense `Tensor` of shape `output_shape`.  Has the same type as
    `sparse_values`.
  """
sparse_to_dense(sparse_indices::Union{AbstractTensor,Void}, output_shape::Union{AbstractTensor,Void}, sparse_values::Union{AbstractTensor,Void}, default_value::AbstractTensor=0, validate_indices::Any=true, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_to_dense(;Dict(:sparse_indices=>sparse_indices, :output_shape=>output_shape, :sparse_values=>sparse_values, :default_value=>default_value, :validate_indices=>validate_indices, :name=>name)...))
export sparse_to_dense
          

"""
Converts a `SparseTensor` of ids into a dense bool indicator tensor.

  The last dimension of `sp_input` is discarded and replaced with the values of
  `sp_input`.  If `sp_input.shape = [D0, D1, ..., Dn, K]`, then
  `output.shape = [D0, D1, ..., Dn, vocab_size]`, where

      output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True

  and False elsewhere in `output`.

  For example, if `sp_input.shape = [2, 3, 4]` with non-empty values:

      [0, 0, 0]: 0
      [0, 1, 0]: 10
      [1, 0, 3]: 103
      [1, 1, 2]: 150
      [1, 1, 3]: 149
      [1, 1, 4]: 150
      [1, 2, 1]: 121

  and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
  tensor with False everywhere except at positions

      (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 149), (1, 1, 150),
      (1, 2, 121).

  Note that repeats are allowed in the input SparseTensor.
  This op is useful for converting `SparseTensor`s into dense formats for
  compatibility with ops that expect dense tensors.

  The input `SparseTensor` must be in row-major order.

  Args:
    sp_input: A `SparseTensor` of type `int32` or `int64`.
    vocab_size: The new size of the last dimension, with
      `all(0 <= sp_input.values < vocab_size)`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A dense bool indicator tensor representing the indices with specified value.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
sparse_to_indicator(sp_input::Union{AbstractTensor,Void}, vocab_size::Union{Int64,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sparse_to_indicator(;Dict(:sp_input=>sp_input, :vocab_size=>vocab_size, :name=>name)...))
export sparse_to_indicator
          

"""
Splits a tensor into `num_split` tensors along one dimension.

  Splits `value` along dimension `split_dim` into `num_split` smaller tensors.
  Requires that `num_split` evenly divide `value.shape[split_dim]`.

  For example:

  ```python
  # 'value' is a tensor with shape [5, 30]
  # Split 'value' into 3 tensors along dimension 1
  split0, split1, split2 = tf.split(1, 3, value)
  tf.shape(split0) ==> [5, 10]
  ```

  Args:
    split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
      Must be in the range `[0, rank(value))`.
    num_split: A Python integer. The number of ways to split.
    value: The `Tensor` to split.
    name: A name for the operation (optional).

  Returns:
    `num_split` `Tensor` objects resulting from splitting `value`.
  """
split_(split_dim::Union{AbstractTensor,Void}, num_split::Union{Int64,Void}, value::Union{AbstractTensor,Void}, name::AbstractString="split") = Tensor(tf.split(;Dict(:split_dim=>split_dim, :num_split=>num_split, :value=>value, :name=>name)...))
export split_
          

"""
Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
sqrt_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sqrt(;Dict(:x=>x, :name=>name)...))
export sqrt_
          

"""
Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
square(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.square(;Dict(:x=>x, :name=>name)...))
export square
          

"""
Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `squeeze_dims`.

  For example:

  ```prettyprint
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t)) ==> [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```prettyprint
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
  ```

  Args:
    input: A `Tensor`. The `input` to squeeze.
    squeeze_dims: An optional list of `ints`. Defaults to `[]`.
      If specified, only squeezes the dimensions listed. The dimension
      index starts at 0. It is an error to squeeze a dimension that is not 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but has one or more dimensions of
    size 1 removed.
  """
squeeze_(input::Union{AbstractTensor,Void}, squeeze_dims::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.squeeze(;Dict(:input=>input, :squeeze_dims=>squeeze_dims, :name=>name)...))
export squeeze_
          

"""
Stops gradient computation.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, this op prevents the contribution of
  its inputs to be taken into account.  Normally, the gradient generator adds ops
  to a graph to compute the derivatives of a specified 'loss' by recursively
  finding out inputs that contributed to its computation.  If you insert this op
  in the graph it inputs are masked from the gradient generator.  They are not
  taken into account for computing gradients.

  This is useful any time you want to compute a value with TensorFlow but need
  to pretend that the value was a constant. Some examples include:

  *  The *EM* algorithm where the *M-step* should not involve backpropagation
     through the output of the *E-step*.
  *  Contrastive divergence training of Boltzmann machines where, when
     differentiating the energy function, the training must not backpropagate
     through the graph that generated the samples from the model.
  *  Adversarial training, where no backprop should happen through the adversarial
     example generation process.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
stop_gradient(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.stop_gradient(;Dict(:input=>input, :name=>name)...))
export stop_gradient
          

"""
Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process.

  Note that the hash function may change from time to time.

  Args:
    string_tensor: A `Tensor` of type `string`.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    A Tensor of the same shape as the input `string_tensor`.
  """
string_to_hash_bucket(string_tensor::Union{AbstractTensor,Void}, num_buckets::Union{Int64,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.string_to_hash_bucket(;Dict(:string_tensor=>string_tensor, :num_buckets=>num_buckets, :name=>name)...))
export string_to_hash_bucket
          

"""
Converts each string in the input Tensor to the specified numeric type.

  (Note that int32 overflow results in an error while float overflow
  results in a rounded value.)

  Args:
    string_tensor: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.int32`. Defaults to `tf.float32`.
      The numeric type to interpret each string in string_tensor as.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
    A Tensor of the same shape as the input `string_tensor`.
  """
string_to_number(string_tensor::Union{AbstractTensor,Void}, out_type::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.string_to_number(;Dict(:string_tensor=>string_tensor, :out_type=>out_type, :name=>name)...))
export string_to_number
          

"""
Returns x - y element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `complex64`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
sub_(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.sub(;Dict(:x=>x, :y=>y, :name=>name)...))
export sub_
          

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
tanh_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.tanh(;Dict(:x=>x, :name=>name)...))
export tanh_
          

"""
Constructs a tensor by tiling a given tensor.

  This operation creates a new tensor by replicating `input` `multiples` times.
  The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
  and the values of `input` are replicated `multiples[i]` times along the 'i'th
  dimension. For example, tiling `[a b c d]` by `[2]` produces
  `[a b c d a b c d]`.

  Args:
    input: A `Tensor`. 1-D or higher.
    multiples: A `Tensor` of type `int32`.
      1-D. Length must be the same as the number of dimensions in `input`
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
tile(input::Union{AbstractTensor,Void}, multiples::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.tile(;Dict(:input=>input, :multiples=>multiples, :name=>name)...))
export tile
          

"""
Casts a tensor to type `bfloat16`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `bfloat16`.

  Raises:
    TypeError: If `x` cannot be cast to the `bfloat16`.
  """
to_bfloat16(x::Union{AbstractTensor,Void}, name::AbstractString="ToBFloat16") = Tensor(tf.to_bfloat16(;Dict(:x=>x, :name=>name)...))
export to_bfloat16
          

"""
Casts a tensor to type `float64`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `float64`.

  Raises:
    TypeError: If `x` cannot be cast to the `float64`.
  """
to_double(x::Union{AbstractTensor,Void}, name::AbstractString="ToDouble") = Tensor(tf.to_double(;Dict(:x=>x, :name=>name)...))
export to_double
          

"""
Casts a tensor to type `float32`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `float32`.

  Raises:
    TypeError: If `x` cannot be cast to the `float32`.
  """
to_float(x::Union{AbstractTensor,Void}, name::AbstractString="ToFloat") = Tensor(tf.to_float(;Dict(:x=>x, :name=>name)...))
export to_float
          

"""
Casts a tensor to type `int32`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `int32`.

  Raises:
    TypeError: If `x` cannot be cast to the `int32`.
  """
to_int32(x::Union{AbstractTensor,Void}, name::AbstractString="ToInt32") = Tensor(tf.to_int32(;Dict(:x=>x, :name=>name)...))
export to_int32
          

"""
Casts a tensor to type `int64`.

  Args:
    x: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` with same shape as `x` with type `int64`.

  Raises:
    TypeError: If `x` cannot be cast to the `int64`.
  """
to_int64(x::Union{AbstractTensor,Void}, name::AbstractString="ToInt64") = Tensor(tf.to_int64(;Dict(:x=>x, :name=>name)...))
export to_int64
          

"""
Returns all variables created with `trainable=True`.

  When passed `trainable=True`, the `Variable()` constructor automatically
  adds new variables to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
  contents of that collection.

  Returns:
    A list of Variable objects.
  """
trainable_variables() = tf.trainable_variables(;Dict()...)
export trainable_variables
          

"""
Transposes `a`. Permutes the dimensions according to `perm`.

  The returned tensor's dimension i will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
  the rank of the input tensor. Hence by default, this operation performs a
  regular matrix transpose on 2-D input Tensors.

  For example:

  ```python
  # 'x' is [[1 2 3]
  #         [4 5 6]]
  tf.transpose(x) ==> [[1 4]
                       [2 5]
                       [3 6]]

  # Equivalently
  tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                    [2 5]
                                    [3 6]]

  # 'perm' is more useful for n-dimensional tensors, for n > 2
  # 'x' is   [[[1  2  3]
  #            [4  5  6]]
  #           [[7  8  9]
  #            [10 11 12]]]
  # Take the transpose of the matrices in dimension-0
  tf.transpose(b, perm=[0, 2, 1]) ==> [[[1  4]
                                        [2  5]
                                        [3  6]]

                                       [[7 10]
                                        [8 11]
                                        [9 12]]]
  ```

  Args:
    a: A `Tensor`.
    perm: A permutation of the dimensions of `a`.
    name: A name for the operation (optional).

  Returns:
    A transposed `Tensor`.
  """
transpose_(a::Union{AbstractTensor,Void}, perm::Any=nothing, name::AbstractString="transpose") = Tensor(tf.transpose(;Dict(:a=>a, :perm=>perm, :name=>name)...))
export transpose_
          

"""
Divides x / y elementwise, always producing floating point results.

  The same as `tf.div` for floating point arguments, but casts integer arguments
  to floating point before dividing so that the result is always floating point.
  This op is generated by normal `x / y` division in Python 3 and in Python 2.7
  with `from __future__ import division`.  If you want integer division that
  rounds down, use `x // y` or `tf.floordiv`.

  `x` and `y` must have the same numeric type.  If the inputs are floating
  point, the output will have the same type.  If the inputs are integral, the
  inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
  and `int64` (matching the behavior of Numpy).

  Args:
    x: `Tensor` numerator of numeric type.
    y: `Tensor` denominator of numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` evaluated in floating point.

  Raises:
    TypeError: If `x` and `y` have different dtypes.
  """
truediv(x::Union{AbstractTensor,Void}, y::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Dtype(tf.truediv(;Dict(:x=>x, :y=>y, :name=>name)...))
export truediv
          

"""
Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the truncated normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  """
truncated_normal(shape::Union{AbstractTensor,DimsType,Void}, mean_::AbstractTensor=0.0, stddev::AbstractTensor=1.0, dtype::Dtype=DT_FLOAT32, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.truncated_normal(;Dict(:shape=>shape, :mean=>mean_, :stddev=>stddev, :dtype=>dtype, :seed=>seed, :name=>name)...))
export truncated_normal
          

"""
Returns an initializer that generates a truncated normal distribution.

  These values are similar to values from a `random_normal_initializer`
  except that values more than two standard deviations from the mean
  are discarded and re-drawn. This is the recommended initializer for
  neural network weights and filters.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with a truncated normal
    distribution.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
truncated_normal_initializer(mean_::AbstractTensor=0.0, stddev::AbstractTensor=1.0, seed::Union{Int64,Void}=nothing, dtype::Dtype=DT_FLOAT32) = Tensor(tf.truncated_normal_initializer(;Dict(:mean=>mean_, :stddev=>stddev, :seed=>seed, :dtype=>dtype)...))
export truncated_normal_initializer
          

"""
Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a "join" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `group` and `with_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    name: (optional) A name to use as a `name_scope` for the operation.
    control_inputs: List of additional ops to finish before returning.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
tuple_(tensors::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing, control_inputs::Any=nothing) = Tensor(tf.tuple(;Dict(:tensors=>tensors, :name=>name, :control_inputs=>control_inputs)...))
export tuple_
          

"""
Returns an initializer that generates tensors without scaling variance.

  When initializing a deep network, it is in principle advantageous to keep
  the scale of the input variance constant, so it does not explode or diminish
  by reaching the final layer. If the input is `x` and the operation `x * W`,
  and we want to initialize `W` uniformly at random, we need to pick `W` from

      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

  to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
  A similar calculation for convolutional networks gives an analogous result
  with `dim` equal to the product of the first 3 dimensions.  When
  nonlinearities are present, we need to multiply this by a constant `factor`.
  See <https://arxiv.org/pdf/1412.6558v3.pdf> for deeper motivation, experiments
  and the calculation of constants. In section 2.3 there, the constants were
  numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

  Args:
    factor: Float.  A multiplicative factor by which the values will be scaled.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
    An initializer that generates tensors with unit variance.

  Raises:
    ValueError: if `dtype` is not a floating point type.
  """
uniform_unit_scaling_initializer(factor_::Any=1.0, seed::Union{Int64,Void}=nothing, dtype::Dtype=DT_FLOAT32) = Tensor(tf.uniform_unit_scaling_initializer(;Dict(:factor=>factor_, :seed=>seed, :dtype=>dtype)...))
export uniform_unit_scaling_initializer
          

"""
Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```prettyprint
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx = unique(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  ```

  Args:
    x: A `Tensor`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx).
    y: A `Tensor`. Has the same type as `x`. 1-D.
    idx: A `Tensor` of type `int32`. 1-D.
  """
unique_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.unique(;Dict(:x=>x, :name=>name)...))
export unique_
          

"""
Unpacks the outer dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` along the first dimension.
  If `num` is not specified (the default), it is inferred from `value`'s shape.
  If `value.shape[0]` is not known, `ValueError` is raised.

  The ith tensor in `output` is the slice `value[i, ...]`. Each tensor in
  `output` has shape `value.shape[1:]`.

  This is the opposite of pack.  The numpy equivalent is

      tf.unpack(x, n) = list(x)

  Args:
    value: A rank `R > 0` `Tensor` to be unpacked.
    num: An `int`. The first dimension of value. Automatically inferred if
      `None` (the default).
    name: A name for the operation (optional).

  Returns:
    The list of `Tensor` objects unpacked from `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
  """
unpack(value::Union{AbstractTensor,Void}, num_::Any=nothing, name::AbstractString="unpack") = Tensor(tf.unpack(;Dict(:value=>value, :num=>num_, :name=>name)...))
export unpack
          

"""
Computes the sum along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
  need not be sorted and need not cover all values in the full
    range of valid values.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  `num_segments` should equal the number of distinct segment IDs.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.
    num_segments: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `num_segments`.
  """
unsorted_segment_sum(data::Union{AbstractTensor,Void}, segment_ids::Union{AbstractTensor,Void}, num_segments::Union{Int64,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.unsorted_segment_sum(;Dict(:data=>data, :segment_ids=>segment_ids, :num_segments=>num_segments, :name=>name)...))
export unsorted_segment_sum
          

"""
Returns a context manager for defining an op that creates variables.

  This context manager validates that the given `values` are from the
  same graph, ensures that that graph is the default graph, and pushes a
  name scope and a variable scope.

  If `name` is not None, it is used as is in the variable scope. If `name`
  is None, then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will made unique be appending `_N` to
  it.

  This is intended to be used when defining generic ops and so reuse is always
  inherited.

  For example, to define a new Python op called `my_op_with_vars`:

  ```python
  def my_op_with_vars(a, b, name=None):
    with tf.variable_op_scope([a, b], name, "MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.get_variable('c')
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  Args:
    values: The list of `Tensor` arguments that are passed to the op function.
    name: The name argument that is passed to the op function, this name is not
      uniquified in the variable scope.
    default_name: The default name to use if the `name` argument is `None`, this
      name will be uniquified.
    initializer: A default initializer to pass to variable scope.

  Returns:
    A context manager for use in defining a Python op.

  Raises:
    ValueError: when trying to reuse within a create scope, or create within
      a reuse scope, or if reuse is not `None` or `True`.
    TypeError: when the types of some arguments are not appropriate.
  """
variable_op_scope() = Dtype(tf.variable_op_scope(;Dict()...))
export variable_op_scope
          

"""
Returns a context for variable scope.

  Variable scope allows to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](../../how_tos/variable_scope/index.md),
  here we present only a few basic examples.

  Simple example of how to create a new variable:

  ```python
  with tf.variable_scope("foo"):
      with tf.variable_scope("bar"):
          v = tf.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```

  Basic example of sharing a variable:

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
  with tf.variable_scope("foo", reuse=True):
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.variable_scope("foo") as scope:
      v = tf.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when
  getting an existing variable in a non-reusing scope.

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
      v1 = tf.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```

  Similarly, we raise an exception when trying to get a variable that
  does not exist in reuse mode.

  ```python
  with tf.variable_scope("foo", reuse=True):
      v = tf.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope,
  then all its sub-scopes become reusing as well.

  Args:
    name_or_scope: `string` or `VariableScope`: the scope to open.
    reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
      well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
    initializer: default initializer for variables within this scope.

  Returns:
    A scope that can be to captured and reused.

  Raises:
    ValueError: when trying to reuse within a create scope, or create within
      a reuse scope, or if reuse is not `None` or `True`.
    TypeError: when the types of some arguments are not appropriate.
  """
variable_scope() = Dtype(tf.variable_scope(;Dict()...))
export variable_scope
          

"""
Assert that the tensor does not contain any NaN's or Inf's.

  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).

  Returns:
    Same tensor as `t`.
  """
verify_tensor_all_finite(t::Union{AbstractTensor,Void}, msg::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf.verify_tensor_all_finite(;Dict(:t=>t, :msg=>msg, :name=>name)...))
export verify_tensor_all_finite
          

"""
Returns locations of true values in a boolean tensor.

  This operation returns the coordinates of true elements in `input`. The
  coordinates are returned in a 2-D tensor where the first dimension (rows)
  represents the number of true elements, and the second dimension (columns)
  represents the coordinates of the true elements. Keep in mind, the shape of
  the output tensor can vary depending on how many true values there are in
  `input`. Indices are output in row-major order.

  For example:

  ```prettyprint
  # 'input' tensor is [[True, False]
  #                    [True, False]]
  # 'input' has two true values, so output has two coordinates.
  # 'input' has rank of 2, so coordinates have two indices.
  where(input) ==> [[0, 0],
                    [1, 0]]

  # `input` tensor is [[[True, False]
  #                     [True, False]]
  #                    [[False, True]
  #                     [False, True]]
  #                    [[False, False]
  #                     [False, True]]]
  # 'input' has 5 true values, so output has 5 coordinates.
  # 'input' has rank of 3, so coordinates have three indices.
  where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]
  ```

  Args:
    input: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
where(input::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf.where(;Dict(:input=>input, :name=>name)...))
export where
          

"""
Creates a tensor with all elements set to zero.

  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to zero.

  For example:

  ```python
  tf.zeros([3, 4], int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  ```

  Args:
    shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  """
zeros_(shape::Union{AbstractTensor,DimsType,Void}, dtype::Dtype=DT_FLOAT32, name::Union{AbstractString,Void}=nothing) = Tensor(tf.zeros(;Dict(:shape=>shape, :dtype=>dtype, :name=>name)...))
export zeros_
          

"""
An adaptor for zeros() to match the Initializer spec."""
zeros_initializer(shape::Union{AbstractTensor,DimsType,Void}, dtype::Dtype=DT_FLOAT32) = tf.zeros_initializer(;Dict(:shape=>shape, :dtype=>dtype)...)
export zeros_initializer
          

"""
Creates a tensor with all elements set to zero.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to zero. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  # 'tensor' is [[1, 2, 3], [4, 5, 6]]
  tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
    `int8`, `int16`, `int32`, `int64`, `uint8`, or `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  """
zeros_like(tensor::Union{AbstractTensor,Void}, dtype::Union{Dtype,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf.zeros_like(;Dict(:tensor=>tensor, :dtype=>dtype, :name=>name)...))
export zeros_like
          end
