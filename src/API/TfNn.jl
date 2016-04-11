"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.6.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module TfNn
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.ops.nn as tf_nn
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Generate the set of all classes.

  Deterministically generates and returns the set of all possible classes.
  For testing purposes.  There is no need to use this, since you might as
  well use full softmax or full logistic regression.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of possible classes.
    unique: A `bool`. Ignored.
      unique.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      This operation deterministically returns the entire range
      `[0, num_sampled]`.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`. All returned values are 1.0.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`. All returned values are 1.0.
  """
all_candidate_sampler(true_classes::Union{AbstractTensor,Void}, num_true::Union{Int64,Void}, num_sampled::Union{Int64,Void}, unique_::Any, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.all_candidate_sampler(;Dict(:true_classes=>true_classes, :num_true=>num_true, :num_sampled=>num_sampled, :unique=>unique_, :seed=>seed, :name=>name)...))
export all_candidate_sampler


"""
Performs the average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
      `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
    ksize: A list of ints that has length >= 4.
      The size of the window for each dimension of the input tensor.
    strides: A list of ints that has length >= 4.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same type as `value`.  The average pooled output tensor.
  """
avg_pool(value::Union{AbstractTensor,Void}, ksize::Any, strides_::Union{PyVectorType,Void}, padding::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.avg_pool(;Dict(:value=>value, :ksize=>ksize, :strides=>strides_, :padding=>padding, :name=>name)...))
export avg_pool


"""
Batch normalization.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
batch_norm_with_global_normalization(t::Union{AbstractTensor,Void}, m_::Union{AbstractTensor,Void}, v::Union{AbstractTensor,Void}, beta_::Union{AbstractTensor,Void}, gamma_::Union{AbstractTensor,Void}, variance_epsilon::Any, scale_after_normalization::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.batch_norm_with_global_normalization(;Dict(:t=>t, :m=>m_, :v=>v, :beta=>beta_, :gamma=>gamma_, :variance_epsilon=>variance_epsilon, :scale_after_normalization=>scale_after_normalization, :name=>name)...))
export batch_norm_with_global_normalization


"""
Adds `bias` to `value`.

  This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.
  Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
  case where both types are quantized.

  Args:
    value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
      `int16`, `int8`, or `complex64`.
    bias: A 1-D `Tensor` with size matching the last dimension of `value`.
      Must be the same type as `value` unless `value` is a quantized type,
      in which case a different quantized type may be used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `value`.
  """
bias_add(value::Union{AbstractTensor,Void}, bias::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.bias_add(;Dict(:value=>value, :bias=>bias, :name=>name)...))
export bias_add


"""
Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      [batch_size x cell.state_size].
    initial_state_bw: (optional) Same as for initial_state_fw.
    dtype: (optional) The data type for the initial state.  Required if either
      of the initial states are not provided.
    sequence_length: (optional) An int64 vector (tensor) of size [batch_size],
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A set of output `Tensors` where:
      outputs is a length T list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs

  Raises:
    TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
bidirectional_rnn(cell_fw::Any, cell_bw::Any, inputs::Union{AbstractTensor,Void}, initial_state_fw::Any=nothing, initial_state_bw::Any=nothing, dtype::Union{Dtype,Void}=nothing, sequence_length::Union{AbstractTensor,Void}=nothing, scope::Any=nothing) = Tensor(tf_nn.bidirectional_rnn(;Dict(:cell_fw=>cell_fw, :cell_bw=>cell_bw, :inputs=>inputs, :initial_state_fw=>initial_state_fw, :initial_state_bw=>initial_state_bw, :dtype=>dtype, :sequence_length=>sequence_length, :scope=>scope)...))
export bidirectional_rnn


"""
Compute the position ids in `sampled_candidates` matching `true_classes`.

  In Candidate Sampling, this operation facilitates virtually removing
  sampled classes which happen to match target classes.  This is done
  in Sampled Softmax and Sampled Logistic.

  See our [Candidate Sampling Algorithms
  Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf).

  We presuppose that the `sampled_candidates` are unique.

  We call it an 'accidental hit' when one of the target classes
  matches one of the sampled classes.  This operation reports
  accidental hits as triples `(index, id, weight)`, where `index`
  represents the row number in `true_classes`, `id` represents the
  position in `sampled_candidates`, and weight is `-FLOAT_MAX`.

  The result of this op should be passed through a `sparse_to_dense`
  operation, then added to the logits of the sampled classes. This
  removes the contradictory effect of accidentally sampling the true
  target classes as noise classes for the same example.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled_candidates output of CandidateSampler.
    num_true: An `int`.  The number of target classes per training example.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    indices: A `Tensor` of type `int32` and shape `[num_accidental_hits]`.
      Values indicate rows in `true_classes`.
    ids: A `Tensor` of type `int64` and shape `[num_accidental_hits]`.
      Values indicate positions in `sampled_candidates`.
    weights: A `Tensor` of type `float` and shape `[num_accidental_hits]`.
      Each value is `-FLOAT_MAX`.

  """
compute_accidental_hits(true_classes::Union{AbstractTensor,Void}, sampled_candidates::Union{AbstractTensor,Void}, num_true::Union{Int64,Void}, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.compute_accidental_hits(;Dict(:true_classes=>true_classes, :sampled_candidates=>sampled_candidates, :num_true=>num_true, :seed=>seed, :name=>name)...))
export compute_accidental_hits


"""
Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
conv2d(input::Union{AbstractTensor,Void}, filter_::Union{AbstractTensor,Void}, strides_::Union{PyVectorType,Void}, padding::Union{AbstractString,Void}, use_cudnn_on_gpu::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.conv2d(;Dict(:input=>input, :filter=>filter_, :strides=>strides_, :padding=>padding, :use_cudnn_on_gpu=>use_cudnn_on_gpu, :name=>name)...))
export conv2d


"""
Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. 4-D with shape
    `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
    the `filter` input of the convolution.
  """
conv2d_backprop_filter(input::Union{AbstractTensor,Void}, filter_sizes::Union{AbstractTensor,Void}, out_backprop::Union{AbstractTensor,Void}, strides_::Union{PyVectorType,Void}, padding::Union{AbstractString,Void}, use_cudnn_on_gpu::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.conv2d_backprop_filter(;Dict(:input=>input, :filter_sizes=>filter_sizes, :out_backprop=>out_backprop, :strides=>strides_, :padding=>padding, :use_cudnn_on_gpu=>use_cudnn_on_gpu, :name=>name)...))
export conv2d_backprop_filter


"""
Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
    4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
    w.r.t. the input of the convolution.
  """
conv2d_backprop_input(input_sizes::Union{AbstractTensor,Void}, filter_::Union{AbstractTensor,Void}, out_backprop::Union{AbstractTensor,Void}, strides_::Union{PyVectorType,Void}, padding::Union{AbstractString,Void}, use_cudnn_on_gpu::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.conv2d_backprop_input(;Dict(:input_sizes=>input_sizes, :filter=>filter_, :out_backprop=>out_backprop, :strides=>strides_, :padding=>padding, :use_cudnn_on_gpu=>use_cudnn_on_gpu, :name=>name)...))
export conv2d_backprop_input


"""
The transpose of `conv2d`.

  This operation is sometimes called "deconvolution" after (Deconvolutional
  Networks)[http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf], but is
  actually the transpose (gradient) of `conv2d` rather than an actual
  deconvolution.

  Args:
    value: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]`.
    filter: A 4-D `Tensor` with the same type as `value` and shape
      `[height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the returned tensor.

  Returns:
    A `Tensor` with the same type as `value`.

  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
conv2d_transpose(value::Union{AbstractTensor,Void}, filter_::Union{AbstractTensor,Void}, output_shape::Union{AbstractTensor,Void}, strides_::Union{PyVectorType,Void}, padding::Any="SAME", name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.conv2d_transpose(;Dict(:value=>value, :filter=>filter_, :output_shape=>output_shape, :strides=>strides_, :padding=>padding, :name=>name)...))
export conv2d_transpose


"""
Depthwise 2-D convolution.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.

  In detail,

      output[b, i, j, k * channel_multiplier + q] =
          sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                       filter[di, dj, k, q]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter: 4-D with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
    strides: 1-D of size 4.  The stride of the sliding window for each
      dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
    name: A name for this operation (optional).

  Returns:
    A 4-D `Tensor` of shape
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
depthwise_conv2d(input::Any, filter_::Any, strides_::Union{PyVectorType,Void}, padding::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.depthwise_conv2d(;Dict(:input=>input, :filter=>filter_, :strides=>strides_, :padding=>padding, :name=>name)...))
export depthwise_conv2d


"""
Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.
  """
dropout(x::Union{AbstractTensor,Void}, keep_prob::Union{AbstractTensor,Void}, noise_shape::Union{AbstractTensor,Void}=nothing, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.dropout(;Dict(:x=>x, :keep_prob=>keep_prob, :noise_shape=>noise_shape, :seed=>seed, :name=>name)...))
export dropout


"""
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
elu(features::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.elu(;Dict(:features=>features, :name=>name)...))
export elu


"""
Looks up `ids` in a list of embedding tensors.

  This function is used to perform parallel lookups on the list of
  tensors in `params`.  It is a generalization of
  [`tf.gather()`](../../api_docs/python/array_ops.md#gather), where `params` is
  interpreted as a partition of a larger embedding tensor.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` according to the `partition_strategy`.
  In all strategies, if the id space does not evenly divide the number of
  partitions, each of the first `(max_id + 1) % len(params)` partitions will
  be assigned one more id.

  If `partition_strategy` is `"mod"`, we assign each id to partition
  `p = id % len(params)`. For instance,
  13 ids are split across 5 partitions as:
  `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

  If `partition_strategy` is `"div"`, we assign ids to partitions in a
  contiguous manner. In this case, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A list of tensors with the same type and which can be concatenated
      along dimension 0. Each `Tensor` must be appropriately sized for the given
      `partition_strategy`.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """
embedding_lookup(params::Union{AbstractTensor,Void}, ids::Union{AbstractTensor,Void}, partition_strategy::Any="mod", name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.embedding_lookup(;Dict(:params=>params, :ids=>ids, :partition_strategy=>partition_strategy, :name=>name)...))
export embedding_lookup


"""
Computes embeddings for the given ids and weights.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor,
      or a list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.
    sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
      where N is typically batch size and M is arbitrary.
    sp_weights: either a SparseTensor of float / double weights, or None to
      indicate all weights should be taken to be 1. If specified, sp_weights
      must have exactly the same shape and indices as sp_ids.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean" and "sum"
      are supported.
      "sum" computes the weighted sum of the embedding results for each row.
      "mean" is the weighted sum divided by the total weight.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by sp_ids, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if
      shape(combined params) = [p0, p1, ..., pm]
    and
      shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]
    then
      shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0

    with combiner="mean", then the output will be a 3x20 matrix where
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = params[0, :] * 1.0
      output[2, :] = params[1, :] * 3.0

  Raises:
    TypeError: If sp_ids is not a SparseTensor, or if sp_weights is neither
      None nor SparseTensor.
    ValueError: If combiner is not one of {"mean", "sum"}.
  """
embedding_lookup_sparse(params::Union{AbstractTensor,Void}, sp_ids::Union{AbstractTensor,Void}, sp_weights::Union{AbstractTensor,Void}, partition_strategy::Any="mod", name::Union{AbstractString,Void}=nothing, combiner::Any="mean") = Tensor(tf_nn.embedding_lookup_sparse(;Dict(:params=>params, :sp_ids=>sp_ids, :sp_weights=>sp_weights, :partition_strategy=>partition_strategy, :name=>name, :combiner=>combiner)...))
export embedding_lookup_sparse


"""
Samples a set of classes using the provided (fixed) base distribution.

  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max]`.

  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.

  The base distribution is read from a file or passed in as an
  in-memory array. There is also an option to skew the distribution by
  applying a distortion power to the weights.

  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in [this
  document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    vocab_file: Each valid line in this file (which should have a CSV-like
      format) corresponds to a valid word ID. IDs are in sequential order,
      starting from num_reserved_ids. The last entry in each line is expected
      to be a value corresponding to the count or relative probability. Exactly
      one of `vocab_file` and `unigrams` needs to be passed to this operation.
    distortion: The distortion is used to skew the unigram probability
      distribution.  Each weight is first raised to the distortion's power
      before adding to the internal unigram distribution. As a result,
      `distortion = 1.0` gives regular unigram sampling (as defined by the vocab
      file), and `distortion = 0.0` gives a uniform distribution.
    num_reserved_ids: Optionally some reserved IDs can be added in the range
      `[0, num_reserved_ids]` by the users. One use case is that a special
      unknown word token is used as ID 0. These IDs will have a sampling
      probability of 0.
    num_shards: A sampler can be used to sample from a subset of the original
      range in order to speed up the whole computation through parallelism. This
      parameter (together with `shard`) indicates the number of partitions that
      are being used in the overall computation.
    shard: A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This
      parameter (together with `num_shards`) indicates the particular partition
      number of the operation, when partitioning is being used.
    unigrams: A list of unigram counts or probabilities, one per ID in
      sequential order. Exactly one of `vocab_file` and `unigrams` should be
      passed to this operation.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.

  """
fixed_unigram_candidate_sampler(true_classes::Union{AbstractTensor,Void}, num_true::Union{Int64,Void}, num_sampled::Union{Int64,Void}, unique_::Any, range_max::Any, vocab_file::Any="", distortion::Any=1.0, num_reserved_ids::Int64=0, num_shards::Int64=1, shard::Any=0, unigrams::Any=Any[], seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.fixed_unigram_candidate_sampler(;Dict(:true_classes=>true_classes, :num_true=>num_true, :num_sampled=>num_sampled, :unique=>unique_, :range_max=>range_max, :vocab_file=>vocab_file, :distortion=>distortion, :num_reserved_ids=>num_reserved_ids, :num_shards=>num_shards, :shard=>shard, :unigrams=>unigrams, :seed=>seed, :name=>name)...))
export fixed_unigram_candidate_sampler


"""
Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  \$\$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)\$\$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. Computed Precision at `k` as a `bool Tensor`.
  """
in_top_k(predictions::Union{AbstractTensor,Void}, targets::Union{AbstractTensor,Void}, k::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.in_top_k(;Dict(:predictions=>predictions, :targets=>targets, :k=>k, :name=>name)...))
export in_top_k


"""
L2 Loss.

  Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `qint8`, `quint8`, `qint32`.
      Typically 2-D, but may have any dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`. 0-D.
  """
l2_loss(t::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.l2_loss(;Dict(:t=>t, :name=>name)...))
export l2_loss


"""
Normalizes along dimension `dim` using an L2 norm.

  For a 1-D tensor with `dim = 0`, computes

      output = x / sqrt(max(sum(x**2), epsilon))

  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `dim`.

  Args:
    x: A `Tensor`.
    dim: Dimension along which to normalize.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` with the same shape as `x`.
  """
l2_normalize(x::Union{AbstractTensor,Void}, dim::Any, epsilon::Any=1.0e-12, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.l2_normalize(;Dict(:x=>x, :dim=>dim, :epsilon=>epsilon, :name=>name)...))
export l2_normalize


"""
Samples a set of classes from a distribution learned during training.

  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max]`.

  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.

  The base distribution for this operation is constructed on the fly
  during training.  It is a unigram distribution over the target
  classes seen so far during training.  Every integer in `[0, range_max]`
  begins with a weight of 1, and is incremented by 1 each time it is
  seen as a target class.  The base distribution is not saved to checkpoints,
  so it is reset when the model is reloaded.

  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in [this
  document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.

  """
learned_unigram_candidate_sampler(true_classes::Union{AbstractTensor,Void}, num_true::Union{Int64,Void}, num_sampled::Union{Int64,Void}, unique_::Any, range_max::Any, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.learned_unigram_candidate_sampler(;Dict(:true_classes=>true_classes, :num_true=>num_true, :num_sampled=>num_sampled, :unique=>unique_, :range_max=>range_max, :seed=>seed, :name=>name)...))
export learned_unigram_candidate_sampler


"""
Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum ** beta)

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)]
  (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor` of type `float32`. 4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
lrn(input::Union{AbstractTensor,Void}, depth_radius::Any=nothing, bias::Any=nothing, alpha::Any=nothing, beta_::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.lrn(;Dict(:input=>input, :depth_radius=>depth_radius, :bias=>bias, :alpha=>alpha, :beta=>beta_, :name=>name)...))
export lrn


"""
Samples a set of classes using a log-uniform (Zipfian) base distribution.

  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max]`.

  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.

  The base distribution for this operation is an approximately log-uniform
  or Zipfian distribution:

  `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

  This sampler is useful when the target classes approximately follow such
  a distribution - for example, if the classes represent words in a lexicon
  sorted in decreasing order of frequency. If your classes are not ordered by
  decreasing frequency, do not use this op.

  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in [this
  document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.
  """
log_uniform_candidate_sampler(true_classes::Union{AbstractTensor,Void}, num_true::Union{Int64,Void}, num_sampled::Union{Int64,Void}, unique_::Any, range_max::Any, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.log_uniform_candidate_sampler(;Dict(:true_classes=>true_classes, :num_true=>num_true, :num_sampled=>num_sampled, :unique=>unique_, :range_max=>range_max, :seed=>seed, :name=>name)...))
export log_uniform_candidate_sampler


"""
Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum ** beta)

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)]
  (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor` of type `float32`. 4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
lrn(input::Union{AbstractTensor,Void}, depth_radius::Any=nothing, bias::Any=nothing, alpha::Any=nothing, beta_::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.lrn(;Dict(:input=>input, :depth_radius=>depth_radius, :bias=>bias, :alpha=>alpha, :beta=>beta_, :name=>name)...))
export lrn


"""
Performs the max pooling on the input.

  Args:
    value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
      type `tf.float32`.
    ksize: A list of ints that has length >= 4.  The size of the window for
      each dimension of the input tensor.
    strides: A list of ints that has length >= 4.  The stride of the sliding
      window for each dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
    name: Optional name for the operation.

  Returns:
    A `Tensor` with type `tf.float32`.  The max pooled output tensor.
  """
max_pool(value::Union{AbstractTensor,Void}, ksize::Any, strides_::Union{PyVectorType,Void}, padding::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.max_pool(;Dict(:value=>value, :ksize=>ksize, :strides=>strides_, :padding=>padding, :name=>name)...))
export max_pool


"""
Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index
  `((b * height + y) * width + x) * channels + c`.

  Args:
    input: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).
    output: A `Tensor` of type `float32`. The max pooled output tensor.
    argmax: A `Tensor` of type `Targmax`. 4-D.  The flattened indices of the max values chosen for each output.
  """
max_pool_with_argmax(input::Union{AbstractTensor,Void}, ksize::Any, strides_::Union{PyVectorType,Void}, padding::Union{AbstractString,Void}, Targmax::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.max_pool_with_argmax(;Dict(:input=>input, :ksize=>ksize, :strides=>strides_, :padding=>padding, :Targmax=>Targmax, :name=>name)...))
export max_pool_with_argmax


"""
Calculate the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.

  For so-called "global normalization" needed for convolutional filters pass
  `axes=[0, 1, 2]` (batch, height, width).  For batch normalization pass
  `axes=[0]` (batch).

  Args:
    x: A `Tensor`.
    axes: array of ints.  Axes along which to compute mean and
      variance.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
moments(x::Union{AbstractTensor,Void}, axes::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.moments(;Dict(:x=>x, :axes=>axes, :name=>name)...))
export moments


"""
Computes and returns the noise-contrastive estimation training loss.

  See [Noise-contrastive estimation: A new estimation principle for
  unnormalized statistical models]
  (http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
  Also see our [Candidate Sampling Algorithms Reference]
  (../../extras/candidate_sampling.pdf)

  Note: In the case where `num_true` > 1, we assign to each target class
  the target probability 1 / `num_true` so that the target probabilities
  sum to 1 per-example.

  Note: It would be useful to allow a variable number of target classes per
  example.  We hope to provide this functionality in a future release.
  For now, if you have a variable number of target classes, you can pad them
  out to a constant number by either repeating them or by padding
  with an otherwise unused class.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  If set to
        `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
        learning to generate log-odds instead of log probabilities.  See
        our [Candidate Sampling Algorithms Reference]
        (../../extras/candidate_sampling.pdf).
        Default is False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example NCE losses.
  """
nce_loss(weights::Union{AbstractTensor,Void}, biases::Union{AbstractTensor,Void}, inputs::Union{AbstractTensor,Void}, labels::Union{AbstractTensor,Void}, num_sampled::Union{Int64,Void}, num_classes::Union{Int64,Void}, num_true::Int64=1, sampled_values::Any=nothing, remove_accidental_hits::Any=false, partition_strategy::Any="mod", name::AbstractString="nce_loss") = Tensor(tf_nn.nce_loss(;Dict(:weights=>weights, :biases=>biases, :inputs=>inputs, :labels=>labels, :num_sampled=>num_sampled, :num_classes=>num_classes, :num_true=>num_true, :sampled_values=>sampled_values, :remove_accidental_hits=>remove_accidental_hits, :partition_strategy=>partition_strategy, :name=>name)...))
export nce_loss


"""
Computes rectified linear: `max(features, 0)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
relu(features::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.relu(;Dict(:features=>features, :name=>name)...))
export relu


"""
Computes Rectified Linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `features`.
  """
relu6(features::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.relu6(;Dict(:features=>features, :name=>name)...))
export relu6


"""
Computes Relu(x * weight + biases).

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "nn_relu_layer" is used.

  Returns:
    A 2-D Tensor computing relu(matmul(x, weights) + biases).
    Dimensions typically: batch, out_units.
  """
relu_layer(x::Union{AbstractTensor,Void}, weights::Union{AbstractTensor,Void}, biases::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.relu_layer(;Dict(:x=>x, :weights=>weights, :biases=>biases, :name=>name)...))
export relu_layer


"""
Creates a recurrent neural network specified by RNNCell "cell".

  The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    states = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
      states.append(state)
    return (outputs, states)

  However, a few other options are available:

  An initial state can be provided.
  If sequence_length is provided, dynamic calculation is performed.

  Dynamic calculation returns, at time t:
    (t >= max(sequence_length)
        ? (zeros(output_shape), zeros(state_shape))
        : cell(input, state)

  Thus saving computational time when unrolling past the max sequence length.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state: (optional) An initial state for the RNN.  This must be
      a tensor of appropriate type and shape [batch_size x cell.state_size].
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: An int64 vector (tensor) size [batch_size].
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, states) where:
      outputs is a length T list of outputs (one for each input)
      states is a length T list of states (one state following each input)

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
rnn(cell_::Any, inputs::Union{AbstractTensor,Void}, initial_state::Any=nothing, dtype::Union{Dtype,Void}=nothing, sequence_length::Union{AbstractTensor,Void}=nothing, scope::Any=nothing) = tf_nn.rnn(;Dict(:cell=>cell_, :inputs=>inputs, :initial_state=>initial_state, :dtype=>dtype, :sequence_length=>sequence_length, :scope=>scope)...)
export rnn


"""
Computes and returns the sampled softmax training loss.

  This is a faster way to train a softmax classifier over a huge number of
  classes.

  This operation is for training only.  It is generally an underestimate of
  the full softmax loss.

  At inference time, you can compute full softmax probabilities with the
  expression `tf.nn.softmax(tf.matmul(inputs, weights) + biases)`.

  See our [Candidate Sampling Algorithms Reference]
  (../../extras/candidate_sampling.pdf)

  Also see Section 3 of http://arxiv.org/abs/1412.2007 for the math.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  """
sampled_softmax_loss(weights::Union{AbstractTensor,Void}, biases::Union{AbstractTensor,Void}, inputs::Union{AbstractTensor,Void}, labels::Union{AbstractTensor,Void}, num_sampled::Union{Int64,Void}, num_classes::Union{Int64,Void}, num_true::Int64=1, sampled_values::Any=nothing, remove_accidental_hits::Any=true, partition_strategy::Any="mod", name::AbstractString="sampled_softmax_loss") = Tensor(tf_nn.sampled_softmax_loss(;Dict(:weights=>weights, :biases=>biases, :inputs=>inputs, :labels=>labels, :num_sampled=>num_sampled, :num_classes=>num_classes, :num_true=>num_true, :sampled_values=>sampled_values, :remove_accidental_hits=>remove_accidental_hits, :partition_strategy=>partition_strategy, :name=>name)...))
export sampled_softmax_loss


"""
2-D convolution with separable filters.

  Performs a depthwise convolution that acts separately on channels followed by
  a pointwise convolution that mixes channels.  Note that this is separability
  between dimensions `[1, 2]` and `3`, not spatial separability between
  dimensions `1` and `2`.

  In detail,

      output[b, i, j, k] = sum_{di, dj, q, r]
          input[b, strides[1] * i + di, strides[2] * j + dj, q] *
          depthwise_filter[di, dj, q, r] *
          pointwise_filter[0, 0, q * channel_multiplier + r, k]

  `strides` controls the strides for the depthwise convolution only, since
  the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
  `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: 4-D `Tensor` with shape `[batch, in_height, in_width, in_channels]`.
    depthwise_filter: 4-D `Tensor` with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
      Contains `in_channels` convolutional filters of depth 1.
    pointwise_filter: 4-D `Tensor` with shape
      `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
      filter to mix channels after `depthwise_filter` has convolved spatially.
    strides: 1-D of size 4.  The strides for the depthwise convolution for
      each dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
    name: A name for this operation (optional).

  Returns:
    A 4-D `Tensor` of shape `[batch, out_height, out_width, out_channels]`.
  """
separable_conv2d(input::Union{AbstractTensor,Void}, depthwise_filter::Union{AbstractTensor,Void}, pointwise_filter::Union{AbstractTensor,Void}, strides_::Union{PyVectorType,Void}, padding::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.separable_conv2d(;Dict(:input=>input, :depthwise_filter=>depthwise_filter, :pointwise_filter=>pointwise_filter, :strides=>strides_, :padding=>padding, :name=>name)...))
export separable_conv2d


# """
# Computes sigmoid of `x` element-wise.

#   Specifically, `y = 1 / (1 + exp(-x))`.

#   Args:
#     x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
#       or `qint32`.
#     name: A name for the operation (optional).

#   Returns:
#     A Tensor with the same type as `x` if `x.dtype != qint32`
#       otherwise the return type is `quint8`.
#   """
# sigmoid(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.sigmoid(;Dict(:x=>x, :name=>name)...))
# export sigmoid


"""
Computes sigmoid cross entropy given `logits`.

  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.

  For brevity, let `x = logits`, `z = targets`.  The logistic loss is

      x - x * z + log(1 + exp(-x))

  To ensure stability and avoid overflow, the implementation uses

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `targets` must have the same type and shape.

  Args:
    logits: A `Tensor` of type `float32` or `float64`.
    targets: A `Tensor` of the same type and shape as `logits`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.
  """
sigmoid_cross_entropy_with_logits(logits::Union{AbstractTensor,Void}, targets::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.sigmoid_cross_entropy_with_logits(;Dict(:logits=>logits, :targets=>targets, :name=>name)...))
export sigmoid_cross_entropy_with_logits


"""
Computes softmax activations.

  For each batch `i` and class `j` we have

      softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

  Args:
    logits: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  """
softmax(logits::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.softmax(;Dict(:logits=>logits, :name=>name)...))
export softmax


"""
Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  `logits` and `labels` must have the same shape `[batch_size, num_classes]`
  and the same dtype (either `float32` or `float64`).

  Args:
    logits: Unscaled log probabilities.
    labels: Each row `labels[i]` must be a valid probability distribution.
    name: A name for the operation (optional).

  Returns:
    A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
    softmax cross entropy loss.
  """
softmax_cross_entropy_with_logits(logits::Any, labels::Any, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.softmax_cross_entropy_with_logits(;Dict(:logits=>logits, :labels=>labels, :name=>name)...))
export softmax_cross_entropy_with_logits


"""
Computes softplus: `log(exp(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
softplus(features::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.softplus(;Dict(:features=>features, :name=>name)...))
export softplus


"""
Computes softsign: `features / (abs(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
softsign(features::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.softsign(;Dict(:features=>features, :name=>name)...))
export softsign


"""
RNN that accepts a state saver for time-truncated RNN calculation.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    state_saver: A state saver object with methods `state` and `save_state`.
    state_name: The name to use with the state_saver.
    sequence_length: (optional) An int64 vector (tensor) size [batch_size].
      See the documentation for rnn() for more details about sequence_length.
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, states) where:
      outputs is a length T list of outputs (one for each input)
      states is a length T list of states (one state following each input)

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
state_saving_rnn(cell_::Any, inputs::Union{AbstractTensor,Void}, state_saver::Any, state_name::Any, sequence_length::Union{AbstractTensor,Void}=nothing, scope::Any=nothing) = tf_nn.state_saving_rnn(;Dict(:cell=>cell_, :inputs=>inputs, :state_saver=>state_saver, :state_name=>state_name, :sequence_length=>sequence_length, :scope=>scope)...)
export state_saving_rnn


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
tanh_(x::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.tanh(;Dict(:x=>x, :name=>name)...))
export tanh_


"""
Returns the values and indices of the `k` largest elements for each row.

  \\(values_{i, j}\\) represents the j-th largest element in \\(input_i\\).

  \\(indices_{i, j}\\) gives the column index of the corresponding element,
  such that \\(input_{i, indices_{i, j}} = values_{i, j}\\). If two
  elements are equal, the lower-index element appears first.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`.
      A `batch_size` x `classes` tensor.
    k: An `int` that is `>= 1`.
      Number of top elements to look for within each row.
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).
    values: A `Tensor`. Has the same type as `input`. A `batch_size` x `k` tensor with the `k` largest elements for
      each row.
    indices: A `Tensor` of type `int32`. A `batch_size` x `k` tensor with the index of each value within
      each row.
  """
top_k(input::Union{AbstractTensor,Void}, k::Any, sorted::Union{Bool,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.top_k(;Dict(:input=>input, :k=>k, :sorted=>sorted, :name=>name)...))
export top_k


"""
Samples a set of classes using a uniform base distribution.

  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max]`.

  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.

  The base distribution for this operation is the uniform distribution
  over the range of integers `[0, range_max]`.

  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in [this
  document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.

  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).

  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.
  """
uniform_candidate_sampler(true_classes::Union{AbstractTensor,Void}, num_true::Union{Int64,Void}, num_sampled::Union{Int64,Void}, unique_::Any, range_max::Any, seed::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.uniform_candidate_sampler(;Dict(:true_classes=>true_classes, :num_true=>num_true, :num_sampled=>num_sampled, :unique=>unique_, :range_max=>range_max, :seed=>seed, :name=>name)...))
export uniform_candidate_sampler


"""
Computes matmul(x, weights) + biases.

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "wx_plus_b" is used.

  Returns:
    A 2-D Tensor computing matmul(x, weights) + biases.
    Dimensions typically: batch, out_units.
  """
xw_plus_b(x::Union{AbstractTensor,Void}, weights::Union{AbstractTensor,Void}, biases::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Tensor(tf_nn.xw_plus_b(;Dict(:x=>x, :weights=>weights, :biases=>biases, :name=>name)...))
export xw_plus_b


"""
Returns the fraction of zeros in `value`.

  If `value` is empty, the result is `nan`.

  This is useful in summaries to measure and report sparsity.  For example,

      z = tf.Relu(...)
      summ = tf.scalar_summary('sparsity', tf.zero_fraction(z))

  Args:
    value: A tensor of numeric type.
    name: A name for the operation (optional).

  Returns:
    The fraction of zeros in `value`, with type `float32`.
  """
zero_fraction(value::Union{AbstractTensor,Void}, name::Union{AbstractString,Void}=nothing) = Dtype(tf_nn.zero_fraction(;Dict(:value=>value, :name=>name)...))
export zero_fraction
          end
