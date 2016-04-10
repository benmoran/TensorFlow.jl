"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.6.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module TfTrain
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.training.training as tf_train
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.
    """
AdagradOptimizer(learning_rate::Any, initial_accumulator_value::Any=0.1, use_locking::Bool=false, name::AbstractString="Adagrad") = Optimizer(tf_train.AdagradOptimizer(;Dict(:learning_rate=>learning_rate, :initial_accumulator_value=>initial_accumulator_value, :use_locking=>use_locking, :name=>name)...))
export AdagradOptimizer
          

"""
Construct a new Adam optimizer.

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    v_0 <- 0 (Initialize initial 2nd moment vector)
    t <- 0 (Initialize timestep)
    ```

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    ```
    t <- t + 1
    lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
    variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
    ```

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
    """
AdamOptimizer(learning_rate::Any=0.001, beta1::Any=0.9, beta2::Any=0.999, epsilon::Any=1.0e-8, use_locking::Bool=false, name::AbstractString="Adam") = Optimizer(tf_train.AdamOptimizer(;Dict(:learning_rate=>learning_rate, :beta1=>beta1, :beta2=>beta2, :epsilon=>epsilon, :use_locking=>use_locking, :name=>name)...))
export AdamOptimizer
          

"""
Create a new Coordinator."""
Coordinator() = tf_train.Coordinator(;Dict()...)
export Coordinator
          

"""
Creates a new ExponentialMovingAverage object.

    The `Apply()` method has to be called to create shadow variables and add
    ops to maintain moving averages.

    The optional `num_updates` parameter allows one to tweak the decay rate
    dynamically. .  It is typical to pass the count of training steps, usually
    kept in a variable that is incremented at each step, in which case the
    decay rate is lower at the start of training.  This makes moving averages
    move faster.  If passed, the actual decay rate used is:

      `min(decay, (1 + num_updates) / (10 + num_updates))`

    Args:
      decay: Float.  The decay to use.
      num_updates: Optional count of number of updates applied to variables.
      name: String. Optional prefix name to use for the name of ops added in
        `Apply()`.
    """
ExponentialMovingAverage(decay::Any, num_updates::Union{Int64,Void}=nothing, name::AbstractString="ExponentialMovingAverage") = tf_train.ExponentialMovingAverage(;Dict(:decay=>decay, :num_updates=>num_updates, :name=>name)...)
export ExponentialMovingAverage
          

"""
Construct a new FTRL optimizer.

    The Ftrl-proximal algorithm, abbreviated for Follow-the-regularized-leader,
    is described in the paper [Ad Click Prediction: a View from the Trenches](
    https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).

    It can give a good performance vs. sparsity tradeoff.

    Ftrl-proximal uses its own global base learning rate and can behave like
    Adagrad with `learning_rate_power=-0.5`, or like gradient descent with
    `learning_rate_power=0.0`.

    The effective learning rate is adjusted per parameter, relative to this
    base learning rate as:

    ```
    effective_learning_rate_i = (learning_rate /
        pow(k + summed_squared_gradients_for_i, learning_rate_power));
    ```

    where k is the small constant `initial_accumulator_value`.

    Note that the real regularization coefficient of `|w|^2` for objective
    function is `1 / lambda_2` if specifying `l2 = lambda_2` as argument when
    using this function.

    Args:
      learning_rate: A float value or a constant float `Tensor`.
      learning_rate_power: A float value, must be less or equal to zero.
      initial_accumulator_value: The starting value for accumulators.
        Only positive values are allowed.
      l1_regularization_strength: A float value, must be greater than or
        equal to zero.
      l2_regularization_strength: A float value, must be greater than or
        equal to zero.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Ftrl".

    Raises:
      ValueError: if one of the arguments is invalid.
    """
FtrlOptimizer(learning_rate::Any, learning_rate_power::Any=-0.5, initial_accumulator_value::Any=0.1, l1_regularization_strength::Any=0.0, l2_regularization_strength::Any=0.0, use_locking::Bool=false, name::AbstractString="Ftrl") = Optimizer(tf_train.FtrlOptimizer(;Dict(:learning_rate=>learning_rate, :learning_rate_power=>learning_rate_power, :initial_accumulator_value=>initial_accumulator_value, :l1_regularization_strength=>l1_regularization_strength, :l2_regularization_strength=>l2_regularization_strength, :use_locking=>use_locking, :name=>name)...))
export FtrlOptimizer
          

"""
Construct a new gradient descent optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
GradientDescentOptimizer(learning_rate::Any, use_locking::Bool=false, name::AbstractString="GradientDescent") = Optimizer(tf_train.GradientDescentOptimizer(;Dict(:learning_rate=>learning_rate, :use_locking=>use_locking, :name=>name)...))
export GradientDescentOptimizer
          

"""
Create a LooperThread.

    Args:
      coord: a Coordinator.
      timer_interval_secs: Time boundaries at which to call Run(), or None
        if it should be called back to back.
      target: Optional callable object that will be executed in the thread.
      args: Optional arguments to pass to `target` when calling it.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
LooperThread(coord::Any, timer_interval_secs::Any, target::Any=nothing, args::Any=nothing) = tf_train.LooperThread(;Dict(:coord=>coord, :timer_interval_secs=>timer_interval_secs, :target=>target, :args=>args)...)
export LooperThread
          

"""
Construct a new Momentum optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
    """
MomentumOptimizer(learning_rate::Any, momentum::Any, use_locking::Bool=false, name::AbstractString="Momentum") = Optimizer(tf_train.MomentumOptimizer(;Dict(:learning_rate=>learning_rate, :momentum=>momentum, :use_locking=>use_locking, :name=>name)...))
export MomentumOptimizer
          

"""
Create a QueueRunner.

    On construction the `QueueRunner` adds an op to close the queue.  That op
    will be run if the enqueue ops raise exceptions.

    When you later call the `create_threads()` method, the `QueueRunner` will
    create one thread for each op in `enqueue_ops`.  Each thread will run its
    enqueue op in parallel with the other threads.  The enqueue ops do not have
    to all be the same op, but it is expected that they all enqueue tensors in
    `queue`.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
    """
QueueRunner(queue::Any, enqueue_ops::Any) = tf_train.QueueRunner(;Dict(:queue=>queue, :enqueue_ops=>enqueue_ops)...)
export QueueRunner
          

"""
Construct a new RMSProp optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: discounting factor for the history/coming gradient
      momentum: a scalar tensor.
      epsilon: small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      name: Optional name prefic for the operations created when applying
        gradients. Defaults to "RMSProp".
    """
RMSPropOptimizer(learning_rate::Any, decay::Any=0.9, momentum::Any=0.0, epsilon::Any=1.0e-10, use_locking::Bool=false, name::AbstractString="RMSProp") = Optimizer(tf_train.RMSPropOptimizer(;Dict(:learning_rate=>learning_rate, :decay=>decay, :momentum=>momentum, :epsilon=>epsilon, :use_locking=>use_locking, :name=>name)...))
export RMSPropOptimizer
          

"""
Creates a `Saver`.

    The constructor adds ops to save and restore variables.

    `var_list` specifies the variables that will be saved and restored. It can
    be passed as a `dict` or a list:

    * A `dict` of names to variables: The keys are the names that will be
      used to save or restore the variables in the checkpoint files.
    * A list of variables: The variables will be keyed with their op name in
      the checkpoint files.

    For example:

    ```python
    v1 = tf.Variable(..., name='v1')
    v2 = tf.Variable(..., name='v2')

    # Pass the variables as a dict:
    saver = tf.train.Saver({'v1': v1, 'v2': v2})

    # Or pass them as a list.
    saver = tf.train.Saver([v1, v2])
    # Passing a list is equivalent to passing a dict with the variable op names
    # as keys:
    saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
    ```

    The optional `reshape` argument, if `True`, allows restoring a variable from
    a save file where the variable had a different shape, but the same number
    of elements and type.  This is useful if you have reshaped a variable and
    want to reload it from an older checkpoint.

    The optional `sharded` argument, if `True`, instructs the saver to shard
    checkpoints per device.

    Args:
      var_list: A list of `Variable` objects or a dictionary mapping names to
        variables.  If `None`, defaults to the list of all variables.
      reshape: If `True`, allows restoring parameters from a checkpoint
        where the variables have a different shape.
      sharded: If `True`, shard the checkpoints, one per device.
      max_to_keep: maximum number of recent checkpoints to keep.
        Defaults to 10,000 hours.
      keep_checkpoint_every_n_hours: How often to keep checkpoints.
        Defaults to 10,000 hours.
      name: string.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A `Bool`, which if true, causes restore of different
        variables to happen sequentially within each device.  This can lower
        memory usage when restoring very large models.
      saver_def: Optional `SaverDef` proto to use instead of running the
        builder. This is only useful for specialty code that wants to recreate
        a `Saver` object for a previously built `Graph` that had a `Saver`.
        The `saver_def` proto should be the one returned by the
        `as_saver_def()` call of the `Saver` that was created for that `Graph`.
      builder: Optional `SaverBuilder` to use if a `saver_def` was not provided.
        Defaults to `BaseSaverBuilder()`.

    Raises:
      TypeError: If `var_list` is invalid.
      ValueError: If any of the keys or values in `var_list` are not unique.
    """
Saver(var_list::Any=nothing, reshape_::Any=false, sharded::Any=false, max_to_keep::Any=5, keep_checkpoint_every_n_hours::Any=10000.0, name::Union{AbstractString,Void}=nothing, restore_sequentially::Any=false, saver_def::Any=nothing, builder::Any=nothing) = tf_train.Saver(;Dict(:var_list=>var_list, :reshape=>reshape_, :sharded=>sharded, :max_to_keep=>max_to_keep, :keep_checkpoint_every_n_hours=>keep_checkpoint_every_n_hours, :name=>name, :restore_sequentially=>restore_sequentially, :saver_def=>saver_def, :builder=>builder)...)
export Saver
          

"""
Creates a `SummaryWriter` and an event file.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_event()`, or
    `add_graph()`.

    If you pass a `graph_def` protocol buffer to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.Session()
    # Create a summary writer, add the 'graph_def' to the event file.
    writer = tf.train.SummaryWriter(<some-directory>, sess.graph_def)
    ```

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      graph_def: A `GraphDef` protocol buffer.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
    """
SummaryWriter(logdir::Any, graph_def::Any=nothing, max_queue::Any=10, flush_secs::Any=120) = tf_train.SummaryWriter(;Dict(:logdir=>logdir, :graph_def=>graph_def, :max_queue=>max_queue, :flush_secs=>flush_secs)...)
export SummaryWriter
          

"""
Adds a `QueueRunner` to a collection in the graph.

  When building a complex model that uses many queues it is often difficult to
  gather all the queue runners that need to be run.  This convenience function
  allows you to add a queue runner to a well known collection in the graph.

  The companion method `start_queue_runners()` can be used to start threads for
  all the collected queue runners.

  Args:
    qr: A `QueueRunner`.
    collection: A `GraphKey` specifying the graph collection to add
      the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.
  """
add_queue_runner(qr_::Any, collection::Any="queue_runners") = tf_train.add_queue_runner(;Dict(:qr=>qr_, :collection=>collection)...)
export add_queue_runner
          

"""
Creates batches of tensors in `tensor_list`.

  This function is implemented using a queue. A `QueueRunner` for the
  queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  If `enqueue_many` is `False`, `tensor_list` is assumed to represent a
  single example.  An input tensor with shape `[x, y, z]` will be output
  as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensor_list` is assumed to represent a
  batch of examples, where the first dimension is indexed by example,
  and all members of `tensor_list` should have the same size in the
  first dimension.  If an input tensor has shape `[*, x, y, z]`, the
  output will have shape `[batch_size, x, y, z]`.  The `capacity` argument
  controls the how long the prefetching is allowed to grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* You must ensure that either (i) the `shapes` argument is
  passed, or (ii) all of the tensors in `tensor_list` must have
  fully-defined shapes. `ValueError` will be raised if neither of
  these conditions holds.

  Args:
    tensor_list: The list of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    num_threads: The number of threads enqueuing `tensor_list`.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list`.
    name: (Optional) A name for the operations.

  Returns:
    A list of tensors with the same number and types as `tensor_list`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list`.
  """
batch(tensor_list::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, num_threads::Int64=1, capacity::Int64=32, enqueue_many::AbstractTensor=false, shapes::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.batch(;Dict(:tensor_list=>tensor_list, :batch_size=>batch_size, :num_threads=>num_threads, :capacity=>capacity, :enqueue_many=>enqueue_many, :shapes=>shapes, :name=>name)...))
export batch
          

"""
Runs a list of tensors to fill a queue to create batches of examples.

  Enqueues a different list of tensors in different threads.
  Implemented using a queue -- a `QueueRunner` for the queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  `len(tensor_list_list)` threads will be started,
  with thread `i` enqueuing the tensors from
  `tensor_list_list[i]`. `tensor_list_list[i1][j]` must match
  `tensor_list_list[i2][j]` in type and shape, except in the first
  dimension if `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensor_list_list[i]` is assumed
  to represent a single example. An input tensor `x` will be output as a
  tensor with shape `[batch_size] + x.shape`.

  If `enqueue_many` is `True`, `tensor_list_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensor_list_list[i]` should have the
  same size in the first dimension.  The slices of any input tensor
  `x` are treated as examples, and the output tensors will have shape
  `[batch_size] + x.shape[1:]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* You must ensure that either (i) the `shapes` argument is
  passed, or (ii) all of the tensors in `tensor_list_list` must have
  fully-defined shapes. `ValueError` will be raised if neither of
  these conditions holds.

  Args:
    tensor_list_list: A list of tuples of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
    name: (Optional) A name for the operations.

  Returns:
    A list of tensors with the same number and types as
    `tensor_list_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list_list`.
  """
batch_join(tensor_list_list::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, capacity::Int64=32, enqueue_many::AbstractTensor=false, shapes::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.batch_join(;Dict(:tensor_list_list=>tensor_list_list, :batch_size=>batch_size, :capacity=>capacity, :enqueue_many=>enqueue_many, :shapes=>shapes, :name=>name)...))
export batch_join
          

"""
Applies exponential decay to the learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step /decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.96, staircase=True)
  optimizer = tf.GradientDescentOptimizer(learning_rate)
  # Passing global_step to minimize() will increment it at each step.
  optimizer.minimize(...my loss..., global_step=global_step)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    staircase: Boolean.  It `True` decay the learning rate at discrete intervals.
    name: string.  Optional name of the operation.  Defaults to 'ExponentialDecay'

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  """
exponential_decay(learning_rate::Union{AbstractTensor,Void}, global_step::Union{AbstractTensor,Void}, decay_steps::Union{AbstractTensor,Void}, decay_rate::Union{AbstractTensor,Void}, staircase::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.exponential_decay(;Dict(:learning_rate=>learning_rate, :global_step=>global_step, :decay_steps=>decay_steps, :decay_rate=>decay_rate, :staircase=>staircase, :name=>name)...))
export exponential_decay
          

"""
Returns CheckpointState proto from the "checkpoint" file.

  If the "checkpoint" file contains a valid CheckpointState
  proto, returns it.

  Args:
    checkpoint_dir: The directory of checkpoints.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Returns:
    A CheckpointState if the state was available, None
    otherwise.
  """
get_checkpoint_state(checkpoint_dir::Any, latest_filename::Any=nothing) = tf_train.get_checkpoint_state(;Dict(:checkpoint_dir=>checkpoint_dir, :latest_filename=>latest_filename)...)
export get_checkpoint_state
          

"""
Small helper to get the global step.

  ```python
  # Creates a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Creates a session.
  sess = tf.Session()
  # Initializes the variable.
  sess.run(global_step_tensor.initializer)
  print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

  global_step: 10
  ```

  Args:
    sess: A brain `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
global_step(sess::Any, global_step_tensor::Union{AbstractTensor,Void}) = tf_train.global_step(;Dict(:sess=>sess, :global_step_tensor=>global_step_tensor)...)
export global_step
          

"""
Finds the filename of latest saved checkpoint file.

  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `Saver.save()`.

  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  """
latest_checkpoint(checkpoint_dir::Any, latest_filename::Any=nothing) = tf_train.latest_checkpoint(;Dict(:checkpoint_dir=>checkpoint_dir, :latest_filename=>latest_filename)...)
export latest_checkpoint
          

"""
Returns tensor `num_epochs` times and then raises an `OutOfRange` error.

  Args:
    tensor: Any `Tensor`.
    num_epochs: An integer (optional).  If specified, limits the number
      of steps the output tensor may be evaluated.
    name: A name for the operations (optional).

  Returns:
    tensor or `OutOfRange`.
  """
limit_epochs(tensor::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.limit_epochs(;Dict(:tensor=>tensor, :num_epochs=>num_epochs, :name=>name)...))
export limit_epochs
          

"""
Save the list of files matching pattern, so it is only computed once.

  Args:
    pattern: A file pattern (glob).
    name: A name for the operations (optional).

  Returns:
    A variable that is initialized to the list of files matching pattern.
  """
match_filenames_once(pattern::Any, name::Union{AbstractString,Void}=nothing) = tf_train.match_filenames_once(;Dict(:pattern=>pattern, :name=>name)...)
export match_filenames_once
          

"""
Produces the integers from 0 to limit-1 in a queue.

  Args:
    limit: An int32 scalar tensor.
    num_epochs: An integer (optional). If specified, `range_input_producer`
      produces each integer `num_epochs` times before generating an
      OutOfRange error. If not specified, `range_input_producer` can cycle
      through the integers an unlimited number of times.
    shuffle: Boolean. If true, the integers are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    name: A name for the operations (optional).

  Returns:
    A Queue with the output integers.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.
  """
range_input_producer(limit::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, shuffle_::Bool=true, seed::Union{Int64,Void}=nothing, capacity::Int64=32, name::Union{AbstractString,Void}=nothing) = tf_train.range_input_producer(;Dict(:limit=>limit, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :name=>name)...)
export range_input_producer
          

"""
Creates batches by randomly shuffling tensors.

  This function adds the following to the current `Graph`:

  * A shuffling queue into which tensors from `tensor_list` are enqueued.
  * A `dequeue_many` operation to create batches from the queue.
  * A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
    from `tensor_list`.

  If `enqueue_many` is `False`, `tensor_list` is assumed to represent a
  single example.  An input tensor with shape `[x, y, z]` will be output
  as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensor_list` is assumed to represent a
  batch of examples, where the first dimension is indexed by example,
  and all members of `tensor_list` should have the same size in the
  first dimension.  If an input tensor has shape `[*, x, y, z]`, the
  output will have shape `[batch_size, x, y, z]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  For example:

  ```python
  # Creates batches of 32 images and 32 labels.
  image_batch, label_batch = tf.train.shuffle_batch(
        [single_image, single_label],
        batch_size=32,
        num_threads=4,
        capacity=50000,
        min_after_dequeue=10000)
  ```

  *N.B.:* You must ensure that either (i) the `shapes` argument is
  passed, or (ii) all of the tensors in `tensor_list` must have
  fully-defined shapes. `ValueError` will be raised if neither of
  these conditions holds.

  Args:
    tensor_list: The list of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    num_threads: The number of threads enqueuing `tensor_list`.
    seed: Seed for the random shuffling within the queue.
    enqueue_many: Whether each tensor in `tensor_list` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list`.
    name: (Optional) A name for the operations.

  Returns:
    A list of tensors with the same number and types as `tensor_list`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list`.
  """
shuffle_batch(tensor_list::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, capacity::Union{Int64,Void}, min_after_dequeue::Any, num_threads::Int64=1, seed::Union{Int64,Void}=nothing, enqueue_many::AbstractTensor=false, shapes::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.shuffle_batch(;Dict(:tensor_list=>tensor_list, :batch_size=>batch_size, :capacity=>capacity, :min_after_dequeue=>min_after_dequeue, :num_threads=>num_threads, :seed=>seed, :enqueue_many=>enqueue_many, :shapes=>shapes, :name=>name)...))
export shuffle_batch
          

"""
Create batches by randomly shuffling tensors.

  This version enqueues a different list of tensors in different threads.
  It adds the following to the current `Graph`:

  * A shuffling queue into which tensors from `tensor_list_list` are enqueued.
  * A `dequeue_many` operation to create batches from the queue.
  * A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
    from `tensor_list_list`.

  `len(tensor_list_list)` threads will be started, with thread `i` enqueuing
  the tensors from `tensor_list_list[i]`. `tensor_list_list[i1][j]` must match
  `tensor_list_list[i2][j]` in type and shape, except in the first dimension if
  `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensor_list_list[i]` is assumed
  to represent a single example.  An input tensor with shape `[x, y,
  z]` will be output as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensor_list_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensor_list_list[i]` should have the
  same size in the first dimension.  If an input tensor has shape `[*, x,
  y, z]`, the output will have shape `[batch_size, x, y, z]`.

  The `capacity` argument controls the how long the prefetching is allowed to
  grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  Args:
    tensor_list_list: A list of tuples of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    seed: Seed for the random shuffling within the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
    name: (Optional) A name for the operations.

  Returns:
    A list of tensors with the same number and types as `tensor_list_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list_list`.
  """
shuffle_batch_join(tensor_list_list::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, capacity::Union{Int64,Void}, min_after_dequeue::Any, seed::Union{Int64,Void}=nothing, enqueue_many::AbstractTensor=false, shapes::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.shuffle_batch_join(;Dict(:tensor_list_list=>tensor_list_list, :batch_size=>batch_size, :capacity=>capacity, :min_after_dequeue=>min_after_dequeue, :seed=>seed, :enqueue_many=>enqueue_many, :shapes=>shapes, :name=>name)...))
export shuffle_batch_join
          

"""
Produces a slice of each `Tensor` in `tensor_list`.

  Implemented using a Queue -- a `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  Args:
    tensor_list: A list of `Tensor` objects. Every `Tensor` in
      `tensor_list` must have the same size in the first dimension.
    num_epochs: An integer (optional). If specified, `slice_input_producer`
      produces each slice `num_epochs` times before generating
      an `OutOfRange` error. If not specified, `slice_input_producer` can cycle
      through the slices an unlimited number of times.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    name: A name for the operations (optional).

  Returns:
    A list of tensors, one for each element of `tensor_list`.  If the tensor
    in `tensor_list` has shape `[N, a, b, .., z]`, then the corresponding output
    tensor will have shape `[a, b, ..., z]`.
  """
slice_input_producer(tensor_list::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, shuffle_::Any=true, seed::Union{Int64,Void}=nothing, capacity::Int64=32, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.slice_input_producer(;Dict(:tensor_list=>tensor_list, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :name=>name)...))
export slice_input_producer
          

"""
Starts all queue runners collected in the graph.

  This is a companion method to `add_queue_runner()`.  It just starts
  threads for all queue runners collected in the graph.  It returns
  the list of all threads.

  Args:
    sess: `Session` used to run the queue ops.  Defaults to the
      default session.
    coord: Optional `Coordinator` for coordinating the started threads.
    daemon: Whether the threads should be marked as `daemons`, meaning
      they don't block program exit.
    start: Set to `False` to only create the threads, not start them.
    collection: A `GraphKey` specifying the graph collection to
      get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

  Returns:
    A list of threads.
  """
start_queue_runners(sess::Any=nothing, coord::Any=nothing, daemon::Bool=true, start_::Bool=true, collection::Any="queue_runners") = tf_train.start_queue_runners(;Dict(:sess=>sess, :coord=>coord, :daemon=>daemon, :start=>start_, :collection=>collection)...)
export start_queue_runners
          

"""
Output strings (e.g. filenames) to a queue for an input pipeline.

  Args:
    string_tensor: A 1-D string tensor with the strings to produce.
    num_epochs: An integer (optional). If specified, `string_input_producer`
      produces each string from `string_tensor` `num_epochs` times before
      generating an OutOfRange error. If not specified, `string_input_producer`
      can cycle through the strings in `string_tensor` an unlimited number of
      times.
    shuffle: Boolean. If true, the strings are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    name: A name for the operations (optional).

  Returns:
    A queue with the output strings.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  Raises:
    ValueError: If the string_tensor is a null Python list.  At runtime,
    will fail with an assertion if string_tensor becomes a null tensor.
  """
string_input_producer(string_tensor::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, shuffle_::Bool=true, seed::Union{Int64,Void}=nothing, capacity::Int64=32, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.string_input_producer(;Dict(:string_tensor=>string_tensor, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :name=>name)...))
export string_input_producer
          

"""
An iterator for reading `Event` protocol buffers from an event file.

  You can use this function to read events written to an event file. It returns
  a Python iterator that yields `Event` protocol buffers.

  Example: Print the contents of an events file.

  ```python
  for e in tf.summary_iterator(path to events file):
      print(e)
  ```

  Example: Print selected summary values.

  ```python
  # This example supposes that the events file contains summaries with a
  # summary value tag 'loss'.  These could have been added by calling
  # `add_summary()`, passing the output of a scalar summary op created with
  # with: `tf.scalar_summary(['loss'], loss_tensor)`.
  for e in tf.summary_iterator(path to events file):
      for v in e.summary.value:
          if v.tag == 'loss':
              print(v.simple_value)
  ```

  See the protocol buffer definitions of
  [Event](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/util/event.proto)
  and
  [Summary](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
  for more information about their attributes.

  Args:
    path: The path to an event file created by a `SummaryWriter`.

  Yields:
    `Event` protocol buffers.
  """
summary_iterator(path::Any) = tf_train.summary_iterator(;Dict(:path=>path)...)
export summary_iterator
          

"""
Updates the content of the 'checkpoint' file.

  This updates the checkpoint file containing a CheckpointState
  proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: list of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Raises:
    RuntimeError: If the save paths conflict.
  """
update_checkpoint_state(save_dir::Any, model_checkpoint_path::Any, all_model_checkpoint_paths::Any=nothing, latest_filename::Any=nothing) = tf_train.update_checkpoint_state(;Dict(:save_dir=>save_dir, :model_checkpoint_path=>model_checkpoint_path, :all_model_checkpoint_paths=>all_model_checkpoint_paths, :latest_filename=>latest_filename)...)
export update_checkpoint_state
          

"""
Writes a graph proto on disk.

  The graph is written as a binary proto unless `as_text` is `True`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_def: A `GraphDef` protocol buffer.
    logdir: Directory where to write the graph.
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.
  """
write_graph(graph_def::Any, logdir::Any, name::Union{AbstractString,Void}, as_text::Bool=true) = tf_train.write_graph(;Dict(:graph_def=>graph_def, :logdir=>logdir, :name=>name, :as_text=>as_text)...)
export write_graph
          end
