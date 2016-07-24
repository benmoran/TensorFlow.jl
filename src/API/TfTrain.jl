"Generated automatically by TensorFlowBuilder, from TensorFlow Python version 0.9.0"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
module TfTrain
using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.training.training as tf_train
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes


"""
Construct a new Adadelta optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
      rho: A `Tensor` or a floating point value. The decay rate.
      epsilon: A `Tensor` or a floating point value.  A constant epsilon used
               to better conditioning the grad update.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adadelta".
    """
AdadeltaOptimizer(learning_rate::Any=0.001, rho::Any=0.95, epsilon::Any=1.0e-8, use_locking::Bool=false, name::AbstractString="Adadelta") = Optimizer(tf_train.AdadeltaOptimizer(;Dict(:learning_rate=>learning_rate, :rho=>rho, :epsilon=>epsilon, :use_locking=>use_locking, :name=>name)...))
export AdadeltaOptimizer
          

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

    Note that in dense implement of this algorithm, m_t, v_t and variable will 
    update even if g is zero, but in sparse implement, m_t, v_t and variable 
    will not update in iterations g is zero.

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
"""
BytesList() = tf_train.BytesList(;Dict()...)
export BytesList
          

"""
Creates a `ClusterSpec`.

    Args:
      cluster: A dictionary mapping one or more job names to lists of network
        addresses, or a `tf.train.ClusterDef` protocol buffer.

    Raises:
      TypeError: If `cluster` is not a dictionary mapping strings to lists
        of strings, and not a `tf.train.ClusterDef` protobuf.
    """
ClusterSpec(cluster::Any) = tf_train.ClusterSpec(;Dict(:cluster=>cluster)...)
export ClusterSpec
          

"""
Create a new Coordinator.

    Args:
      clean_stop_exception_types: Optional tuple of Exception types that should
        cause a clean stop of the coordinator. If an exception of one of these
        types is reported to `request_stop(ex)` the coordinator will behave as
        if `request_stop(None)` was called.  Defaults to
        `(tf.errors.OutOfRangeError,)` which is used by input queues to signal
        the end of input. When feeding training data from a Python iterator it
        is common to add `StopIteration` to this list.
    """
Coordinator(clean_stop_exception_types::Any=nothing) = tf_train.Coordinator(;Dict(:clean_stop_exception_types=>clean_stop_exception_types)...)
export Coordinator
          

"""
"""
Example() = tf_train.Example(;Dict()...)
export Example
          

"""
Creates a new ExponentialMovingAverage object.

    The `apply()` method has to be called to create shadow variables and add
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
        `apply()`.
    """
ExponentialMovingAverage(decay::Any, num_updates::Union{Int64,Void}=nothing, name::AbstractString="ExponentialMovingAverage") = tf_train.ExponentialMovingAverage(;Dict(:decay=>decay, :num_updates=>num_updates, :name=>name)...)
export ExponentialMovingAverage
          

"""
"""
Feature() = tf_train.Feature(;Dict()...)
export Feature
          

"""
"""
FeatureList() = tf_train.FeatureList(;Dict()...)
export FeatureList
          

"""
"""
FeatureLists() = tf_train.FeatureLists(;Dict()...)
export FeatureLists
          

"""
"""
Features() = tf_train.Features(;Dict()...)
export Features
          

"""
"""
FloatList() = tf_train.FloatList(;Dict()...)
export FloatList
          

"""
Construct a new FTRL optimizer.

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
      ValueError: If one of the arguments is invalid.
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
"""
Int64List() = tf_train.Int64List(;Dict()...)
export Int64List
          

"""
Create a LooperThread.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Time boundaries at which to call Run(), or None
        if it should be called back to back.
      target: Optional callable object that will be executed in the thread.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
LooperThread(coord::Any, timer_interval_secs::Any, target::Any=nothing, args::Any=nothing, kwargs::Any=nothing) = tf_train.LooperThread(;Dict(:coord=>coord, :timer_interval_secs=>timer_interval_secs, :target=>target, :args=>args, :kwargs=>kwargs)...)
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
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer. If specified,
        recreates the QueueRunner from its contents. `queue_runner_def` and the
        other arguments are mutually exclusive.

    Raises:
      ValueError: If both `queue_runner_def` and `queue` are both specified.
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
    """
QueueRunner(queue::Any=nothing, enqueue_ops::Any=nothing, close_op::Any=nothing, cancel_op::Any=nothing, queue_runner_def::Any=nothing) = tf_train.QueueRunner(;Dict(:queue=>queue, :enqueue_ops=>enqueue_ops, :close_op=>close_op, :cancel_op=>cancel_op, :queue_runner_def=>queue_runner_def)...)
export QueueRunner
          

"""
Construct a new RMSProp optimizer.

    Note that in dense implement of this algorithm, m_t and v_t will 
    update even if g is zero, but in sparse implement, m_t and v_t 
    will not update in iterations g is zero.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: Discounting factor for the history/coming gradient
      momentum: A scalar tensor.
      epsilon: Small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      name: Optional name prefix for the operations created when applying
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
      max_to_keep: Maximum number of recent checkpoints to keep.
        Defaults to 5.
      keep_checkpoint_every_n_hours: How often to keep checkpoints.
        Defaults to 10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
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
"""
SaverDef() = tf_train.SaverDef(;Dict()...)
export SaverDef
          

"""
"""
SequenceExample() = tf_train.SequenceExample(;Dict()...)
export SequenceExample
          

"""
Creates a new server with the given definition.

    The `job_name`, `task_index`, and `protocol` arguments are optional, and
    override any information provided in `server_or_cluster_def`.

    Args:
      server_or_cluster_def: A `tf.train.ServerDef` or
        `tf.train.ClusterDef` protocol buffer, or a
        `tf.train.ClusterSpec` object, describing the server to be
        created and/or the cluster of which it is a member.
      job_name: (Optional.) Specifies the name of the job of which the server
        is a member. Defaults to the value in `server_or_cluster_def`, if
        specified.
      task_index: (Optional.) Specifies the task index of the server in its
        job. Defaults to the value in `server_or_cluster_def`, if specified.
        Otherwise defaults to 0 if the server's job has only one task.
      protocol: (Optional.) Specifies the protocol to be used by the server.
        Acceptable values include `"grpc"`. Defaults to the value in
        `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.
      config: (Options.) A `tf.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server
        after creating it. Defaults to `True`.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow server.
    """
Server(server_or_cluster_def::Any, job_name::Any=nothing, task_index::Any=nothing, protocol::Any=nothing, config::Any=nothing, start_::Any=true) = tf_train.Server(;Dict(:server_or_cluster_def=>server_or_cluster_def, :job_name=>job_name, :task_index=>task_index, :protocol=>protocol, :config=>config, :start=>start_)...)
export Server
          

"""
Creates a SessionManager.

    The `local_init_op` is an `Operation` that is run always after a new session
    was created. If `None`, this step is skipped.

    The `ready_op` is an `Operation` used to check if the model is ready.  The
    model is considered ready if that operation returns an empty string tensor.
    If the operation returns non empty string tensor, the elements are
    concatenated and used to indicate to the user why the model is not ready.

    If `ready_op` is `None`, the model is not checked for readiness.

    `recovery_wait_secs` is the number of seconds between checks that
    the model is ready.  It is used by processes to wait for a model to
    be initialized or restored.  Defaults to 30 seconds.

    Args:
      local_init_op: An `Operation` run immediately after session creation.
         Usually used to initialize tables and local variables.
      ready_op: An `Operation` to check if the model is initialized.
      graph: The `Graph` that the model will use.
      recovery_wait_secs: Seconds between checks for the model to be ready.
    """
SessionManager(local_init_op::Any=nothing, ready_op::Any=nothing, graph::Any=nothing, recovery_wait_secs::Any=30) = tf_train.SessionManager(;Dict(:local_init_op=>local_init_op, :ready_op=>ready_op, :graph=>graph, :recovery_wait_secs=>recovery_wait_secs)...)
export SessionManager
          

"""
Creates a `SummaryWriter` and an event file.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_session_log()`,
    `add_event()`, or `add_graph()`.

    If you pass a `Graph` to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.Session()
    # Create a summary writer, add the 'graph' to the event file.
    writer = tf.train.SummaryWriter(<some-directory>, sess.graph)
    ```

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      graph: A `Graph` object, such as `sess.graph`.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      graph_def: DEPRECATED: Use the `graph` argument instead.
    """
SummaryWriter(logdir::Any, graph::Any=nothing, max_queue::Any=10, flush_secs::Any=120, graph_def::Any=nothing) = tf_train.SummaryWriter(;Dict(:logdir=>logdir, :graph=>graph, :max_queue=>max_queue, :flush_secs=>flush_secs, :graph_def=>graph_def)...)
export SummaryWriter
          

"""
Create a `Supervisor`.

    Args:
      graph: A `Graph`.  The graph that the model will use.  Defaults to the
        default `Graph`.  The supervisor may add operations to the graph before
        creating a session, but the graph should not be modified by the caller
        after passing it to the supervisor.
      ready_op: 1-D string `Tensor`.  This tensor is evaluated by supervisors in
        `prepare_or_wait_for_session()` to check if the model is ready to use.
        The model is considered ready if it returns an empty array.  Defaults to
        the tensor returned from `tf.report_uninitialized_variables()`  If
        `None`, the model is not checked for readiness.
      is_chief: If True, create a chief supervisor in charge of initializing
        and restoring the model.  If False, create a supervisor that relies
        on a chief supervisor for inits and restore.
      init_op: `Operation`.  Used by chief supervisors to initialize the model
        when it can not be recovered.  Defaults to an `Operation` that
        initializes all variables.  If `None`, no initialization is done
        automatically unless you pass a value for `init_fn`, see below.
      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
        This feed dictionary will be used when `init_op` is evaluated.
      local_init_op: `Operation`. Used by all supervisors to run initializations
        that should run for every new supervisor instance. By default these
        are table initializers and initializers for local variables.
        If `None`, no further per supervisor-instance initialization is
        done automatically.
      logdir: A string.  Optional path to a directory where to checkpoint the
        model and log events for the visualizer.  Used by chief supervisors.
        The directory will be created if it does not exist.
      summary_op: An `Operation` that returns a Summary for the event logs.
        Used by chief supervisors if a `logdir` was specified.  Defaults to the
        operation returned from merge_all_summaries().  If `None`, summaries are
        not computed automatically.
      saver: A Saver object.  Used by chief supervisors if a `logdir` was
        specified.  Defaults to the saved returned by Saver().
        If `None`, the model is not saved automatically.
      global_step: An integer Tensor of size 1 that counts steps.  The value
        from 'global_step' is used in summaries and checkpoint filenames.
        Default to the op named 'global_step' in the graph if it exists, is of
        rank 1, size 1, and of type tf.int32 ot tf.int64.  If `None` the global
        step is not recorded in summaries and checkpoint files.  Used by chief
        supervisors if a `logdir` was specified.
      save_summaries_secs: Number of seconds between the computation of
        summaries for the event log.  Defaults to 120 seconds.  Pass 0 to
        disable summaries.
      save_model_secs: Number of seconds between the creation of model
        checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.
      recovery_wait_secs: Number of seconds between checks that the model
        is ready.  Used by supervisors when waiting for a chief supervisor
        to initialize or restore the model.  Defaults to 30 seconds.
      stop_grace_secs: Grace period, in seconds, given to running threads to
        stop when `stop()` is called.  Defaults to 120 seconds.
      checkpoint_basename: The basename for checkpoint saving.
      session_manager: `SessionManager`, which manages Session creation and
        recovery. If it is `None`, a default `SessionManager` will be created
        with the set of arguments passed in for backwards compatibility.
      summary_writer: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None`
        to indicate that no summaries should be written.
      init_fn: Optional callable used to initialize the model. Called
        after the optional `init_op` is called.  The callable must accept one
        argument, the session being initialized.

    Returns:
      A `Supervisor`.
    """
Supervisor(graph::Any=nothing, ready_op::Any=0, is_chief::Any=true, init_op::Any=0, init_feed_dict::Any=nothing, local_init_op::Any=0, logdir::Any=nothing, summary_op::Any=0, saver::Any=0, global_step::Any=0, save_summaries_secs::Any=120, save_model_secs::Any=600, recovery_wait_secs::Any=30, stop_grace_secs::Any=120, checkpoint_basename::Any="model.ckpt", session_manager::Any=nothing, summary_writer::Any=0, init_fn::Any=nothing) = tf_train.Supervisor(;Dict(:graph=>graph, :ready_op=>ready_op, :is_chief=>is_chief, :init_op=>init_op, :init_feed_dict=>init_feed_dict, :local_init_op=>local_init_op, :logdir=>logdir, :summary_op=>summary_op, :saver=>saver, :global_step=>global_step, :save_summaries_secs=>save_summaries_secs, :save_model_secs=>save_model_secs, :recovery_wait_secs=>recovery_wait_secs, :stop_grace_secs=>stop_grace_secs, :checkpoint_basename=>checkpoint_basename, :session_manager=>session_manager, :summary_writer=>summary_writer, :init_fn=>init_fn)...)
export Supervisor
          

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
Creates batches of tensors in `tensors`.

  The argument `tensors` can be a list or a dictionary of tensors.
  The value returned by the function will be of the same type
  as `tensors`.

  This function is implemented using a queue. A `QueueRunner` for the
  queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  If `enqueue_many` is `False`, `tensors` is assumed to represent a single
  example.  An input tensor with shape `[x, y, z]` will be output as a tensor
  with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensors` is assumed to represent a batch of
  examples, where the first dimension is indexed by example, and all members of
  `tensor_list` should have the same size in the first dimension.  If an input
  tensor has shape `[*, x, y, z]`, the output will have shape `[batch_size, x,
  y, z]`.  The `capacity` argument controls the how long the prefetching is
  allowed to grow the queues.

  The returned operation is a dequeue operation and will throw
  `tf.errors.OutOfRangeError` if the input queue is exhausted. If this
  operation is feeding another input queue, its queue runner will catch
  this exception, however, if this operation is used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensors` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have shape `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queue is closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape` method will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    num_threads: The number of threads enqueuing `tensor_list`.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same types as `tensors`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors`.
  """
batch(tensors::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, num_threads::Int64=1, capacity::Int64=32, enqueue_many::AbstractTensor=false, shapes::Any=nothing, dynamic_pad::Bool=false, allow_smaller_final_batch::Any=false, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.batch(;Dict(:tensors=>tensors, :batch_size=>batch_size, :num_threads=>num_threads, :capacity=>capacity, :enqueue_many=>enqueue_many, :shapes=>shapes, :dynamic_pad=>dynamic_pad, :allow_smaller_final_batch=>allow_smaller_final_batch, :shared_name=>shared_name, :name=>name)...))
export batch
          

"""
Runs a list of tensors to fill a queue to create batches of examples.

  The `tensors_list` argument is a list of tuples of tensors, or a list of
  dictionaries of tensors.  Each element in the list is treated similarily
  to the `tensors` argument of `tf.train.batch()`.

  Enqueues a different list of tensors in different threads.
  Implemented using a queue -- a `QueueRunner` for the queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  `len(tensors_list)` threads will be started,
  with thread `i` enqueuing the tensors from
  `tensors_list[i]`. `tensors_list[i1][j]` must match
  `tensors_list[i2][j]` in type and shape, except in the first
  dimension if `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
  to represent a single example. An input tensor `x` will be output as a
  tensor with shape `[batch_size] + x.shape`.

  If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensors_list[i]` should have the
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

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensors_list` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have value `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queue is closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape` method will have a first `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list_list[i]`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
    shared_name: (Optional) If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same number and types as
    `tensors_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensor_list_list`.
  """
batch_join(tensors_list::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, capacity::Int64=32, enqueue_many::AbstractTensor=false, shapes::Any=nothing, dynamic_pad::Bool=false, allow_smaller_final_batch::Any=false, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.batch_join(;Dict(:tensors_list=>tensors_list, :batch_size=>batch_size, :capacity=>capacity, :enqueue_many=>enqueue_many, :shapes=>shapes, :dynamic_pad=>dynamic_pad, :allow_smaller_final_batch=>allow_smaller_final_batch, :shared_name=>shared_name, :name=>name)...))
export batch_join
          

"""
"""
do_quantize_training_on_graphdef(input_graph::Any, num_bits::Union{Int64,Void}) = tf_train.do_quantize_training_on_graphdef(;Dict(:input_graph=>input_graph, :num_bits=>num_bits)...)
export do_quantize_training_on_graphdef
          

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

  If the argument `staircase` is `True`, then `global_step / decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
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
    staircase: Boolean.  It `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to 
      'ExponentialDecay'

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  """
exponential_decay(learning_rate::Union{AbstractTensor,Void}, global_step::Union{AbstractTensor,Void}, decay_steps::Union{AbstractTensor,Void}, decay_rate::Union{AbstractTensor,Void}, staircase::Bool=false, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.exponential_decay(;Dict(:learning_rate=>learning_rate, :global_step=>global_step, :decay_steps=>decay_steps, :decay_rate=>decay_rate, :staircase=>staircase, :name=>name)...))
export exponential_decay
          

"""
Returns `MetaGraphDef` proto. Optionally writes it to filename.

  This function exports the graph, saver, and collection objects into
  `MetaGraphDef` protocol buffer with the intension of it being imported
  at a later time or location to restart training, run inference, or be
  a subgraph.

  Args:
    filename: Optional filename including the path for writing the
      generated `MetaGraphDef` protocol buffer.
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.

  Returns:
    A `MetaGraphDef` proto.
  """
export_meta_graph(filename::Any=nothing, meta_info_def::Any=nothing, graph_def::Any=nothing, saver_def::Any=nothing, collection_list::Any=nothing, as_text::Bool=false) = tf_train.export_meta_graph(;Dict(:filename=>filename, :meta_info_def=>meta_info_def, :graph_def=>graph_def, :saver_def=>saver_def, :collection_list=>collection_list, :as_text=>as_text)...)
export export_meta_graph
          

"""
Generates a checkpoint state proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.

  Returns:
    CheckpointState proto with model_checkpoint_path and
    all_model_checkpoint_paths updated to either absolute paths or
    relative paths to the current save_dir.
  """
generate_checkpoint_state_proto(save_dir::Any, model_checkpoint_path::Any, all_model_checkpoint_paths::Any=nothing) = tf_train.generate_checkpoint_state_proto(;Dict(:save_dir=>save_dir, :model_checkpoint_path=>model_checkpoint_path, :all_model_checkpoint_paths=>all_model_checkpoint_paths)...)
export generate_checkpoint_state_proto
          

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

  Raises:
    ValueError: if the checkpoint read doesn't have model_checkpoint_path set.
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
    sess: A TensorFlow `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
global_step(sess::Union{AbstractTensor,Void}, global_step_tensor::Union{AbstractTensor,Void}) = tf_train.global_step(;Dict(:sess=>sess, :global_step_tensor=>global_step_tensor)...)
export global_step
          

"""
Recreates a Graph saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates all the collections, and returns a saver
  constructed from the `saver_def` field.

  In combination with `export_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  ```Python
  ...
  # Create a saver.
  saver = tf.train.Saver(...variables...)
  # Remember the training_op we want to run by adding it to a collection.
  tf.add_to_collection('train_op', train_op)
  sess = tf.Session()
  for step in xrange(1000000):
      sess.run(train_op)
      if step % 1000 == 0:
          # Saves checkpoint, which by default also exports a meta_graph
          # named 'my-model-global_step.meta'.
          saver.save(sess, 'my-model', global_step=step)
  ```

  Later we can continue training from this saved `meta_graph` without building
  the model from scratch.

  ```Python
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() returns a list. In this example we only want the
    # first one.
    train_op = tf.get_collection('train_op')[0]
    for step in xrange(1000000):
      sess.run(train_op)
  ```

  NOTE: Restarting training from saved `meta_graph` only works if the
  device assignments have not changed.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.

  Returns:
    A saver constructed from `saver_def` in `MetaGraphDef` or None.

    A None value is returned if no variables exist in the `MetaGraphDef`
    (i.e., there are no variables to restore).
  """
import_meta_graph(meta_graph_or_file::Any) = tf_train.import_meta_graph(;Dict(:meta_graph_or_file=>meta_graph_or_file)...)
export import_meta_graph
          

"""
Output the rows of `input_tensor` to a queue for an input pipeline.

  Args:
    input_tensor: A tensor with the rows to produce. Must be at
      one-dimensional. Must either have a fully-defined shape, or
      `element_shape` must be defined.
    element_shape: (Optional.) A `TensorShape` representing the shape of a
      row of `input_tensor`, if it cannot be inferred.
    num_epochs: (Optional.) An integer. If specified `input_producer` produces
      each row of `input_tensor` `num_epochs` times before generating an
      `OutOfRange` error. If not specified, `input_producer` can cycle through
      the rows of `input_tensor` an unlimited number of times.
    shuffle: (Optional.) A boolean. If true, the rows are randomly shuffled
      within each eopch.
    seed: (Optional.) An integer. The seed to use if `shuffle` is true.
    capacity: (Optional.) The capacity of the queue to be used for buffering
      the input.
    shared_name: (Optional.) If set, this queue will be shared under the given
      name across multiple sessions.
    summary_name: (Optional.) If set, a scalar summary for the current queue
      size will be generated, using this name as part of the tag.
    name: (Optional.) A name for queue.

  Returns:
    A queue with the output rows.  A `QueueRunner` for the queue is
    added to the current `QUEUE_RUNNER` collection of the current
    graph.

  Raises:
    ValueError: If the shape of the input cannot be inferred from the arguments.
  """
input_producer(input_tensor::Union{AbstractTensor,Void}, element_shape::Union{AbstractTensor,Void}=nothing, num_epochs::Union{Int64,Void}=nothing, shuffle_::Any=true, seed::Union{Int64,Void}=nothing, capacity::Any=32, shared_name::Any=nothing, summary_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = tf_train.input_producer(;Dict(:input_tensor=>input_tensor, :element_shape=>element_shape, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :shared_name=>shared_name, :summary_name=>summary_name, :name=>name)...)
export input_producer
          

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
    num_epochs: A positive integer (optional).  If specified, limits the number
      of steps the output tensor may be evaluated.
    name: A name for the operations (optional).

  Returns:
    tensor or `OutOfRange`.

  Raises:
    ValueError: if `num_epochs` is invalid.
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
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: A name for the operations (optional).

  Returns:
    A Queue with the output integers.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.
  """
range_input_producer(limit::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, shuffle_::Bool=true, seed::Union{Int64,Void}=nothing, capacity::Int64=32, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = tf_train.range_input_producer(;Dict(:limit=>limit, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :shared_name=>shared_name, :name=>name)...)
export range_input_producer
          

"""
Return a `device function` to use when building a Graph for replicas.

  Device Functions are used in `with tf.device(device_function):` statement to
  automatically assign devices to `Operation` objects as they are constructed,
  Device constraints are added from the inner-most context first, working
  outwards. The merging behavior adds constraints to fields that are yet unset
  by a more inner context. Currently the fields are (job, task, cpu/gpu).

  If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.

  For example,

  ```python
  # To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
  # jobs on hosts worker0, worker1 and worker2.
  cluster_spec = {
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
  with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
    # Build your graph
    v1 = tf.Variable(...)  # assigned to /job:ps/task:0
    v2 = tf.Variable(...)  # assigned to /job:ps/task:1
    v3 = tf.Variable(...)  # assigned to /job:ps/task:0
  # Run compute
  ```

  Args:
    ps_tasks: Number of tasks in the `ps` job.
    ps_device: String.  Device of the `ps` job.  If empty no `ps` job is used.
      Defaults to `ps`.
    worker_device: String.  Device of the `worker` job.  If empty no `worker`
      job is used.
    merge_devices: `Boolean`. If `True`, merges or only sets a device if the
      device constraint is completely unset. merges device specification rather
      than overriding them.
    cluster: `ClusterDef` proto or `ClusterSpec`.
    ps_ops: List of `Operation` objects that need to be placed on `ps` devices.

  Returns:
    A function to pass to `tf.device()`.

  Raises:
    TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer.
  """
replica_device_setter(ps_tasks::Any=0, ps_device::Any="/job:ps", worker_device::Any="/job:worker", merge_devices::Any=true, cluster::Any=nothing, ps_ops::Any=nothing) = tf_train.replica_device_setter(;Dict(:ps_tasks=>ps_tasks, :ps_device=>ps_device, :worker_device=>worker_device, :merge_devices=>merge_devices, :cluster=>cluster, :ps_ops=>ps_ops)...)
export replica_device_setter
          

"""
Creates batches by randomly shuffling tensors.

  This function adds the following to the current `Graph`:

  * A shuffling queue into which tensors from `tensors` are enqueued.
  * A `dequeue_many` operation to create batches from the queue.
  * A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
    from `tensors`.

  If `enqueue_many` is `False`, `tensors` is assumed to represent a
  single example.  An input tensor with shape `[x, y, z]` will be output
  as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensors` is assumed to represent a
  batch of examples, where the first dimension is indexed by example,
  and all members of `tensors` should have the same size in the
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
  passed, or (ii) all of the tensors in `tensors` must have
  fully-defined shapes. `ValueError` will be raised if neither of
  these conditions holds.

  Args:
    tensors: The list or dictionary of tensors to enqueue.
    batch_size: The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    num_threads: The number of threads enqueuing `tensor_list`.
    seed: Seed for the random shuffling within the queue.
    enqueue_many: Whether each tensor in `tensor_list` is a single example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensor_list`.
    shared_name: (Optional) If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the types as `tensors`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors`.
  """
shuffle_batch(tensors::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, capacity::Union{Int64,Void}, min_after_dequeue::Any, num_threads::Int64=1, seed::Union{Int64,Void}=nothing, enqueue_many::AbstractTensor=false, shapes::Any=nothing, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.shuffle_batch(;Dict(:tensors=>tensors, :batch_size=>batch_size, :capacity=>capacity, :min_after_dequeue=>min_after_dequeue, :num_threads=>num_threads, :seed=>seed, :enqueue_many=>enqueue_many, :shapes=>shapes, :shared_name=>shared_name, :name=>name)...))
export shuffle_batch
          

"""
Create batches by randomly shuffling tensors.

  The `tensors_list` argument is a list of tuples of tensors, or a list of
  dictionaries of tensors.  Each element in the list is treated similarily
  to the `tensors` argument of `tf.train.shuffle_batch()`.

  This version enqueues a different list of tensors in different threads.
  It adds the following to the current `Graph`:

  * A shuffling queue into which tensors from `tensors_list` are enqueued.
  * A `dequeue_many` operation to create batches from the queue.
  * A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
    from `tensors_list`.

  `len(tensors_list)` threads will be started, with thread `i` enqueuing
  the tensors from `tensors_list[i]`. `tensors_list[i1][j]` must match
  `tensors_list[i2][j]` in type and shape, except in the first dimension if
  `enqueue_many` is true.

  If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
  to represent a single example.  An input tensor with shape `[x, y, z]`
  will be output as a tensor with shape `[batch_size, x, y, z]`.

  If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
  represent a batch of examples, where the first dimension is indexed
  by example, and all members of `tensors_list[i]` should have the
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
    tensors_list: A list of tuples or dictionaries of tensors to enqueue.
    batch_size: An integer. The new batch size pulled from the queue.
    capacity: An integer. The maximum number of elements in the queue.
    min_after_dequeue: Minimum number elements in the queue after a
      dequeue, used to ensure a level of mixing of elements.
    seed: Seed for the random shuffling within the queue.
    enqueue_many: Whether each tensor in `tensor_list_list` is a single
      example.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors_list[i]`.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A list or dictionary of tensors with the same number and types as
    `tensors_list[i]`.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors_list`.
  """
shuffle_batch_join(tensors_list::Union{AbstractTensor,Void}, batch_size::Union{Int64,Void}, capacity::Union{Int64,Void}, min_after_dequeue::Any, seed::Union{Int64,Void}=nothing, enqueue_many::AbstractTensor=false, shapes::Any=nothing, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.shuffle_batch_join(;Dict(:tensors_list=>tensors_list, :batch_size=>batch_size, :capacity=>capacity, :min_after_dequeue=>min_after_dequeue, :seed=>seed, :enqueue_many=>enqueue_many, :shapes=>shapes, :shared_name=>shared_name, :name=>name)...))
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
    shuffle: Boolean. If true, the integers are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: A name for the operations (optional).

  Returns:
    A list of tensors, one for each element of `tensor_list`.  If the tensor
    in `tensor_list` has shape `[N, a, b, .., z]`, then the corresponding output
    tensor will have shape `[a, b, ..., z]`.

  Raises:
    ValueError: if `slice_input_producer` produces nothing from `tensor_list`.
  """
slice_input_producer(tensor_list::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, shuffle_::Bool=true, seed::Union{Int64,Void}=nothing, capacity::Int64=32, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.slice_input_producer(;Dict(:tensor_list=>tensor_list, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :shared_name=>shared_name, :name=>name)...))
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
      generating an `OutOfRange` error. If not specified,
      `string_input_producer` can cycle through the strings in `string_tensor`
      an unlimited number of times.
    shuffle: Boolean. If true, the strings are randomly shuffled within each
      epoch.
    seed: An integer (optional). Seed used if shuffle == True.
    capacity: An integer. Sets the queue capacity.
    shared_name: (optional). If set, this queue will be shared under the given
      name across multiple sessions.
    name: A name for the operations (optional).

  Returns:
    A queue with the output strings.  A `QueueRunner` for the Queue
    is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  Raises:
    ValueError: If the string_tensor is a null Python list.  At runtime,
    will fail with an assertion if string_tensor becomes a null tensor.
  """
string_input_producer(string_tensor::Union{AbstractTensor,Void}, num_epochs::Union{Int64,Void}=nothing, shuffle_::Bool=true, seed::Union{Int64,Void}=nothing, capacity::Int64=32, shared_name::Any=nothing, name::Union{AbstractString,Void}=nothing) = Tensor(tf_train.string_input_producer(;Dict(:string_tensor=>string_tensor, :num_epochs=>num_epochs, :shuffle=>shuffle_, :seed=>seed, :capacity=>capacity, :shared_name=>shared_name, :name=>name)...))
export string_input_producer
          

"""
An iterator for reading `Event` protocol buffers from an event file.

  You can use this function to read events written to an event file. It returns
  a Python iterator that yields `Event` protocol buffers.

  Example: Print the contents of an events file.

  ```python
  for e in tf.train.summary_iterator(path to events file):
      print(e)
  ```

  Example: Print selected summary values.

  ```python
  # This example supposes that the events file contains summaries with a
  # summary value tag 'loss'.  These could have been added by calling
  # `add_summary()`, passing the output of a scalar summary op created with
  # with: `tf.scalar_summary(['loss'], loss_tensor)`.
  for e in tf.train.summary_iterator(path to events file):
      for v in e.summary.value:
          if v.tag == 'loss':
              print(v.simple_value)
  ```

  See the protocol buffer definitions of
  [Event](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
  and
  [Summary](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
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
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
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
Writes a graph proto to a file.

  The graph is written as a binary proto unless `as_text` is `True`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_def: A `GraphDef` protocol buffer.
    logdir: Directory where to write the graph. This can refer to remote
      filesystems, such as Google Cloud Storage (GCS).
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.
  """
write_graph(graph_def::Any, logdir::Any, name::Union{AbstractString,Void}, as_text::Bool=true) = tf_train.write_graph(;Dict(:graph_def=>graph_def, :logdir=>logdir, :name=>name, :as_text=>as_text)...)
export write_graph
          end
