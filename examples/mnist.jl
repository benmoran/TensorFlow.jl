# A Multilayer Perceptron implementation example using TensorFlow library, wrapped with TensorFlow.jl.
# Using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
# Adapted from the Python example by Aymeric Damien at https://github.com/aymericdamien/TensorFlow-Examples/

using TensorFlow
using TensorFlow.Train
using TensorFlow.InputData
import TensorFlow: DT_FLOAT32
import TensorFlow.API: relu, softmax_cross_entropy_with_logits, AdamOptimizer,
                       arg_max, equal, cast

mnist = read_data_sets("/tmp/data/", one_hot=true)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 28 * 28 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = Placeholder(DT_FLOAT32, [-1, n_input])
y = Placeholder(DT_FLOAT32, [-1, n_classes])

# Create model
function multilayer_perceptron(_X, _weights, _biases)
  layer_1 = relu(_X * _weights["h1"] + _biases["b1"]) #Hidden layer with RELU activation
  layer_2 = relu(layer_1 * _weights["h2"] + _biases["b2"]) #Hidden layer with RELU activation
  return layer_2 * _weights["out"] + _biases["out"]
end

# Store layers weight & bias
weights = Dict(
    "h1"  => Variable(randn(Tensor, [n_input, n_hidden_1])),
    "h2"  => Variable(randn(Tensor, [n_hidden_1, n_hidden_2])),
    "out" => Variable(randn(Tensor, [n_hidden_2, n_classes]))
)
biases = Dict(
    "b1"  => Variable(randn(Tensor, [n_hidden_1])),
    "b2"  => Variable(randn(Tensor, [n_hidden_2])),
    "out" => Variable(randn(Tensor, [n_classes]))
)

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = mean(softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = minimize(AdamOptimizer(learning_rate), cost) # Adam Optimizer

# Initializing the variables
init = initialize_all_variables()

sess = Session()
try
  run(sess, init)

  # Training cycle
  for epoch in 1:training_epochs
    avg_cost = 0.
    total_batch = div(num_examples(mnist.train), batch_size)

    # Loop over all batches
    for i in 1:total_batch
      batch_xs, batch_ys = next_batch(mnist.train, batch_size)
      # Fit training using batch data
      run(sess, optimizer, FeedDict(x => batch_xs, y => batch_ys))
      # Compute average loss
      batch_average_cost = run(sess, cost, FeedDict(x => batch_xs,
                                                    y => batch_ys))
      avg_cost += batch_average_cost / (total_batch * batch_size)
    end

    # Display logs per epoch step
    if epoch % display_step == 0
      println("Epoch $(epoch)  cost=$(avg_cost)")
    end
  end
  println("Optimization Finished")

  # Test model
  correct_prediction = (arg_max(pred, Tensor(1)) == arg_max(y, Tensor(1)))
  # Calculate accuracy
  accuracy = mean(cast(correct_prediction, DT_FLOAT32))
  acc = run(sess, accuracy, FeedDict(x => images(mnist.test),
                                     y => labels(mnist.test)))
  println("Accuracy:", acc)
finally
  close(sess)
end

