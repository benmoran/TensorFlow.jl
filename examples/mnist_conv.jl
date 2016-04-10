# A conv net implementation example using TensorFlow library, wrapped with TensorFlow.jl.
# Using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

using TensorFlow
using TensorFlow.InputData
import TensorFlow.CoreTypes: DT_FLOAT32
import TensorFlow.API: relu, conv2d, softmax_cross_entropy_with_logits,
                       AdamOptimizer, arg_max, equal, cast


# Learning options
learning_rate = 0.001
training_epochs = 1
batch_size = 100
display_step = 1

mnist = read_data_sets("/tmp/data/", one_hot=true)

network_description = Dict(
    :input_size => [28, 28, 1],
    :conv_layers => [
        Dict(
	    :kernel => 5,
	    :stride => 2,
	    :filters => 32,
	    :activation => relu,
	),
        Dict(
	    :kernel => 5,
	    :stride => 2,
	    :filters => 64,
	    :activation => relu,
	),
    ],
    :fc_layers => [
        Dict(
	    :size => 1024,
	    :activation => relu,
	),
        Dict(
	    :size => 10,
	    :activation => identity,
	)
    ],
)


function conv_output_size(description)
    current_size = description[:input_size]

    for layer in description[:conv_layers]
	n_input_channels = current_size[3]
	height = current_size[1] / layer[:stride]
	width = current_size[2] / layer[:stride]
	current_size = [height, width, layer[:filters]]
    end

    return Int(prod(current_size))
end


# Construct parameter arrays
rand_weights(dimensions) =  Variable(randn(Tensor, dimensions))


function network_parameters(description)
    parameters = Dict()

    # parameters for conv layers
    parameters[:conv_layers] = []

    n_prev_channels = description[:input_size][3]

    for layer in description[:conv_layers]
        weights = rand_weights(Int[layer[:kernel], layer[:kernel],
	                           n_prev_channels, layer[:filters]])
        biases = rand_weights([layer[:filters]])

	layer_parameters = Dict(
	    :weights => weights,
	    :biases => biases,
	)

	# append parameters to array
	push!(parameters[:conv_layers], layer_parameters)

        n_prev_channels = layer[:filters]
    end

    # parameters for fully connected layers
    parameters[:fc_layers] = []

    n_prev_outputs = conv_output_size(description)

    for layer in description[:fc_layers]
        weights = Variable(randn(Tensor, Int[n_prev_outputs, layer[:size]]))
        biases = Variable(randn(Tensor, [layer[:size]]))

	layer_parameters = Dict(
	    :weights => weights,
	    :biases => biases,
	)

	# append parameters to array
	push!(parameters[:fc_layers], layer_parameters)

	n_prev_outputs = layer[:size]
    end

    return parameters
end


# Returns a node in a tensorflow graph corresponging to the network output
function network_output(input, description, parameters)
    # TODO could have a version of this function where if you don't pass parameters,
    # then the function creates and randomly initialises parameters

    reshaped_input = reshape(input, [-1; description[:input_size]])
    current_output = reshaped_input

    for (l, p) in zip(description[:conv_layers], parameters[:conv_layers])
	current_output = conv2d(current_output, p[:weights],
	                        [1, l[:stride], l[:stride], 1],
			        "SAME") + p[:biases]
	current_output = l[:activation](current_output)
    end

    current_output_size = conv_output_size(description)
    current_output = reshape(current_output, [-1; current_output_size])

    # parameters for fully connected layers
    for (l, p) in zip(description[:fc_layers], parameters[:fc_layers])
	current_output = current_output * p[:weights] + p[:biases]
	current_output = l[:activation](current_output)
    end

    return current_output
end


# tf graph inputs
x = Placeholder(DT_FLOAT32, [-1, 784])
y = Placeholder(DT_FLOAT32, [-1, 10])

# construct model
parameters = network_parameters(network_description)
pred = network_output(x, network_description, parameters)

# Define loss and optimizer
cost = mean(softmax_cross_entropy_with_logits(pred, y))
optimizer = minimize(AdamOptimizer(learning_rate), cost)

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
