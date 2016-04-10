# This script demonstrates how we can use polymorphism with TensorFlow, in
# particular, how we can define a function, such that when we pass it regular
# Julia arrays, we get regular Julia arrays as output and when we pass it
# TensorFlow Tensors, it gives us back a computational graph computing the
# equivalent function


using Base.Test

using TensorFlow
using TensorFlow.Train
using TensorFlow.InputData
import TensorFlow.API: relu, softmax_cross_entropy_with_logits, AdamOptimizer


################################################################################
#                               Define computation
################################################################################
# The computation here is generic, it can be applied to standard Julia arrays
# or to TensorFlow types
function multilayer_perceptron(X, W1, b1, W2, b2, W3, b3)
  h1 = relu(W1 * X .+ b1)
  h2 = relu(W2 * h1 .+ b2)
  output = W3 * h2 .+ b3
  return output
end


################################################################################
#                 Apply Computation to Standard Julia Arrays
################################################################################
batch_size = 100
n_inputs  = 784
n_hidden1 = 256 
n_hidden2 = 256
n_classes = 10

# The variables we are passing in for the computation here are standard Julia
# Array types
X = randn(n_inputs, batch_size)
W1, b1 = randn(n_hidden1, n_inputs), randn(n_hidden1, 1)
W2, b2 = randn(n_hidden2, n_hidden1), randn(n_hidden2, 1)
W3, b3 = randn(n_classes, n_hidden2), randn(n_classes, 1)
y = rand(1:10, batch_size)
y_one_hot = zeros(n_classes, batch_size)

for i in 1:batch_size
  y_one_hot[y, i] = 1.
end


# We need to define some of the functions we used in the definition of
# `multilayer_perceptron` for AbstractArray types
relu(X::AbstractArray) = max(0, X)

function safe_logsumexp(x)
  u = maximum(x, 1)
  return log(sum(exp(x .- u))) .+ u
end


function softmax_cross_entropy_with_logits(pred::AbstractArray,
                                           y_one_hot::AbstractArray)
  Z = safe_logsumexp(pred) 
  return sum(pred .* y_one_hot, 1) - Z
end


pred = multilayer_perceptron(X, W1, b1, W2, b2, W3, b3)
loss = mean(softmax_cross_entropy_with_logits(pred, y_one_hot))


# The result of the computation is of standard Julia Array types
println("\nResult of applying multilayer_perceptron to Float64 Array:")
@show typeof(pred)
@show loss

# We will keep this value for later, just to make sure we get approximately the
# same results as in the TensorFlow computation
pred_value_jl = pred


################################################################################
#                 Apply Computation to TensorFlow Tensors
################################################################################
# Convert all the variables to appropriate TensorFlow types
X = Placeholder(X)
W1, b1 = Variable(W1), Variable(b1)
W2, b2 = Variable(W2), Variable(b2)
W3, b3 = Variable(W3), Variable(b3)
y_one_hot = Placeholder(y_one_hot)

pred = multilayer_perceptron(X, W1, b1, W2, b2, W3, b3)
loss = mean(softmax_cross_entropy_with_logits(pred, y_one_hot))

# The result of the computation is a node in a TensorFlow computational graph
println("\nResult of applying multilayer_perceptron to TensorFlow types:")
@show typeof(pred)
@show loss


################################################################################
#                 Run the TensorFlow Computational Graph 
################################################################################
# Initializing the variables
init = initialize_all_variables()
sess = Session()

try
  run(sess, init)
  pred_value_tf = run(sess, pred)

  println("\nWe run the TensorFlow computational graph for `pred` and test")
  println("the result has approximately the same value as the value")
  println("we computed with Julia arrays:")

  # Check that the computation with TensorFlow types and Julia types give
  # approximately the same numerical result
  @show (pred_value_jl ≈ pred_value_tf)
finally
  close(sess)
end


println()

