using TensorFlow
using TensorFlow.API
import TensorFlow.API: AdamOptimizer

v = Variable(constant(1f0))
cost = sum(v)
optimizer = minimize(AdamOptimizer(0.001), cost)

s = Session()

run(s, initialize_all_variables())
run(s, optimizer)

close(s)

