using TensorFlow
using TensorFlow.API

v = Variable(randn(Tensor, [2, 3]))
@test isa(v, TensorFlow.AbstractTensor)

@test v.x in TensorFlow.API.trainable_variables()

@test isa(randn(Tensor, [1, 2]) * randn(Tensor, [2, 3]), Tensor)

ph = Placeholder(DT_FLOAT32, [2, 2])
@test isa(ph * v, TensorFlow.AbstractTensor)

s = TensorFlow.API.InteractiveSession()
x = run(s, randn(Tensor, [3,4]))
close(s)
@test isa(x, Array{Float32,2})

@test isa(ph * v, TensorFlow.AbstractTensor)

# TODO why is all_variables() empty when trainable_variables
#@test !isempty(TensorFlow.API.all_variables())
