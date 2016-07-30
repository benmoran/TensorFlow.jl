using PyCall

using TensorFlow
import TensorFlow.API: relu, conv2d
import TensorFlow: DT_FLOAT32

I = constant(eye(2))

println(I)
println(Tensor)

@test isa(I, Tensor)
@test isa(I * I, Tensor)
@test isa(I + I, Tensor)
@test isa(I - I, Tensor)
@test isa(size(I), Tuple)

@test isa(relu(I), Tensor)

ph1 = Placeholder(DT_FLOAT32, [2, 2])
@test isa(ph1, AbstractTensor)

ph2 = Placeholder(DT_FLOAT32, [-1, 2])
@test isa(ph2, AbstractTensor)

x = reshape(I, [1, 4])
@test isa(x, AbstractTensor)

x = reshape(I, [1, -1])
@test isa(x, AbstractTensor)


# TODO ValueError('Shapes (4, 4) and (?, ?, ?, ?) must have the same rank',)
#conv2d(constant(eye(4)), constant(eye(4)), [1,1,1,1], "SAME")


d = Dict(ph1 => 1)
