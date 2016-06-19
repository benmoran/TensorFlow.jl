module Models

module RNN
using PyCall
export LSTMCell

@pyimport tensorflow.python.ops.rnn_cell as rnn_cell

immutable LSTMCell
  x::PyObject
end
PyObject(o::LSTMCell) = o.x


end # RNN

end # Models
