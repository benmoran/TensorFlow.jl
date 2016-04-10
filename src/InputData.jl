module InputData
import ..TensorFlow: DT_FLOAT32
import PyCall

export DataSet, DataSets, num_examples, next_batch, read_data_sets, images, labels

# HACK - better way to find this Python file?
unshift!(PyCall.PyVector(PyCall.pyimport("sys")["path"]),
         joinpath(Pkg.dir(), "TensorFlow", "src"))
@PyCall.pyimport input_data


type DataSet
  x::PyCall.PyObject
end
num_examples(ds::DataSet) = ds.x[:num_examples]
images(ds::DataSet) = ds.x[:images]
labels(ds::DataSet) = ds.x[:labels]


type DataSets
  train::DataSet
  validation::DataSet
  test::DataSet
end

function next_batch{N<:Integer}(dataset::DataSet, batch_size::N, fake_data=false)
  return dataset.x[:next_batch](batch_size, fake_data)
end

function read_data_sets(train_dir::AbstractString;
                        fake_data=false, one_hot=false, dtype=DT_FLOAT32)
  pydatasets = input_data.read_data_sets(train_dir, fake_data, one_hot, dtype.x)
  return DataSets(DataSet(pydatasets[:train]),
                  DataSet(pydatasets[:validation]),
                  DataSet(pydatasets[:test]))
end

end # module
