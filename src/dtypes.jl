# N.B. This file is maintained in TensorFlowBuilder source
export dtypesym

const DTYPE_SYMS = Symbol[:float32,
                          :float64,
                          :bfloat16,
                          :complex64,
                          :int8,
                          :uint8,
                          :int32,
                          :int64,
                          :bool,
                          :string,
                          :qint8,
                          :quint8,
                          :qint16,:quint16,:qint32,]
dtypesym(s::Symbol) = symbol("DT_$(uppercase(string(s)))")
dtypesym(o::PyObject) = dtypesym(symbol(o.x[:name]))

const DTYPES = Dict{PyObject,AbstractString}()

for dtype in DTYPE_SYMS
    sym = dtypesym(dtype)
    symsym = string(sym)
    if dtype in names(tf)
        @eval begin
            const $sym = Dtype(tf.$(dtype))
            export $sym
            DTYPES[$sym.x] = $symsym
        end
    end
end
