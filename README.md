# TensorFlow

Wraps the TensorFlow Python library in Julia, via the `PyCall` package.


This is done in two stages:

- The low-level interface in the modules underneath `TensorFlow.API` aims to provide a faithful, direct mapping to the original Python functions, merely adding some Julia type annotations to the function declarations.

- The `TensorFlow.Idiomatic` module then implements methods for standard Julia base functions using the `API` interface, so that manipulating `Tensor` objects and others can be done in the same way as working with ordinary Array objects.

Note that functions which are passed `Tensor` objects will not directly return the results, but will instead return TensorFlow graphs that can be run later in a `Session`.  See the `examples` directory, especially `examples/polymorphism.jl`.


## Additional notes for package developers

The [`TensorFlowBuilder`](https://github.com/benmoran/TensorFlowBuilder.jl) package contains code to generate the Julia source in the `TensorFlow.API` module by introspecting the Python package.  It is not necessary to use this pacakge, but could be helpful to correct bugs in the wrapper, or to extend it further.
