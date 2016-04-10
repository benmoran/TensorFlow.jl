# TensorFlow

Wraps the TensorFlow Python library in Julia, via the `PyCall` package.

The low-level interface in the modules underneath `TensorFlow.API` aims to provide a faithful, direct mapping to the original Python functions, merely adding some Julia type annotations to the function declarations.

The `TensorFlow.Idiomatic` module then implements methods for standard Julia base functions using the `API` interface, so that manipulating `Tensor` objects and others can be done in the same way as working with ordinary Array objects.
