# This optional Makefile can be useful when working with TensorFlowBuilder.

SRC_DIR := ../TensorFlowBuilder/generated/TensorFlow/src

all:	regenerate compare

regenerate:
	make -C ../TensorFlowBuilder generate

compare:
	diff -x \*~ -x \*Helper.jl --suppress-common-lines -r src/API ${SRC_DIR}/API 
	diff --suppress-common-lines src/CoreTypes.jl ${SRC_DIR}/CoreTypes.jl
	diff --suppress-common-lines src/API.jl ${SRC_DIR}/API.jl
	diff --suppress-common-lines src/types.jl ${SRC_DIR}/types.jl
	diff --suppress-common-lines src/dtypes.jl ${SRC_DIR}/dtypes.jl

install:
	make -C ../TensorFlowBuilder install
