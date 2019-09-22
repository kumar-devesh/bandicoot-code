#!/bin/bash
#
# Generate the headers for each benchmark csv output.
#
# For each benchmark we'll have
#
# task, device, backend, elem_type, rows, cols, trial, time

# temporary
device="rtx2080ti"
trials=5;

echo "task, device, backend, elem_type, rows, cols, trial, time" > results.csv;

# square benchmarks
for r in 100 300 1000 3000 10000 30000
do
  ./matmul $device $trials $r $r results.csv;
done

# non-square benchmarks
for r in 100 1000 10000 100000 1000000
do
  ./matmul $device $trials 100 $r results.csv;
done

# square matrix * vector
for r in 100 3000 10000 30000
do
  ./matvec $device $trials $r results.csv;
done

# cholesky decomposition
for r in 100 300 1000 3000 10000 30000
do
  ./chol $device $trials $r results.csv;
done

# accu()
for r in 100 1000 10000 100000 1000000 10000000 100000000 1000000000
do
  ./accu $device $trials $r results.csv;
done

# fill()
for r in 100 1000 10000 100000 1000000 10000000 100000000 1000000000
do
  ./fill $device $trials $r results.csv;
done

# copy matrix
for r in 100 300 1000 3000 10000 30000
do
  ./copy $device $trials $r $r results.csv;
done

# submatrix copy
for r in 100 300 1000 3000 10000 30000
do
  ./submat_copy $device $trials $r $r results.csv;
done
