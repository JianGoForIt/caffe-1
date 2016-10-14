#!/usr/bin/env sh

TOOLS=./build/tools

echo "starting data server"
OMP_NUM_THREADS=1 $TOOLS/caffe data_server \
    --solver=examples/cifar10/cifar10_full_solver_data_server.prototxt  \
    --listen_address=tcp://127.0.0.1:2341 &

sleep 3
echo "executing 4 nodes with mpirun"

OMP_NUM_THREADS=1 \
mpirun -v \
-hostfile examples/cifar10/hosts.txt -n 5 \
$TOOLS/caffe train --solver=examples/cifar10/cifar10_full_solver_sync_param_server.prototxt \
--param_server_solver=examples/cifar10/cifar10_full_solver_dummy.prototxt \
--snapshot=examples/cifar10/warm_start_iter_5000.solverstate.h5 \
--param_server=mpi --n_group=1

ls ./build
