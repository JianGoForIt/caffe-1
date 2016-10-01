#!/usr/bin/env sh

TOOLS=./build/tools

echo "starting data server"
OMP_NUM_THREADS=1 $TOOLS/caffe data_server \
    --solver=experiment/cifar10/cifar10_grid_solver_data_server.prototxt  \
    --listen_address=tcp://127.0.0.1:1234 &

sleep 3
echo "executing 4 nodes with mpirun"

OMP_NUM_THREADS=1 \
mpirun -v \
-hostfile experiment/cifar10/hosts.txt -n 5 \
$TOOLS/caffe train --solver=experiment/cifar10/cifar10_grid_solver_worker.prototxt \
--param_server_solver=experiment/cifar10/cifar10_grid_solver_async_param_server.prototxt \
--snapshot=experiment/cifar10/snapshot/warm_start/warm_start_iter_5000.solverstate.h5 \
--param_server=mpi --n_group=1

ls ./build
