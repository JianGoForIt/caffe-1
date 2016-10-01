#!/usr/bin/env sh
# 
# All modification made by Intel Corporation: © 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

TOOLS=./build/tools

# <<<<<<< 37c817036ccffc64123c1ff02f612e06f63f3a1e
echo "executing 4 nodes with mpirun"

OMP_NUM_THREADS=1 \
mpirun -l -host 127.0.0.1 -n 4 \
$TOOLS/caffe train --solver=examples/cifar10/cifar10_full_solver.prototxt --param_server=mpi
#=======
#echo "starting data server"
#OMP_NUM_THREADS=1 $TOOLS/caffe data_server \
#    --solver=examples/cifar10/cifar10_full_solver_data_server.prototxt  \
#    --listen_address=tcp://127.0.0.1:2341 &
#
#sleep 3
#echo "executing 4 nodes with mpirun"
#
#OMP_NUM_THREADS=1 \
#mpirun -v \
#-hostfile examples/cifar10/hosts.txt -n 5 \
#$TOOLS/caffe train --solver=examples/cifar10/cifar10_full_solver_sync_param_server.prototxt \
#--param_server_solver=examples/cifar10/cifar10_full_solver_dummy.prototxt \
#--snapshot=examples/cifar10/warm_start_iter_5000.solverstate.h5 \
#--param_server=mpi --n_group=1
#
#ls ./build
#>>>>>>> First commit for single server asynchronous architecture
