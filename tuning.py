#!/usr/bin/python
'''
Need the following in PYTHONPATH
[CAFFE]/python
[CAFFE]/python/caffe/proto

eg., by adding lines like this to ~/.bashrc
export PYTHONPATH=/home/migish/workspace/caffe-1/python:$PYTHONPATH
export PYTHONPATH=/home/migish/workspace/caffe-1/python/caffe/proto:$PYTHONPATH
'''

import caffe_pb2
import google.protobuf
import os
import time

### PARAMETERS ###

base_dir = 'experiment/cifar10'

experiment_name = 'tuning'

n_workers = 4

all_n_groups = [1, 2, 4]

all_base_lr = [0.0001, 0.001, 0.01] 
all_base_lr = [0.0001, 0.001] 

all_momentum = [0.0, 0.3, 0.6, 0.9]
all_momentum = [0.0, 0.9]

time_out = 60

group_batch_size = 100

##################



def prepare_experiment(experiment_name='tuning',
                       base_dir='experiment/cifar10', 
                       base_lr=None, momentum=None,
                       snapshot_prefix=None, random_seed=None,
                       max_iter=None, batch_size=None,
                       n_workers=4, n_groups=1):
    
    n_nodes = n_workers + 1
    
    solver_file_name='cifar10_grid_solver_async_param_server.prototxt'
    input_solver_file_name = 'template_'+solver_file_name
    create_solver_file(input_solver_file_name, output_solver_file_name=solver_file_name, base_dir=base_dir, 
                       base_lr=base_lr, momentum=momentum,
                       snapshot_prefix=snapshot_prefix, random_seed=random_seed,
                       max_iter=max_iter)

    solver_file_name='cifar10_grid_solver_worker.prototxt'
    input_solver_file_name = 'template_'+solver_file_name
    create_solver_file(solver_file_name, output_solver_file_name=solver_file_name, base_dir=base_dir,
                       base_lr=base_lr, momentum=momentum,
                       snapshot_prefix=snapshot_prefix, random_seed=random_seed,
                       max_iter=max_iter)


    network_file_name = 'cifar10_grid_train_test_data_server.prototxt'
    input_network_file_name = 'template_'+network_file_name
    create_network_file(input_network_file_name, output_network_file_name=network_file_name, base_dir=base_dir,
                        batch_size=batch_size) 

    network_file_name = 'cifar10_grid_train_test_async_param_server.prototxt'
    input_network_file_name = 'template_'+network_file_name
    create_network_file(input_network_file_name, output_network_file_name=network_file_name, base_dir=base_dir,
                        batch_size=batch_size) 

    network_file_name = 'cifar10_grid_train_test_worker.prototxt'
    input_network_file_name = 'template_'+network_file_name
    create_train_test_worker_file(input_network_file_name, output_network_file_name=network_file_name, base_dir=base_dir,
                        batch_size=batch_size) 

    script_file_name = 'train_cifar10_grid.sh'
    input_script_file_name = 'template_'+script_file_name

    sed_call = ('''sed -e 's/\-n [0-9]*/-n ''' + str(n_nodes) 
                + '''/g' ''' + base_dir + '/' + input_script_file_name + ''' | \
                sed -e 's/\-\-n_group=[0-9]*/--n_group=''' + str(n_groups) + '''/g' > ''' + base_dir + '/' + script_file_name)

    if os.system(sed_call):
        raise "Sed call for training script failed"
        
    log_file_name = (
             'W=' + str(n_workers)
             + '_G=' + str(n_groups)
             + '_LR=' + str(base_lr)
            )
    if not (momentum is None):
        log_file_name += '_mu=' + str(momentum)
        
    log_file_name += '.log'

    os.system('mkdir -p ' + base_dir + '/log/' + experiment_name)

    log_dir = base_dir + '/log/' + experiment_name

    os.system('touch '+ log_dir+'/'+log_file_name)
    os.system('rm current.log')
    os.system('ln -s '+ log_dir+'/'+log_file_name + ' current.log')
    print 'Saving log in: ' + log_dir + '/' + log_file_name
    
    run_command = 'bash '+ base_dir+'/'+'train_cifar10_grid.sh 2> ' + log_dir+'/'+log_file_name
    
    return run_command

def run_experiment(run_command, time_out=20):
    if os.system(run_command):
        print "Experiment run command failed"
        return

    time.sleep(time_out)

    os.system('killall caffe')


def load_solver_parameters(solver_file_name, base_dir='.'):
    full_solver_file_name = base_dir+'/'+solver_file_name
    solver = caffe_pb2.SolverParameter()
    with open(full_solver_file_name, 'r') as f:       
        google.protobuf.text_format.Merge(str(f.read()), solver)
    return solver


def save_solver_parameters(solver, output_solver_file_name='temp_solver.prototxt', base_dir='.'):
    full_output_solver_file_name = base_dir + '/' + output_solver_file_name

    with open(full_output_solver_file_name, 'w') as f:
        f.write(str(solver))

def create_solver_file(input_solver_file_name, output_solver_file_name='temp_solver.prototxt', base_dir='.',
                       base_lr=None, momentum=None,
                       snapshot_prefix=None, random_seed=None, max_iter=None):
    solver = load_solver_parameters(input_solver_file_name, base_dir)
    
    if not (base_lr is None):
        solver.base_lr = base_lr
        
    if not (momentum is None):
        solver.momentum = momentum

    if not (snapshot_prefix is None):
        solver.snapshot_prefix = snapshot_prefix

    if not (random_seed is None):
        solver.random_seed = random_seed
    
    if not (max_iter is None):
        solver.max_iter = max_iter
    
    save_solver_parameters(solver, output_solver_file_name, base_dir)


def load_network_parameters(network_file_name, base_dir='.'):
    full_network_file_name = base_dir+'/'+network_file_name
    network = caffe_pb2.NetParameter()
    with open(full_network_file_name, 'r') as f:
        google.protobuf.text_format.Merge(str(f.read()), network)
        return network

def save_network_parameters(network, output_network_file_name='temp_network.prototxt', base_dir='.'):
    full_output_network_file_name = base_dir+'/'+output_network_file_name

    with open(full_output_network_file_name, 'w') as f:
        f.write(str(network))

def create_network_file(input_network_file_name, output_network_file_name='temp_network.prototxt', base_dir='.',
                      batch_size=None):
    network = load_network_parameters(input_network_file_name, base_dir)

    if not (batch_size is None):
        network.layer[0].data_param.batch_size = batch_size

    save_network_parameters(network, output_network_file_name, base_dir)

def create_train_test_worker_file(input_network_file_name, output_network_file_name='temp_network.prototxt', base_dir='.',
                     batch_size=None):
    network = load_network_parameters(input_network_file_name, base_dir)

    if not (batch_size is None):
        assert(network.layer[0].type=="RemoteData")
        assert(network.layer[0].include[0].phase==0)
        network.layer[0].remote_data_param.shape[0].dim[0] = batch_size
        network.layer[0].remote_data_param.shape[1].dim[0] = batch_size

    save_network_parameters(network, output_network_file_name, base_dir)


if __name__ == "__main__":
    exp_name_date = experiment_name + '_' + time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    for base_lr in all_base_lr:
        for n_groups in all_n_groups:
            worker_batch_size =  n_groups * (group_batch_size / n_workers)
            for momentum in all_momentum:
                print 
                print ( 'Running experiment: Workers='+str(n_workers) 
                        + '\tGroups=' + str(n_groups) 
                        + '\tBatch=' + str(worker_batch_size) 
                        + '\tLR=' + str(base_lr)
                        + '\tmu=' + str(momentum)
                )
                run_command = prepare_experiment(experiment_name=exp_name_date,
                                                base_dir=base_dir,
                                                base_lr=base_lr,
                                                momentum=momentum,
                                                n_workers=n_workers,
                                                n_groups=n_groups,
                                                batch_size=worker_batch_size
                )

                run_experiment(run_command, time_out=time_out)



