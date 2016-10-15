import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import re
import datetime

from os import listdir

# PLOTTING

COLOR_SYNC='#255FA7'
COLOR_ASYNC='#E66826'

# PROCESSING

def moving_average(signal, window_size=50):
    window = np.ones(int(window_size))/float(window_size)
    conv = np.convolve(signal, window, 'same')
    conv = np.convolve(conv, window, 'same')
    conv[:window_size] = np.nan
    conv[-window_size:] = np.nan
    return conv

# EXTRACTING

def get_times_losses(all_lines):
    dt0 = all_lines[0][0]
    all_seconds = [(line[0]-dt0).total_seconds() for line in all_lines]
    all_losses = [line[3] for line in all_lines]
    return all_seconds, all_losses

# LOADING

def load_results(log_file_name):
    all_lines = []

    with open(log_file_name, "r") as f:
        for line in f.readlines():
            if re.match('(.*)solver\.cpp(.*)Iteration(.*)loss(.*)', line):
                (
                 _,time_field,pid,source_line,worker_id,check_iteration,
                 niter,check_loss,check_equals,loss
                ) =line.split()
                assert(check_iteration=="Iteration")
                assert(check_loss=="loss")
                assert(check_equals=='=')
                loss=float(loss)
                niter=int(niter[:-1])
                worker_id = int(worker_id[1:-1])
                dt = datetime.datetime.strptime(time_field, "%H:%M:%S.%f")
                #print time, ' Worker=', worker_id, ' Iteration=', niter, ' Loss=',loss
                new_line = [dt, worker_id, niter, loss]
                all_lines.append(new_line)

    return all_lines

