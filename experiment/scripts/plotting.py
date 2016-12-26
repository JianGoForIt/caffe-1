import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import re
import datetime

from os import listdir

# PLOTTING

COLOR_SYNC='#255FA7'
COLOR_ASYNC='#E66826'

LOSS_INDEX = 3

# PROCESSING

def string_from_dict(dct, dct_minus={}):
    return ' '.join([p+'='+str(dct[p]) for p in list(set(dct.keys())-set(dct_minus.keys())) ]
                    )


def plot_matching_configs(all_configs, target_config, W=10, group_color=False):
    all_configs = get_matching_configs(all_configs,target_config)
    for config in all_configs:
        G = int(config[0]['G'])
        n_workers = int(config[0]['W'])
        losses_per_batch = int(np.round(float(n_workers)/G))

        if len(config[1]) < 2*W*losses_per_batch:
            print config[0], ' run too short. Was is terminated?'
            continue

        all_seconds, all_losses = get_times_losses(config[1])

        if group_color:
            plt.plot(all_seconds, moving_average(all_losses,window_size=W*losses_per_batch),
                 '-',linewidth=2, label=str(config[0]), 
                 color=cm.hot((config[0]['G']+1.0)/float(n_workers+2)) )
        else:
            plt.plot(all_seconds, moving_average(all_losses,window_size=W*losses_per_batch),
                 '-',linewidth=2, label=' '.join([p+'='+str(config[0][p]) for p in list(set(config[0].keys())-set(target_config.keys())) ] ) 
            )

    plt.grid()
    #plt.axis([None, None, 0.7, 1.0]);
    if group_color:
        plt.title('Darker colors = Smaller number of groups (G)')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Smoothed loss')
    plt.xlabel('Time since snapshot')

def get_matching_configs(all_configs, target_config):
    matching_configs = []
    for this_ac in all_configs:
        config = this_ac[0]
        match = True
        for key in config.keys():
            if key in target_config and config[key]!=target_config[key]:
                match = False
        if match:
            matching_configs.append(this_ac)
    return matching_configs

def get_best_config_per_group(n_workers, all_configs, W=100):
    best_configs = {}
    best_losses = {}
    for config in all_configs:
        G = int(config[0]['G'])
        losses_per_batch = int(np.round(float(n_workers)/G))
        Weff = W*losses_per_batch
        losses = [line[LOSS_INDEX] for line in config[1]]

        if len(losses) < 2*Weff:
            print config[0], ' run too short. Was is terminated?'
            continue
            
        achieved_loss = min(moving_average(losses,Weff)[Weff:-Weff])
        if G not in best_configs.keys():
            best_configs[G] = [config[0].copy(), config[1]] 
            best_losses[G] = achieved_loss
        else:
            if achieved_loss < best_losses[G]:
                best_configs[G] = [config[0].copy(), config[1]] 
                best_losses[G] = achieved_loss

    return best_configs

def moving_average(signal, window_size=50):
    window = np.ones(int(window_size))/float(window_size)
    conv = np.convolve(signal, window, 'same')
    # conv = np.convolve(conv, window, 'same')
    conv[:window_size] = np.nan
    conv[-window_size:] = np.nan
    return conv
    

# EXTRACTING

def get_times_losses(all_lines):
    dt0 = all_lines[0][0]
    all_seconds = [(line[0]-dt0).total_seconds() % 86400 for line in all_lines]
    all_losses = [line[3] for line in all_lines]
    return all_seconds, all_losses

# LOADING

def load_results(log_file_name):
    all_lines = []

    with open(log_file_name, "r") as f:
        for line in f.readlines():
            if re.match('(.*)solver\.cpp(.*)Iteration(.*)loss(.*)', line):
                try:
                    (
                     _,time_field,pid,source_line,worker_id,check_iteration,
                     niter,check_loss,check_equals,loss
                    ) =line.split()
                except ValueError:
                    print line
                    continue
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

def load_profiling_info(log_file_name):
    all_events = []

    with open(log_file_name, "r") as f:
        for line in f.readlines():
            parsed_dict = parse_profile_log_line(line)
            if parsed_dict:
                all_events.append(parsed_dict)
                
    return sorted(all_events,key=(lambda elmt: elmt['time']))

def load_loss_info(log_file_name):
    all_losses = []

    with open(log_file_name, "r") as f:
        for line in f.readlines():
            parsed_dict = parse_loss_log_line(line)
            if parsed_dict:
                all_losses.append(parsed_dict)
                
    return all_losses

def parse_profile_log_line(line):
    parsed_dict = parse_generic_log_line(line)

    if not parsed_dict:
        return parsed_dict
    
    # Parse profiling line
    m=re.search('\[(?P<worker_id>[0-9]*)\] PROFILING (?P<prof_event>(BEGIN)|(END))\[(?P<prof_name>[^\]]*)\]', line)
    #m=re.search('\[(?P<worker_id>[0-9]*)\] PROFILING (?P<prof_event>(BEGIN)|(END))\[(?P<prof_name>[^\]]*)\] Iteration (?P<iteration>[0-9]*)', line)
    #m=re.search('PROFILING (?P<prof_event>(BEGIN)|(END))', line)
    #m=re.search('PROFILING', line)
    if m:
        parsed_dict['worker_id']=int(m.groupdict()['worker_id'])
        parsed_dict['prof_event']=m.groupdict()['prof_event']
        parsed_dict['prof_name']=m.groupdict()['prof_name']
        #parsed_dict['iteration']=int(m.groupdict()['iteration'])
        return parsed_dict
    else:
        return {}

def parse_loss_log_line(line):
    parsed_dict = parse_generic_log_line(line)
    
    if not parsed_dict:
        return parsed_dict
        
    # Parse loss line
#    m=re.search('\[(?P<worker_id>[0-9]*)\] Iteration (?P<iteration>[0-9]*), loss = (?P<loss>[0-9]*)', line)
    m=re.search('\[(?P<worker_id>[0-9]*)\] Iteration (?P<iteration>[0-9]*), loss = (?P<loss>[0-9e\-\.]*)$', line)
    
    if m:
        parsed_dict['worker_id']=int(m.groupdict()['worker_id'])
        parsed_dict['iteration']=int(m.groupdict()['iteration'])
        parsed_dict['loss']=float(m.groupdict()['loss'])
        return parsed_dict
    else:
        return {}

def parse_generic_log_line(line):
    parsed_dict = {}
    m=re.search('[IE](?P<date>[0-9]*) (?P<time>[^ ]*) (?P<pid>[0-9]*) (?P<source_line>[^\]]*)\] (?P<remainder>.*)', line) 
    if m:
        dt = datetime.datetime.strptime(m.groupdict()['date']+m.groupdict()['time'], "%m%d%H:%M:%S.%f")
        dt = dt.replace(year=datetime.datetime.now().year)
        parsed_dict['time']=dt
        parsed_dict['source_line']=m.groupdict()['source_line']
        parsed_dict['pid']=int(m.groupdict()['pid'])
        parsed_dict['remainder']=m.groupdict()['remainder']        

    return parsed_dict
