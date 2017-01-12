#ifndef __PROFILING__
#define __PROFILING__

#include "caffe/internode/mpiutil.hpp"

//#define PROFILING

// PROFILING

#define PROFILE_BEGIN(name)      \
    LOG(INFO) << caffe::internode::mpi_get_current_proc_rank_as_string()    \
              << " PROFILING BEGIN[" << name << "]"

              
#define PROFILE_END(name)      \
    LOG(INFO) << caffe::internode::mpi_get_current_proc_rank_as_string()    \
              << " PROFILING END[" << name << "]"

#endif
