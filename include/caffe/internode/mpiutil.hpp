#ifndef CAFFE_UTIL_MPIUTIL_H_
#define CAFFE_UTIL_MPIUTIL_H_

#include <map>
#include <string>

// Modified by Jian
#include <mpi.h>

// #ifndef MSG_ASYNC
// #define MSG_ASYNC 

// Modified by Jian
// Compile time mapping from typename Dtype to MPI_Datatype
template <typename Dtype>
MPI_Datatype DtypeToMPIDtype();


namespace caffe {
namespace internode {

// Modified by Jian
extern int nGroup;
extern int nServer;

int mpi_get_current_proc_rank();
std::string mpi_get_current_proc_rank_as_string();
int mpi_get_comm_size();
std::string mpi_get_current_proc_name();
std::string mpi_get_error_string(int errorcode);

void mpi_init(int argc, char** argv);
void mpi_finalize();

int LayerIdToServerRank(int n_layer, int layer_id);

}  // namespace internode
}  // namespace caffe

#endif   // CAFFE_UTIL_MPIUTIL_H_

