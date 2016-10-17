// implementation: Jian Zhang
// Time Oct 2 2016

#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <boost/thread.hpp>
#include <cstdlib>
#include <mpi.h>

#include "caffe/caffe.hpp"
#include "caffe/MultiSolver.hpp"
#include "caffe/multinode/BlobInfo.hpp"
#include "caffe/internode/mpiutil.hpp"

namespace caffe {
namespace async_param_server {

using std::make_pair;

// TODO modify decoding strategy
// we use TAG = layer_id * 1e6 + blob_id * 1e4 + part_id to identify parts location
struct TaskRequest {
  int root_rank_;
  int layer_id_; 
  int blob_id_;
  int part_id_;
  MPI_Request mpi_request_;

  TaskRequest(int root_rank, int layer_id, int blob_id, int part_id) :
    root_rank_(root_rank), layer_id_(layer_id), blob_id_(blob_id), part_id_(part_id) {}
  void ParseInfo(int64_t mpi_rank, int64_t mpi_tag) {
    root_rank_ = mpi_rank;
    layer_id_ = mpi_tag / 1e6;
    mpi_tag -= layer_id_ * 1e6;
    blob_id_ = mpi_tag / 1e4;
    mpi_tag -= blob_id_ * 1e4;
    part_id_ = mpi_tag;
  }
  int GetTag() {
    return layer_id_ * 1e6 + blob_id_ * 1e4 + part_id_;
  }
};

struct TaskQueue {
  boost::mutex queue_mutex_;
  std::deque<TaskRequest> queue_;
};


// protocol:
// when get a non-blocking mpi receive, comm thread submit a job to the 
// update_tasks_ queue. 
// The compute thread will check the update_tasks_ queue. After it finishes
// update, the compute thread will submit request to send_tasks_ queue.
// In the communicate loop, the thead consider send task first, and then 
// process receive tasks.
template <typename Dtype>
class AsyncParamServer {
public:
  AsyncParamServer(boost::shared_ptr<Solver<Dtype> > solver);
  ~AsyncParamServer() {
    // setup the mpi buffers
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    for (int i = 0; i < caffe::internode::nGroup; i++) {
      int root_rank = mpi_size / caffe::internode::nGroup * i;
      for (int j = 0; j < solver_->net()->layers().size(); j++)
        for (int k = 0; k < solver_->net()->layers()[j]->blobs().size(); k++) {
          std::free(send_buf_[make_pair(root_rank, make_pair(j, k) ) ].first);
          std::free(recv_buf_[make_pair(root_rank, make_pair(j, k) ) ].first);
        }
    }
    delete[] update_tasks_;
  };
  // in the update task, the compute thread 
  // 0. lock the mutex on blob
  // 1. copy buffer to solvers diff buffer
  // 2. perform updates
  // 3. copy the model to the corresponding mpi buffer
  // 4. submit a send task
  // 5. unlock the mutex blob
  void ProcessUpdateTask();
  // in the Send task, we use non-blocking send for model parts going back to roots
  // We do not need to care about the request. Because if the blocking recv
  // has not finished on root, it will not start a new send task
  void ProcessSendTask();
  // We iterate over the recv_tasks_ vector, when the request is done, we start a
  // new corresponding MPI non-blocking recv call.
  void ProcessRecvTask();
  void ComputeLoop();
  void CommLoop();
  void Run();

  // some access function
  int IdToRootRank(int root_id) { 
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 
    return (mpi_size - 1) / caffe::internode::nGroup * root_id;
  }
  int RootRankToId(int root_rank) { 
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 
    return root_rank / ( (mpi_size - 1) / caffe::internode::nGroup);
  }

private:
  // for communication
  // // protector for blob on the solver
  // std::map<std::pair<int, int>, boost::mutex> solver_blob_mutex_;
  // std::vector<TaskQueue> update_tasks_;
  TaskQueue* update_tasks_;
  std::deque<TaskRequest> send_tasks_;
  boost::mutex send_queue_mutex_;
  int recv_tasks_iter_;
  std::vector<TaskRequest> recv_tasks_;
  std::map<std::pair<int, std::pair<int, int> >, int> rank_layer_blob_to_vec_pos;
  // root_rank, layer_id, blob_id
  std::map<std::pair<int, std::pair<int, int> >, std::pair<Dtype*, int64_t> > recv_buf_;
  std::map<std::pair<int, std::pair<int, int> >, std::pair<Dtype*, int64_t> > send_buf_;

  // for computation
  boost::shared_ptr<Solver<Dtype> > solver_;
  shared_ptr<BlobAccessor<Dtype> > blob_accessor_;
  shared_ptr<BlobConstInfo> const_info_;

  // for termination: count the number of operations 
  // needed in total
  int64_t send_cnt_;
  int64_t update_cnt_; 


  // iter for different blobs
  std::map<std::pair<int, int>, int64_t> async_iter_;

  // total number of blobs which need updates
  int n_blob_to_update_;
  // this is bundled with the update_tasks_ queue, they use the same
  // mutex lock
  int n_blob_current_root_;
  // this indicates the rank of the root current being updated.
  // it is the index of the root not the actual mpi rank
  int current_root_id_;
};

} // end of namespace async_param_server

} // end of namespace caffe

