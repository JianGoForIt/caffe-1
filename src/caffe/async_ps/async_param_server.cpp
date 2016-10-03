#include <cstdlib>
#include <climits>
#include "caffe/async_ps/async_param_server.hpp"
#include "caffe/internode/mpiutil.hpp"

namespace caffe {
namespace async_param_server {

using std::make_pair

template <typename Dtype>
AsyncParamServer<Dtype>::AsyncParamServer(boost::shared_ptr<Solver<Dtype> > solver) :
  solver_(boost::make_shared<MultiSolver<Dtype> >(solver) ),
  blob_accessor_(BlobInfoFactory<Dtype>::create_blob_accessor(solver) ),
  const_info(BlobInfoFactory<Dtype>::create_const_info(
    solver, std::LONG_MAX) ), recv_task_iter_(0), send_cnt_(0) {

  // setup the mpi buffers and recv task vector
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  for (int i = 0; i < caffe::internode::nGroup; i++) {
    int root_rank = mpi_size / nGroup * i;
    for (int j = 0; j < solver_->net()->layers().size; j++)
      for (int k = 0; k < solver->net()->layers()[j].size; k++) {
        int64_t blob_size = solver->net()->layers()[j][k].count();
        buf_ptr_.insert(make_pair(make_pair(root_rank, 
          make_pair(j, k) ), make_pair(NULL, 0) ) );
        buf_ptr_[make_pair(make_pair(root_rank, make_pair(j, k) ) ) ] = 
          make_pair( (Dtype*)std::malloc(sizeof(Dtype) * blob_size), blob_size);

        // TODO note here according to the intel impelmentation, there is only 1 part for each blob
        TaskRequest recv_task(root_rank, j, k, 0);
        recv_tasks_.push_back(recv_task);
      }
  }

  // setup solver blob mutex 
  for (int j = 0; j < solver_->net()->layers().size; j++)
    for (int k = 0; k < solver->net()->layers()[j].size; k++) {
      solver_blob_mutex_.insert(make_pair(make_pair(j, k), new boost::mutex) );
    }
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ProcessUpdateTask() {
  update_queue_mutex_.lock();
  if (!update_tasks_.empty() ) {
    TaskRequest task = update_tasks_.pop_front();
    update_queue_mutex_.unlock();
    // copy to diff in solver
    Blob<Dtype>* blob = blob_accessor_->get_blob(task.layer_id_, task.blob_id_);
    Dtype* solver_diff = blob->mutable_diff();
    Dtype* mpi_buf = buf_ptr_[].first;
    int64_t count = buf_ptr_[].second;
    std::assert(count == blob.count() );
    std::memcpy(solver_diff, mpi_buf, sizeof(Dtype) * count);
    // apply update
    int param_id = solver_->net()->get_layer_learnable_param_ids(task.layer_id_)[tasks.blob_id_];
    solver_->ApplyUpdate(param_id);
    solver_->net()->ClearParamDiffs(param_id);

    // copy model(data) in solver to mpi buffer
    Dtype* solver_data = blob->mutable_data();
    std::memcpy(mpi_buf, solver_data, sizeof(Dtype) * count);

    send_queue_mutex_.lock();
    send_tasks_.push_back(task);
    send_queue_mutex_.unlock();
  }
  else
    update_queue_mutex_.unlock();
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ProcessSendTask() {
  send_queue_mutex_.lock();
  if (!send_queue_mutex_.empty() ) {
    int root_rank = send_tasks_.front().root_rank_;
    int layer_id = send_tasks_.front().layer_id_;
    int blob_id = send_tasks_.front().blob_id_;
    int tag = send_tasks_.front().GetTag();
    send_tasks.pop_front();
    // unlock to permit insert send task on the other thread
    send_queue_mutex_.unlock();
    std::pair<Dtype*, int64_t> buf = 
      buf_ptr_[make_pair(root_rank, make_pair(layer_id, blob_id) ) ];
    Dtype* ptr = buf.first;
    int count = buf.second;
    // We do not need to care about the request. Because if the blocking recv
    // has not finished on root, it will not start a new send task
    MPI_Isend(ptr, count, DtypeToMPIDtype<Dtype>::type, root_rank, 
      tag, MPI_COMM_WORLD, MPI_REQUEST_NULL);
    // start a new listening to wait for message from roots
    MPI_Irecv(ptr, count, DtypeToMPIDtype<Dtype>::type, root_rank,
      tag, MPI_COMM_WORLD, MPI_REQUEST_NULL);
  }
  else
    send_queue_mutex_.unlock();
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ProcessRecvTask() {
  int flag = 0;
  MPI_Request request = recv_tasks_[recv_tasks_iter_].mpi_request;
  MPI_Test(MPI_Request, &flags, MPI_STATUS_IGNORE);
  while(flag == 0) {
    MPI_Test(MPI_Request, &flags, MPI_STATUS_IGNORE);
    recv_tasks_iter_ = (recv_tasks_iter_ + 1) % recv_tasks_.size;
  }
  // TODO currently no need to lock the solver buffer
  update_queue_mutex_.lock();
  update_tasks_.push_back(recv_tasks_[recv_tasks_iter_] );
  update_queue_mutex_.unlock();
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ComputeLoop() {
  int64_t total_update =     
    nGroup * recv_tasks_.size * solver_->param().max_iter();
  do {
    ProcessUpdateTask();
  } while(update_cnt_ < total_update);

}


template <typename Dtype>
void AsyncParamServer<Dtype>::CommLoop() {
  int64_t total_send = 
    nGroup * recv_tasks_.size * solver_->param().max_iter();
  do {
    ProcessSendTask();
    ProcessRecvTask();
  } while(send_cnt_ < total_send);
}


template <typename Dtype>
void AsyncParamServer<Dtype>::Run() {
  // spawn compute thread
  std::thread compute_thread(&AsyncParamServer<Dtype>::ComputeLoop, this);
  // spawn communication thread
  std::thread comm_thread(&AsyncParamServer<Dtype>::CommLoop, this);

  compute_thread.join();
  comm_thread.join();
}


} // end of namespace async_param_server
 
} // end of namespace caffe