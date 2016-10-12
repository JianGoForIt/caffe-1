#include <cstdlib>
#include <climits>
#include <boost/make_shared.hpp>
#include <thread>

#include "caffe/caffe.hpp"
#include "caffe/async_ps/async_param_server.hpp"
#include "caffe/internode/mpiutil.hpp"

namespace caffe {
namespace async_param_server {

using std::make_pair;

template <typename Dtype>
AsyncParamServer<Dtype>::AsyncParamServer(boost::shared_ptr<Solver<Dtype> > solver) :
  recv_tasks_iter_(0), 
  solver_(solver),
  blob_accessor_(BlobInfoFactory<Dtype>::create_blob_accessor(solver) ),
  const_info_(BlobInfoFactory<Dtype>::create_const_info(solver, LONG_MAX) ), 
  send_cnt_(0), update_cnt_(0) {

  // setup the mpi buffers and recv task vector
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  for (int i = 0; i < caffe::internode::nGroup; i++) {
    int root_rank = mpi_size / caffe::internode::nGroup * i;
    for (int j = 0; j < solver_->net()->layers().size(); j++)
      for (int k = 0; k < solver_->net()->layers()[j]->blobs().size(); k++) {
        int64_t blob_size = solver_->net()->layers()[j]->blobs()[k]->count();
        // buf_ptr_.insert(make_pair(make_pair(root_rank, 
        //   make_pair(j, k) ), make_pair(NULL, 0) ) );
        Dtype* buf = (Dtype*)std::malloc(sizeof(Dtype) * blob_size);
        recv_buf_[make_pair(root_rank, make_pair(j, k) ) ] = 
          make_pair(buf, blob_size);

        // setup buf for recv
        // TODO note here according to the intel impelmentation, 
        // there is only 1 part for each blob.
        // Start listening before the task is insert into the queue
        TaskRequest recv_task(root_rank, j, k, 0);
        recv_tasks_.push_back(recv_task);
        rank_layer_blob_to_vec_pos[make_pair(root_rank, make_pair(j, k) ) ] = 
          recv_tasks_.size() - 1;
        // start a new listening to wait for message from roots
        MPI_Irecv(buf, blob_size, DtypeToMPIDtype<Dtype>(), root_rank,
          recv_task.GetTag(), MPI_COMM_WORLD, &(recv_tasks_[recv_tasks_.size() - 1].mpi_request_) );
      
        // Setup buf for send
        buf = (Dtype*)std::malloc(sizeof(Dtype) * blob_size);
        send_buf_[make_pair(root_rank, make_pair(j, k) ) ] = 
          make_pair(buf, blob_size);

        // setup iter
        async_iter_[make_pair(i, j) ] = solver_->iter();
      }
  }
}


// TODO Jian how to get the correct iter number potentially get the version and set iter before update
template <typename Dtype>
void AsyncParamServer<Dtype>::ProcessUpdateTask() {
  std::deque<TaskRequest> to_update;
  update_queue_mutex_.lock();
  to_update.swap(update_tasks_);
  update_queue_mutex_.unlock();
  while (!to_update.empty() ) {
    TaskRequest task = to_update.front();
    to_update.pop_front();

    // copy to diff in solver
    Blob<Dtype>* blob = blob_accessor_->get_blob(task.layer_id_, task.blob_id_);
    Dtype* solver_diff = blob->mutable_cpu_diff();
    Dtype* mpi_buf = 
      recv_buf_[make_pair(task.root_rank_, make_pair(task.layer_id_, task.blob_id_) ) ].first;
    int64_t count = 
      recv_buf_[make_pair(task.root_rank_, make_pair(task.layer_id_, task.blob_id_) ) ].second;
    assert(count == blob->count() );
    std::memcpy(solver_diff, mpi_buf, sizeof(Dtype) * count);
    // apply update
    int blob_wise_iter = async_iter_[make_pair(task.layer_id_, task.blob_id_) ];

    solver_->set_iter(blob_wise_iter);
    int param_id = solver_->net()->get_layer_learnable_param_ids(task.layer_id_)[task.blob_id_];
    
    // if (task.layer_id_ == 12 && task.blob_id_ == 0) {
    //   // DEBUG
    //   LOG(INFO) << "blob 12 0 check Learning rate " << solver_->GetLearningRate();
    //   while(1);
    // }

    solver_->ApplyUpdate(param_id);


    // // DEBUG
    // LOG(INFO) << "UPDATE root rank " << task.root_rank_ << " layer " << task.layer_id_ << " blob " << task.blob_id_;

    solver_->net()->ClearParamDiffs(param_id);
    async_iter_[make_pair(task.layer_id_, task.blob_id_) ] += 1;
    update_cnt_ += 1;

    // copy model(data) in solver to mpi buffer
    mpi_buf = send_buf_[make_pair(task.root_rank_, make_pair(task.layer_id_, task.blob_id_) ) ].first;
    Dtype* solver_data = blob->mutable_cpu_data();
    std::memcpy(mpi_buf, solver_data, sizeof(Dtype) * count);

    send_queue_mutex_.lock();
    send_tasks_.push_back(task);
    send_queue_mutex_.unlock();

    // DEBUG
    LOG(INFO) << " push send task for " << task.root_rank_ 
      << " " << task.layer_id_ << " " << task.blob_id_;
  }
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ProcessSendTask() {
  std::deque<TaskRequest> to_send;
  send_queue_mutex_.lock();
  to_send.swap(send_tasks_);
  send_queue_mutex_.unlock();
  while (!to_send.empty() ) {
    int root_rank = to_send.front().root_rank_;
    int layer_id = to_send.front().layer_id_;
    int blob_id = to_send.front().blob_id_;
    int tag = to_send.front().GetTag();
    to_send.pop_front();

    // DEBUG
    LOG(INFO) << " launched send task for " << root_rank << " " << layer_id << " " << blob_id;

    std::pair<Dtype*, int64_t> buf = 
      send_buf_[make_pair(root_rank, make_pair(layer_id, blob_id) ) ];
    Dtype* ptr = buf.first;
    int count = buf.second;
    // We do not need to care about the request. Because if the blocking recv
    // has not finished on root, it will not start a new send task
    MPI_Request dump_request;

    MPI_Isend(ptr, count, DtypeToMPIDtype<Dtype>(), root_rank, 
      tag, MPI_COMM_WORLD, &dump_request);
    send_cnt_ += 1;
    // start a new listening to wait for message from roots
    ptr = recv_buf_[make_pair(root_rank, make_pair(layer_id, blob_id) ) ].first;
    int vec_pos = rank_layer_blob_to_vec_pos[make_pair(root_rank, make_pair(layer_id, blob_id) ) ];
    MPI_Irecv(ptr, count, DtypeToMPIDtype<Dtype>(), root_rank,
      tag, MPI_COMM_WORLD, &(recv_tasks_[vec_pos].mpi_request_) );

  }
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ProcessRecvTask() {
  int flag = 0;
  for (int i = 0; i < recv_tasks_.size(); i++) {
    if (recv_tasks_[recv_tasks_iter_].mpi_request_ != MPI_REQUEST_NULL) {
      MPI_Test(&(recv_tasks_[recv_tasks_iter_].mpi_request_), &flag, MPI_STATUS_IGNORE);
      if (flag) {
        // currently no need to lock the solver buffer, as comp thread
        // takes care of two copy operations.
        update_queue_mutex_.lock();
        update_tasks_.push_back(recv_tasks_[recv_tasks_iter_] );
        update_queue_mutex_.unlock();
      }
    }
    recv_tasks_iter_ = (recv_tasks_iter_ + 1) % recv_tasks_.size();
    if (flag)
      break;
  }
}


template <typename Dtype>
void AsyncParamServer<Dtype>::ComputeLoop() {
  int64_t total_update =     
    caffe::internode::nGroup * recv_tasks_.size() * solver_->param().max_iter();
  do {
    ProcessUpdateTask();
  } while(update_cnt_ < total_update);

}


template <typename Dtype>
void AsyncParamServer<Dtype>::CommLoop() {
  int64_t total_send = 
    caffe::internode::nGroup * recv_tasks_.size() * solver_->param().max_iter();
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
  // std::thread comm_thread(&AsyncParamServer<Dtype>::CommLoop, this);
  CommLoop();

  compute_thread.join();
  // comm_thread.join();
}


template class AsyncParamServer<float>;
template class AsyncParamServer<double>;


} // end of namespace async_param_server
 
} // end of namespace caffe