#ifdef USE_MPI
#include <mpi.h>
#include "caffe/internode/mpiutil.hpp"
#endif

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/thread.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "caffe/internode/broadcast_callback.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/serialization/BlobCodec.hpp"


// Modified by Jian
#include <sys/types.h>
#include <unistd.h>



const int MSG_TAG = 1972;
// Message tag to terminate all processes.
// https://en.wikipedia.org/wiki/Seppuku
const int SEPPUKU_TAG = 0xDEAD;

namespace caffe {
namespace internode {

extern boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon>);

#ifdef USE_MPI

typedef boost::function<void(bool, int, int)> RequestCallback;

struct MpiRequest {
  boost::shared_ptr<MPI_Request> req;
  RequestCallback callback;
  bool ok;
  int sender;
  int size;
};

namespace {

}  // namespace

class MpiTreeClient : public TreeWaypoint {
  boost::shared_ptr<Daemon> daemon;
  std::vector<Handler*> handlers;
  std::vector<MpiRequest> requests;
  std::vector<MpiRequest> requests_to_process;
  std::vector<char> buffer;
  mutable boost::recursive_mutex mtx;
  mutable boost::optional<boost::thread::id> main_thread_id;
  mutable bool finished;

  void set_recv() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    MpiRequest req = {
      boost::make_shared<MPI_Request>(MPI_REQUEST_NULL),
      boost::bind(&MpiTreeClient::received, this, _1, _2, _3),
      false, 0, 0};
    MPI_Irecv(&buffer.front(), buffer.size(),
              MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
              req.req.get());

    DLOG(INFO) << "**** (set_recv) requests: " << requests.size();
    requests.push_back(req);
  }

  void received(bool ok, int size, int sender) {
    if (ok) {
      std::vector<Handler*> to_call;
      {
        boost::recursive_mutex::scoped_lock lock(mtx);
        to_call = handlers;
      }
      DLOG(INFO) << "[proc " << id() << "] received buffer of size: " << size;
      if (sender == parent()) {
        for (int i = 0; i < handlers.size(); ++i) {
          to_call[i]->received_from_parent(&buffer.front(), size);
        }
      } else {
        for (int i = 0; i < handlers.size(); ++i) {

    //           //DEBUG
    //           int MPI_rank;
    //   MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    // LOG(INFO) << "trace track ckpt 2 from child rank " << MPI_rank;


          to_call[i]->received_from_child(&buffer.front(), size, sender);
        }
      }
    } else {
      LOG(ERROR) << "RECEIVED FAILED";
    }

    set_recv();
  }

  bool is_ready(MpiRequest& request) {
      boost::recursive_mutex::scoped_lock lock(mtx);
      int flag = 0;
      MPI_Status status;
      int result = MPI_Test(request.req.get(), &flag, &status);
      if (flag) {
        request.ok = (result == MPI_SUCCESS);
        if (!request.ok) {
          LOG(ERROR) << "ERROR: " << mpi_get_error_string(result);
        }
        request.sender = status.MPI_SOURCE;
        result = MPI_Get_count(&status, MPI_CHAR, &request.size);
        if( SEPPUKU_TAG == status.MPI_TAG ) {
           finished=true;
           return false;
        }
        request.ok = (result == MPI_SUCCESS);
        if (!request.ok) {
          LOG(ERROR) << "ERROR: " << mpi_get_error_string(result);
        }
      }
      return flag;
  }

  bool is_finished() const {
      boost::recursive_mutex::scoped_lock lock(mtx);
      return finished;
  }

 public:
  explicit MpiTreeClient(boost::shared_ptr<Daemon> daemon)
      : daemon(daemon)
      , finished(false) {
    post(daemon);
  }

  virtual boost::shared_ptr<Daemon> get_daemon() {
    return daemon;
  }

  virtual void lets_die_together() const {
    if( id() == parent() ){ // if root has ended, shut down all nodes
        for (int i = 0; i < total_nodes(); i++) {
          if (i != id()) {
            MPI_Send(
                    &i, // can be anything, SEPPUKU_TAG shut downs caffe,
                    1,  // message is irrelevant
                    MPI_CHAR,
                    i,
                    SEPPUKU_TAG,
                    MPI_COMM_WORLD);
          }
        }
        finished = true;
    } else {
        // else if internode has ended,
        // wait for message from toot to do it (some messages may be pending)
        while(!is_finished()) {
            boost::this_thread::sleep(boost::posix_time::milliseconds(50));
        }
    }
  }

  virtual void set_buffer_size(size_t max_packet_size) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    buffer.resize(max_packet_size);
    set_recv();
  }

  virtual void async_send_to_parent(const char* buffer,
                                    size_t size,
                                    SentCallback callback) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    RemoteId parent_id = parent();

    MpiRequest req = {
      boost::make_shared<MPI_Request>(MPI_REQUEST_NULL),
      boost::bind(callback, _1),
      false, 0, 0};
    MPI_Isend(const_cast<char*>(buffer),
              size,
              MPI_CHAR,
              parent_id,
              MSG_TAG,
              MPI_COMM_WORLD,
              req.req.get());
    DLOG(INFO) << "**** (async_send_to_parent) requests: " << requests.size();
    requests.push_back(req);
  }

  virtual void async_send_to_children(const char* buff,
                                      size_t size,
                                      SentCallback callback) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    std::vector<RemoteId> children_ids = children();

    DLOG(INFO) << "[proc " << id() << "] sending buffer of size: " << size;


    BroadcastCallback<SentCallback> broadcast_callback(callback);
    int total_requests = 0;
    for (int i = 0; i < children_ids.size(); ++i) {
      MpiRequest req = {
        boost::make_shared<MPI_Request>(MPI_REQUEST_NULL),
        broadcast_callback,
        false, 0, 0};
      MPI_Isend(const_cast<char*>(buff),
                size,
                MPI_CHAR,
                children_ids[i],
                MSG_TAG,
                MPI_COMM_WORLD,
                req.req.get());
      requests.push_back(req);
      total_requests = requests.size();
    }
    DLOG(INFO) << "**** (async_send_to_children) requests: " << total_requests;


    // DEBUG
    int MPI_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
    if (MPI_rank == 0)
    LOG(INFO) << " communication rank " << MPI_rank << " thread id " << boost::this_thread::get_id() << " proc " << ::getpid();
 


  }

  virtual void register_receive_handler(Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    boost::recursive_mutex::scoped_lock lock(mtx);
    return mpi_get_current_proc_rank();
  }

  virtual int total_nodes() const {
    boost::recursive_mutex::scoped_lock lock(mtx);
    return mpi_get_comm_size();
  }

  // virtual std::vector<RemoteId> children() const {
  //   boost::recursive_mutex::scoped_lock lock(mtx);
  //   std::vector<RemoteId> children;
  //   RemoteId parent = id();
  //   int count = mpi_get_comm_size();

  //   if (count < 2) return children;

  //   if (parent * 2 + 1 < count) children.push_back( parent*2+1 );
  //   if (parent * 2 + 2 < count) children.push_back( parent*2+2 );

  //   return children;
  // }

  // virtual RemoteId parent() const {
  //   boost::recursive_mutex::scoped_lock lock(mtx);
  //   RemoteId current = id();

  //   if (current == 0) return 0;

  //   return floor( (current-1)/2 );
  // }

  // Modified by Jian
  virtual std::vector<RemoteId> children() const {
    boost::recursive_mutex::scoped_lock lock(mtx);
    std::vector<RemoteId> children;
    RemoteId current = id();
    int count = mpi_get_comm_size();

    int group_size = (count - 1) / nGroup;
    int group_id = current / group_size;
    
    // if (group_size < 2) return children;

    int offset = group_id * group_size;
    int residual = current - offset;
    if (residual * 2 + 1 < group_size) children.push_back(residual * 2 + 1 + offset);
    if (residual * 2 + 2 < group_size) children.push_back(residual * 2 + 2 + offset);

    // // DEBUG
    // for (int i = 0; i < children.size(); i++)
    //   std::cout << "ckpt children: current " << current << " child " << children[i];

    return children;
  }

  // Modified by Jian
  virtual RemoteId parent() const {
    boost::recursive_mutex::scoped_lock lock(mtx);
    RemoteId current = id();

    int count = mpi_get_comm_size();
    int group_size = (count - 1) / nGroup;
    int group_id = current / group_size;

    if (current % group_size == 0) {

      // // DEBUG
      // std::cout << "ckpt parent: current " << current << " parent " << current << std::endl;

      return current;
    }
    else {
      int offset = group_id * group_size;
      int residual = current - offset;

      // // DEBUG
      // std::cout << "ckpt parent: current " << current << " parent " << floor( (residual - 1) / 2 ) + offset << std::endl;

      return floor( (residual - 1) / 2 ) + offset;
    }
  }


  virtual void poll_one(shared_ptr<Daemon> daemon) {
    // Modified by Jian
    {
      boost::recursive_mutex::scoped_lock lock(mtx);
      if (!main_thread_id) {
        main_thread_id = boost::this_thread::get_id();
      }
      assert(*main_thread_id == boost::this_thread::get_id());      
    }


    {

      // DEBUG
      // LOG(INFO) << " poll one in thread " << boost::this_thread::get_id();
      // while(1);



      boost::recursive_mutex::scoped_lock lock(mtx);
      if (!main_thread_id) {
        main_thread_id = boost::this_thread::get_id();
      }
      assert(*main_thread_id == boost::this_thread::get_id());
      if (!requests.empty()) {
        requests_to_process.insert(
          requests_to_process.end(), requests.begin(), requests.end());
        requests.clear();
      }
    }
    typedef std::vector<MpiRequest>::iterator It;
    It middle = std::stable_partition(
      requests_to_process.begin(), requests_to_process.end(),
      boost::bind(&MpiTreeClient::is_ready, this, _1));
    for (It it = requests_to_process.begin(); it != middle; ++it) {
      it->callback(it->ok, it->size, it->sender);
    }
    requests_to_process.erase(requests_to_process.begin(), middle);
    if (requests_to_process.size() > 100) {
      LOG(WARNING) << "a lot of requests to process in tree cluster: "
                   << requests_to_process.size();
    }
    post(daemon);
  }

  virtual void post(shared_ptr<Daemon> daemon) {
    get_io_service(daemon).post(
      boost::bind(&MpiTreeClient::poll_one, this, daemon));
  }
};

TreeWaypoint* TreeWaypoint::get_instance() {
  static boost::shared_ptr<Daemon> daemon = create_communication_daemon();
  static MpiTreeClient instance(daemon);
  return &instance;
}

#else

TreeWaypoint* TreeWaypoint::get_instance() {
  LOG(ERROR) << "can't use MPI";
  throw std::runtime_error("can't use MPI");
}

#endif

}  // namespace internode
}  // namespace caffe

