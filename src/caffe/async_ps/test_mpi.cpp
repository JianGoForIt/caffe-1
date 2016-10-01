#include <mpi.h>
#include <cassert>

int main() {
  int permit;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &permit);
  std::cout << permit << " " << MPI_THREAD_SINGLE << " " << MPI_THREAD_MULTIPLE << std::endl;
  assert(permit == MPI_THREAD_MULTIPLE);

  MPI_Finalize();
  return 0;
}