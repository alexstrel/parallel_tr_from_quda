#include <limits>
#include <iostream>
#include <array>
#include <stdexcept>
//
#include <mpi.h>
//
#include <malloc_quda.h>
#include <tune_quda.h>
#include <quda.h>
#include <comm_quda.h>
#include <quda_api.h>
#include <device.h>
#include <timer.h>
//#include <cuda_runtime.h>
//#include <register_traits.h>
//#include <complex_quda.h>
#include <transform_reduce.h>

template<typename Tp>
struct AlignedAllocator {
   public:

     typedef Tp value_type;

     AlignedAllocator () {};

     AlignedAllocator(const AlignedAllocator&) { }

     template<typename Tp1> constexpr AlignedAllocator(const AlignedAllocator<Tp1>&) { }

     ~AlignedAllocator() { }

     Tp* address(Tp& x) const { return &x; }

     std::size_t  max_size() const throw() { return size_t(-1) / sizeof(Tp); }

     [[nodiscard]] Tp* allocate(std::size_t n){

       Tp* ptr = nullptr;
       auto err = cudaMallocManaged((void **)&ptr,n*sizeof(Tp));

       if( err != cudaSuccess ) {
         ptr = (Tp *) NULL;
         std::cerr << " cudaMallocManaged failed for " << n*sizeof(Tp) << " bytes " <<cudaGetErrorString(err)<< std::endl;
         //assert(0);
       }

       return ptr;
     }

     void deallocate( Tp* p, std::size_t n) noexcept {
       cudaFree((void *)p);
       return;
     }
};


std::array<int, 4> gridsize_from_cmdline = {1, 1, 1, 1};
int rank_order = 0;//col => 0, row => 1
#if 1
//!< Profiler for initQuda
static TimeProfile profileInit("initQuda");

//!< Profiler for endQuda
static TimeProfile profileEnd("endQuda");

//!< Profiler for GEMM and other BLAS
static TimeProfile profileBLAS("blasQuda");
TimeProfile &getProfileBLAS() { return profileBLAS; }
#endif
//!< Profiler for toal time spend between init and end
static TimeProfile profileInit2End("initQuda-endQuda",false);


static bool enable_profiler = false;
static bool do_not_profile_quda = false;

static void profilerStart(const char *f)
{
  static std::vector<int> target_list;
  static bool enable = false;
  static bool init = false;
  if (!init) {
    char *profile_target_env = getenv("QUDA_ENABLE_TARGET_PROFILE"); // selectively enable profiling for a given solve

    if ( profile_target_env ) {
      std::stringstream target_stream(profile_target_env);

      int target;
      while(target_stream >> target) {
       target_list.push_back(target);
       if (target_stream.peek() == ',') target_stream.ignore();
     }

     if (target_list.size() > 0) {
       std::sort(target_list.begin(), target_list.end());
       target_list.erase( unique( target_list.begin(), target_list.end() ), target_list.end() );
       warningQuda("Targeted profiling enabled for %lu functions\n", target_list.size());
       enable = true;
     }
   }

    char* donotprofile_env = getenv("QUDA_DO_NOT_PROFILE"); // disable profiling of QUDA parts
    if (donotprofile_env && (!(strcmp(donotprofile_env, "0") == 0)))  {
      do_not_profile_quda=true;
      printfQuda("Disabling profiling in QUDA\n");
    }
    init = true;
  }

  static int target_count = 0;
  static unsigned int i = 0;
  if (do_not_profile_quda){
    device::profile::stop();
    printfQuda("Stopping profiling in QUDA\n");
  } else {
    if (enable) {
      if (i < target_list.size() && target_count++ == target_list[i]) {
        enable_profiler = true;
        printfQuda("Starting profiling for %s\n", f);
        device::profile::start();
        i++; // advance to next target
    }
  }
}
}

static void profilerStop(const char *f) {
  if (do_not_profile_quda) {
    device::profile::start();
  } else {

    if (enable_profiler) {
      printfQuda("Stopping profiling for %s\n", f);
      device::profile::stop();
      enable_profiler = false;
    }
  }
}


namespace quda {
  void printLaunchTimer();
}




MPI_Comm MPI_COMM_HANDLE_USER;
static bool user_set_comm_handle = false;

void setMPICommHandleQuda(void *mycomm)//??
{
  MPI_COMM_HANDLE_USER = *((MPI_Comm *)mycomm);
  user_set_comm_handle = true;
}

template <typename T, typename count_t> struct compute_axpyDot {
  const T *x;
  T *y;
  const T a;
  count_t n_items;

  compute_axpyDot(const T a_, const T *x_, T *y_,  count_t n) : a(a_), x(x_), y(y_), n_items(n) {}

  T operator() (count_t idx, count_t j = 0) const {
    y[idx] = a*x[idx] + y[idx];
    return (y[idx]*x[idx]);
  }
};


int lex_rank_from_coords_t(const int *coords, void *)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

int lex_rank_from_coords_x(const int *coords, void *)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

void initComms(int argc, char **argv, int *const commDims)
{
  MPI_Init(&argc, &argv);

  QudaCommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;//see quda.h

  comm_init(4, commDims, func, NULL, user_set_comm_handle, (void *)&MPI_COMM_HANDLE_USER);
  
  //printfQuda("Rank order is %s major (%s running fastest)\n", rank_order == 0 ? "column" : "row", rank_order == 0 ? "t" : "x");
}

void initComms(int argc, char **argv, std::array<int, 4> &commDims) { initComms(argc, argv, commDims.data()); }

void finalizeComms()
{
  comm_finalize();
  MPI_Finalize();
}


int main(int argc, char **argv) {
   //
   profileInit2End.TPSTART(QUDA_PROFILE_TOTAL);
#if 1      
   profileInit.TPSTART(QUDA_PROFILE_TOTAL);
   profileInit.TPSTART(QUDA_PROFILE_INIT);
#endif
   std::cout << "Begin init: " << std::endl;
   initComms(argc, argv, gridsize_from_cmdline);
   
   quda::device::init(0);
   
   loadTuneCache();   

   quda::device::create_context();

   loadTuneCache();

   // initalize the memory pool allocators
   quda::pool::init();

   quda::reducer::init();
   std::cout << "..done." << std::endl;   
#if 1
   profileInit.TPSTOP(QUDA_PROFILE_INIT);
   profileInit.TPSTOP(QUDA_PROFILE_TOTAL);
#endif
   
   constexpr int N = 1024*1024;	
   //
   using alloc = AlignedAllocator<float>;
   std::vector<float, alloc> x(N, 1.0);
   //
   QudaFieldLocation location = QUDA_CUDA_FIELD_LOCATION;
   //
   std::cout << "Begin transform reduce:: " << std::endl;
   float result = quda::transform_reduce(location, x.data(), N, quda::identity<float>(), 0.0f, quda::plus<float>());  
   //
   std::cout << std::fixed << result << std::endl;
   
   quda::reducer::destroy();  
//   
   pool::flush_pinned();
   pool::flush_device();
   saveTuneCache();
   saveProfile();
//   
    
   finalizeComms();

   return 0;	
}




