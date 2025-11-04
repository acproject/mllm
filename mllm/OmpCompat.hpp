#ifndef MLLM_OMP_COMPAT_HPP
#define MLLM_OMP_COMPAT_HPP

#if defined(_OPENMP)
#include <omp.h>
#else
// Minimal fallback implementations when OpenMP is not available
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_num_procs(void) { return 1; }
static inline int omp_in_parallel(void) { return 0; }
static inline void omp_set_num_threads(int) {}
static inline void omp_set_max_active_levels(int) {}
static inline int omp_get_max_active_levels(void) { return 1; }
#endif

#endif // MLLM_OMP_COMPAT_HPP