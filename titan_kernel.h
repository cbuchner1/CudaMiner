#ifndef TITAN_KERNEL_H
#define TITAN_KERNEL_H

#include "salsa_kernel.h"

class TitanKernel : public KernelInterface
{
public:
    TitanKernel();

    virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V);
    virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache);

    virtual char get_identifier() { return 't'; }
    virtual int get_major_version() { return 3; }
    virtual int get_minor_version() { return 5; }

    virtual int max_warps_per_block() { return 32; }
    virtual int get_texel_width() { return 4; }
    virtual bool no_textures() { return true; }
    virtual int threads_per_wu() { return 4; }
    virtual bool support_lookup_gap() { return true; }
    virtual cudaFuncCache cache_config() { return cudaFuncCachePreferL1; }
};

#endif // #ifndef TITAN_KERNEL_H
