#ifndef NV_KERNEL_H
#define NV_KERNEL_H

#include "salsa_kernel.h"

class NVKernel : public KernelInterface
{
public:
    NVKernel();

    virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V);
    virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache);

    virtual bool bindtexture_1D(uint32_t *d_V, size_t size);
    virtual bool bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch);
    virtual bool unbindtexture_1D();
    virtual bool unbindtexture_2D();

    virtual char get_identifier() { return 'K'; };
    virtual int get_major_version() { return 3; };
    virtual int get_minor_version() { return 0; };

    virtual int max_warps_per_block() { return 32; };
    virtual int get_texel_width() { return 4; };
    virtual bool support_lookup_gap() { return true; }
    virtual cudaSharedMemConfig shared_mem_config() { return cudaSharedMemBankSizeFourByte; }
    virtual cudaFuncCache cache_config() { return cudaFuncCachePreferL1; }

    virtual bool prepare_keccak256(int thr_id, const uint32_t host_pdata[20], const uint32_t ptarget[8]);
    virtual void do_keccak256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h = false);

    virtual bool prepare_blake256(int thr_id, const uint32_t host_pdata[20], const uint32_t host_ptarget[8]);
    virtual void do_blake256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h = false);
};

#endif // #ifndef NV_KERNEL_H
