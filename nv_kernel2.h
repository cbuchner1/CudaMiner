#ifndef NV2_KERNEL_H
#define NV2_KERNEL_H

#include "salsa_kernel.h"
#include "miner.h"

class NV2Kernel : public KernelInterface
{
public:
    NV2Kernel();

    virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V);
    virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache);

    virtual char get_identifier() { return 'T'; };
    virtual int get_major_version() { return 3; };
    virtual int get_minor_version() { return 5; };

    virtual int max_warps_per_block() { return opt_algo == ALGO_BLAKE ? 32 : 24; };
    virtual int get_texel_width() { return 4; };
    virtual bool no_textures() { return true; }
    virtual bool support_lookup_gap() { return true; }
    virtual cudaSharedMemConfig shared_mem_config() { return cudaSharedMemBankSizeFourByte; }
    virtual cudaFuncCache cache_config() { return cudaFuncCachePreferL1; }

    virtual bool prepare_keccak256(int thr_id, const uint32_t host_pdata[20], const uint32_t ptarget[8]);
    virtual void do_keccak256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h = false);

    virtual bool prepare_blake256(int thr_id, const uint32_t host_pdata[20], const uint32_t host_ptarget[8]);
    virtual void do_blake256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h = false);
};

#endif // #ifndef NV2_KERNEL_H
