//
// Contains the autotuning logic and some utility functions.
// Note that all CUDA kernels have been moved to other .cu files
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=124
//

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <map>
#include <algorithm>

#include <cuda.h>

#include "salsa_kernel.h"

#include "titan_kernel.h"
#include "spinlock_kernel.h"
#include "fermi_kernel.h"
#include "legacy_kernel.h"
#include "test_kernel.h"
#include "kepler_kernel.h"

#include "miner.h"

// require CUDA 5.5 driver API
#define DMAJ 5
#define DMIN 5

// some globals containing pointers to device memory (for chunked allocation)
// [8] indexes up to 8 threads (0...7)
int       MAXWARPS[8];
uint32_t* h_V[8][1024];
uint32_t  h_V_extra[8][1024];

extern "C" int cuda_num_devices()
{
    int version;
    int err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess)
    {
        applog(LOG_ERR, "Unable to query CUDA driver version! Is an nVidia driver installed?");
        exit(1);
    }

    int maj = version / 1000, min = version % 100; // same as in deviceQuery sample
    if (maj < DMAJ || (maj == DMAJ && min < DMIN))
    {
        applog(LOG_ERR, "Driver does not support CUDA %d.%d API! Update your nVidia driver!", DMAJ, DMIN);
        exit(1);
    }

    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        applog(LOG_ERR, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }
    return GPU_N;
}

bool validate_config(char *config, int &b, int &w, KernelInterface **kernel = NULL, cudaDeviceProp *props = NULL)
{
    bool success = false;
    char kernelid = ' ';
    if (config != NULL)
    {
        if (config[0] == 'T' || config[0] == 'S' || config[0] == 'K' || config[0] == 'F' || config[0] == 'L' ||
            config[0] == 'X') {
            kernelid = config[0];
            config++;
        }

        if (config[0] >= '0' && config[0] <= '9')
            if (sscanf(config, "%dx%d", &b, &w) == 2)
                success = true;

        if (success && kernel != NULL)
        {
            switch (kernelid)
            {
                case 'T': *kernel = new TitanKernel(); break;
                case 'S': *kernel = new SpinlockKernel(); break;
                case 'K': *kernel = new KeplerKernel(); break;
                case 'F': *kernel = new FermiKernel(); break;
                case 'L': *kernel = new LegacyKernel(); break;
                case 'X': *kernel = new TestKernel(); break;
                case ' ': // choose based on device architecture
                     if (props->major == 3 && props->minor == 5)
                    *kernel = new TitanKernel();
                else if (props->major == 3 && props->minor == 0)
                    *kernel = new KeplerKernel();
                else if (props->major == 2)
                    *kernel = new FermiKernel();
                else if (props->major == 1)
                    *kernel = new LegacyKernel();
                break;
            }
        }
    }
    return success;
}

std::map<int, int> context_blocks;
std::map<int, int> context_wpb;
std::map<int, bool> context_concurrent;
std::map<int, KernelInterface *> context_kernel;
std::map<int, uint32_t *> context_idata[2];
std::map<int, uint32_t *> context_odata[2];
std::map<int, cudaStream_t> context_streams[2];
std::map<int, uint32_t *> context_X[2];
std::map<int, uint32_t *> context_H[2];
std::map<int, cudaEvent_t> context_serialize[2];

// for SHA256 hashing on GPU
std::map<int, uint32_t *> context_tstate[2];
std::map<int, uint32_t *> context_ostate[2];
std::map<int, uint32_t *> context_hash[2];

int find_optimal_blockcount(int thr_id, KernelInterface* &kernel, bool &concurrent, int &wpb);

extern "C" void cuda_shutdown(int thr_id)
{
    checkCudaErrors(cudaStreamSynchronize(context_streams[0][thr_id]));
    checkCudaErrors(cudaStreamSynchronize(context_streams[1][thr_id]));
    cudaThreadExit();
}

extern "C" int cuda_throughput(int thr_id)
{
    int GRID_BLOCKS, WARPS_PER_BLOCK;
    if (context_blocks.find(thr_id) == context_blocks.end())
    {
#if 0
        CUcontext ctx;
        cuCtxCreate( &ctx, CU_CTX_SCHED_YIELD, device_map[thr_id] );
        cuCtxSetCurrent(ctx);
#else
        cudaSetDeviceFlags(cudaDeviceScheduleYield);
        cudaSetDevice(device_map[thr_id]);
        cudaFree(0);
#endif

        KernelInterface *kernel;
        bool concurrent; GRID_BLOCKS = find_optimal_blockcount(thr_id, kernel, concurrent, WARPS_PER_BLOCK);
        unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;
        unsigned int state_size = WU_PER_LAUNCH * sizeof(uint32_t) * 8;

        // allocate device memory for scrypt_core inputs and outputs
        uint32_t *tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_idata[0][thr_id] = tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_idata[1][thr_id] = tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_odata[0][thr_id] = tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, mem_size)); context_odata[1][thr_id] = tmp;

        // allocate pinned host memory for scrypt hashes
        checkCudaErrors(cudaHostAlloc((void **) &tmp, state_size, cudaHostAllocDefault)); context_H[0][thr_id] = tmp;
        checkCudaErrors(cudaHostAlloc((void **) &tmp, state_size, cudaHostAllocDefault)); context_H[1][thr_id] = tmp;

        if (parallel < 2)
        {
            // allocate pinned host memory for scrypt_core input/output
            checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[0][thr_id] = tmp;
            checkCudaErrors(cudaHostAlloc((void **) &tmp, mem_size, cudaHostAllocDefault)); context_X[1][thr_id] = tmp;
        }
        else
        {
            // allocate tstate, ostate, scrypt hash device memory
            checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_tstate[0][thr_id] = tmp;
            checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_tstate[1][thr_id] = tmp;
            checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_ostate[0][thr_id] = tmp;
            checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_ostate[1][thr_id] = tmp;
            checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[0][thr_id] = tmp;
            checkCudaErrors(cudaMalloc((void **) &tmp, state_size)); context_hash[1][thr_id] = tmp;
        }

        // create two CUDA streams
        cudaStream_t tmp2;
        checkCudaErrors( cudaStreamCreate(&tmp2) ); context_streams[0][thr_id] = tmp2;
        checkCudaErrors( cudaStreamCreate(&tmp2) ); context_streams[1][thr_id] = tmp2;

        // events used to serialize the kernel launches (we don't want any overlapping of kernels)
        cudaEvent_t tmp4;
        checkCudaErrors(cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); context_serialize[0][thr_id] = tmp4;
        checkCudaErrors(cudaEventCreateWithFlags(&tmp4, cudaEventDisableTiming)); context_serialize[1][thr_id] = tmp4;
        cudaEventRecord(context_serialize[1][thr_id]);

        context_kernel[thr_id] = kernel;
        context_concurrent[thr_id] = concurrent;
        context_blocks[thr_id] = GRID_BLOCKS;
        context_wpb[thr_id] = WARPS_PER_BLOCK;
    }

    GRID_BLOCKS = context_blocks[thr_id];
    WARPS_PER_BLOCK = context_wpb[thr_id];
    return WU_PER_LAUNCH;
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10, 8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11, 8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12, 8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13, 8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
//    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}

#ifdef WIN32
#include <windows.h>
static int console_width()
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
}
#else
int console_width()
{
    return 999;
}
#endif

int find_optimal_blockcount(int thr_id, KernelInterface* &kernel, bool &concurrent, int &WARPS_PER_BLOCK)
{
    int cw = console_width();
    int optimal_blocks = 0;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_map[thr_id]);
    concurrent = (props.concurrentKernels > 0);

    device_name[thr_id] = strdup(props.name);
    applog(LOG_INFO, "GPU #%d: %s with compute capability %d.%d", device_map[thr_id], props.name, props.major, props.minor);

    WARPS_PER_BLOCK = -1;

    // if not specified, use interactive mode for devices that have the watchdog timer enabled
    if (device_interactive[thr_id] == -1)
        device_interactive[thr_id] = props.kernelExecTimeoutEnabled;

    // turn off texture cache if not otherwise specified
    if (device_texturecache[thr_id] == -1)
        device_texturecache[thr_id] = 0;

    // if not otherwise specified or required, turn single memory allocations off as they reduce
    // the amount of memory that we can allocate on Windows Vista, 7 and 8 (WDDM driver model issue)
    if (device_singlememory[thr_id] == -1) device_singlememory[thr_id] = 0;

    // figure out which kernel implementation to use
    if (!validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK, &kernel, &props)) {
             if ((device_config[thr_id] != NULL && device_config[thr_id][0] == 'T') ||
                 ((device_config[thr_id] == NULL || !strcasecmp(device_config[thr_id], "auto")) && (props.major == 3 && props.minor == 5)))
            kernel = new TitanKernel();
        else if  (device_config[thr_id] != NULL && device_config[thr_id][0] == 'S')
            kernel = new SpinlockKernel();
        else if  ((device_config[thr_id] != NULL && device_config[thr_id][0] == 'K') ||
                 ((device_config[thr_id] == NULL || !strcasecmp(device_config[thr_id], "auto")) && (props.major == 3 && props.minor == 0)))
            kernel = new KeplerKernel();
        else if ((device_config[thr_id] != NULL && device_config[thr_id][0] == 'F') ||
                 ((device_config[thr_id] == NULL || !strcasecmp(device_config[thr_id], "auto")) && props.major == 2))
            kernel = new FermiKernel();
        else if ((device_config[thr_id] != NULL && device_config[thr_id][0] == 'L') ||
                 ((device_config[thr_id] == NULL || !strcasecmp(device_config[thr_id], "auto")) && props.major == 1))
            kernel = new LegacyKernel();
        else if ((device_config[thr_id] != NULL && device_config[thr_id][0] == 'X'))
            kernel = new TestKernel();
    }

    if (kernel->get_major_version() > props.major || kernel->get_major_version() == props.major && kernel->get_minor_version() > props.minor)
    {
        applog(LOG_ERR, "GPU #%d: the '%c' kernel requires %d.%d capability!", device_map[thr_id], kernel->get_identifier(), kernel->get_major_version(), kernel->get_minor_version());
    }

    // set whatever cache configuration and shared memory bank mode the kernel prefers
    cudaDeviceSetCacheConfig(kernel->cache_config());
    cudaDeviceSetSharedMemConfig(kernel->shared_mem_config());

    // some kernels (e.g. Titan) do not support the texture cache
    if (kernel->no_textures() && device_texturecache[thr_id]) {
        applog(LOG_WARNING, "GPU #%d: the '%c' kernel ignores the texture cache argument", device_map[thr_id], kernel->get_identifier());
        device_texturecache[thr_id] = 0;
    }

    // Texture caching only works with single memory allocation
    if (device_texturecache[thr_id]) device_singlememory[thr_id] = 1;

    if (kernel->single_memory() && !device_singlememory[thr_id]) {
        applog(LOG_WARNING, "GPU #%d: the '%c' kernel requires single memory allocation", device_map[thr_id], kernel->get_identifier());
        device_singlememory[thr_id] = 1;
    }

    applog(LOG_INFO, "GPU #%d: interactive: %d, tex-cache: %d%c, single-alloc: %d", device_map[thr_id],
           (device_interactive[thr_id]  != 0) ? 1 : 0,
           (device_texturecache[thr_id] != 0) ? device_texturecache[thr_id] : 0, (device_texturecache[thr_id] != 0) ? 'D' : ' ',
           (device_singlememory[thr_id] != 0) ? 1 : 0 );

    // compute highest MAXWARPS numbers for kernels allowing cudaBindTexture to succeed
    int MW_1D_4 = 134217728 / (SCRATCH * WU_PER_WARP / 4); // for uint4_t textures
    int MW_1D_2 = 134217728 / (SCRATCH * WU_PER_WARP / 2); // for uint2_t textures
    int MW_1D = kernel->get_texel_width() == 2 ? MW_1D_2 : MW_1D_4;

    uint32_t *d_V = NULL;
    if (device_singlememory[thr_id])
    {
        // if no launch config was specified, we simply
        // allocate the single largest memory chunk on the device that we can get
        if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK)) {
            MAXWARPS[thr_id] = optimal_blocks * WARPS_PER_BLOCK;
        }
        else {
            // compute no. of warps to allocate the largest number producing a single memory block below 4GB
            for (int warp = 0x7FFFFFFF / (SCRATCH * WU_PER_WARP * sizeof(uint32_t)); warp >= 1; --warp) {
                cudaGetLastError(); // clear the error state
                checkCudaErrors(cudaMalloc((void **)&d_V, SCRATCH * WU_PER_WARP * warp * sizeof(uint32_t)));
                if (cudaGetLastError() == cudaSuccess) {
                    checkCudaErrors(cudaFree(d_V)); d_V = NULL;
                    MAXWARPS[thr_id] = 90*warp/100; // Windows needs some breathing room to operate safely
                                                    // in particular when binding large 1D or 2D textures
                    break;
                }
            }
        }

        // now allocate a buffer for determined MAXWARPS setting
        cudaGetLastError(); // clear the error state
        checkCudaErrors(cudaMalloc((void **)&d_V, SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t)));
        if (cudaGetLastError() == cudaSuccess) {
            for (int i=0; i < MAXWARPS[thr_id]; ++i)
                h_V[thr_id][i] = d_V + SCRATCH * WU_PER_WARP * i;

            if (device_texturecache[thr_id] == 1)
            {
                if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
                {
                    if ( optimal_blocks * WARPS_PER_BLOCK > MW_1D )
                        applog(LOG_ERR, "GPU #%d: Given launch config '%s' exceeds limits for 1D cache.", device_map[thr_id], device_config[thr_id]);
                }
                // bind linear memory to a 1D texture reference
                if (kernel->get_texel_width() == 2)
                    kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * std::min(MAXWARPS[thr_id],MW_1D_2) * sizeof(uint32_t));
                else
                    kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * std::min(MAXWARPS[thr_id],MW_1D_4) * sizeof(uint32_t));
            }
            else if (device_texturecache[thr_id] == 2)
            {
                // bind pitch linear memory to a 2D texture reference
                if (kernel->get_texel_width() == 2)
                    kernel->bindtexture_2D(d_V, SCRATCH/2, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
                else
                    kernel->bindtexture_2D(d_V, SCRATCH/4, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
            }
        }
    }
    else
    {
        if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
            MAXWARPS[thr_id] = optimal_blocks * WARPS_PER_BLOCK;
        else
            MAXWARPS[thr_id] = 1024;

        // chunked memory allocation up to device limits
        int warp;
        for (warp = 0; warp < MAXWARPS[thr_id]; ++warp) {
            // work around partition camping problems by adding an offset
            h_V_extra[thr_id][warp] = (props.major == 1) ? (16 * (rand()%(16384/16))) : 0;
            cudaGetLastError(); // clear the error state
            checkCudaErrors(cudaMalloc((void **) &h_V[thr_id][warp], (SCRATCH * WU_PER_WARP + h_V_extra[thr_id][warp])*sizeof(uint32_t)));
            if (cudaGetLastError() == cudaSuccess) h_V[thr_id][warp] += h_V_extra[thr_id][warp];
            else {
                h_V_extra[thr_id][warp] = 0;
                // back off by two allocations to have some breathing room
                for (int i=0; warp > 0 && i < 2; ++i) {
                    warp--;
                    checkCudaErrors(cudaFree(h_V[thr_id][warp]-h_V_extra[thr_id][warp]));
                    h_V[thr_id][warp] = NULL; h_V_extra[thr_id][warp] = 0;
                }
                break;
            }
        }
        MAXWARPS[thr_id] = warp;
    }
    kernel->set_scratchbuf_constants(MAXWARPS[thr_id], h_V[thr_id]);

    if (validate_config(device_config[thr_id], optimal_blocks, WARPS_PER_BLOCK))
    {
        if (optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
            applog(LOG_ERR, "GPU #%d: Given launch config '%s' requires too much memory.", device_map[thr_id], device_config[thr_id]);
    }
    else
    {
        if (device_config[thr_id] != NULL && strcasecmp("auto", device_config[thr_id]))
            applog(LOG_WARNING, "GPU #%d: Given launch config '%s' does not validate.", device_map[thr_id], device_config[thr_id]);

        if (autotune)
        {
            applog(LOG_INFO, "GPU #%d: Performing auto-tuning (Patience...)", device_map[thr_id]);

            // allocate device memory
            unsigned int mem_size = MAXWARPS[thr_id] * WU_PER_WARP * sizeof(uint32_t) * 32;
            uint32_t *d_idata;
            checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
            uint32_t *d_odata;
            checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

            // pre-initialize some device memory
            uint32_t *h_idata = (uint32_t*)malloc(mem_size);
            for (unsigned int i=0; i < mem_size/sizeof(uint32_t); ++i) h_idata[i] = i*2654435761UL; // knuth's method
            checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));
            free(h_idata);

            double best_khash_sec = 0.0;
            int best_wpb = 0;

            // auto-tuning loop
            {
                // compute highest MAXWARPS number that we can support based on texture cache mode
                int MW = (device_texturecache[thr_id] == 1) ? std::min(MAXWARPS[thr_id],MW_1D) : MAXWARPS[thr_id];

                applog(LOG_INFO, "GPU #%d: maximum warps: %d", device_map[thr_id], MW);

                for (int GRID_BLOCKS = 1; !abort_flag && GRID_BLOCKS <= MW; ++GRID_BLOCKS)
                {
                    double kHash[24+1] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
                    for (WARPS_PER_BLOCK = 1; !abort_flag && WARPS_PER_BLOCK <= kernel->max_warps_per_block(); ++WARPS_PER_BLOCK)
                    {
                        double khash_sec = 0;
                        if (GRID_BLOCKS * WARPS_PER_BLOCK <= MW)
                        {
                            // setup execution parameters
                            dim3  grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
                            dim3  threads(WU_PER_BLOCK, 1, 1);

                            struct timeval tv_start, tv_end;
                            double tdelta = 0;

                            cudaDeviceSynchronize();
                            gettimeofday(&tv_start, NULL);
                            int repeat = 0;
                            bool r = false;
                            while (repeat < 3)  // average up to 3 measurements for better exactness
                            {
                                r=kernel->run_kernel(grid, threads, WARPS_PER_BLOCK, thr_id, NULL, d_idata, d_odata, device_interactive[thr_id], true, device_texturecache[thr_id]);
                                cudaDeviceSynchronize();
                                if (!r || cudaPeekAtLastError() != cudaSuccess) break;
                                ++repeat;
                                gettimeofday(&tv_end, NULL);
                                // bail out if 50ms taken (to speed up autotuning...)
                                if ((1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec)) > 0.05) break;
                            }
                            if (cudaGetLastError() != cudaSuccess || !r) continue;

                            tdelta = (1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec)) / repeat;

                            if (device_interactive[thr_id] && GRID_BLOCKS > 2*props.multiProcessorCount && tdelta > 1.0/30)
                                if (WARPS_PER_BLOCK == 1) goto skip; else goto skip2;

                            khash_sec = WU_PER_LAUNCH / (tdelta * 1e3);
                            kHash[WARPS_PER_BLOCK] = khash_sec;
                            if (khash_sec > best_khash_sec) {
                                optimal_blocks = GRID_BLOCKS;
                                best_khash_sec = khash_sec;
                                best_wpb = WARPS_PER_BLOCK;
                            }
                        }
                    }
skip2:              ;
                    if (opt_debug) {
                        if (GRID_BLOCKS == 1) {
                            char line[256] = "    ";
                            for (int i=1; i<=kernel->max_warps_per_block(); ++i) {
                                char tmp[16]; sprintf(tmp, "   x%-2d", i);
                                strcat(line, tmp);
                                if (cw == 80 && (i == 8 || i == 16)) strcat(line, "\n                          ");
                            }
                            applog(LOG_DEBUG, line);
                        }
                        char line[256]; sprintf(line, "%3d:", GRID_BLOCKS);
                        for (int i=1; i<=kernel->max_warps_per_block(); ++i) {
                            char tmp[16];
                            if (kHash[i]>0)
                                sprintf(tmp, "%5.1f%c", kHash[i], (i<kernel->max_warps_per_block())?'|':' ');
                            else
                                sprintf(tmp, "     %c", (i<kernel->max_warps_per_block())?'|':' ');
                            strcat(line, tmp);
                            if (cw == 80 && (i == 8 || i == 16)) strcat(line, "\n                          ");
                        }
                        strcat(line, "kH/s");
                        applog(LOG_DEBUG, line);
                    }
                }
skip:           ;
            }

            checkCudaErrors(cudaFree(d_odata));
            checkCudaErrors(cudaFree(d_idata));

            WARPS_PER_BLOCK = best_wpb;
            applog(LOG_INFO, "GPU #%d: %7.2f khash/s with configuration %c%dx%d", device_map[thr_id], best_khash_sec, kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);
        }
        else
        {
            // Heuristics for finding a good kernel launch configuration

            // base the initial block estimate on the number of multiprocessors
            int device_cores = props.multiProcessorCount * _ConvertSMVer2Cores(props.major, props.minor);

            // defaults, in case nothing else is chosen below
            optimal_blocks = 4 * device_cores / WU_PER_WARP;
            WARPS_PER_BLOCK = 2;

            // Based on compute capability, pick a known good block x warp configuration.
            if (props.major == 3)
            {
                if (props.minor == 0) // GK104, GK106, GK107
                {
                    if (MAXWARPS[thr_id] > (int)(optimal_blocks * 1.7261905) * 2)
                    {
                        // this results in 290x2 configuration on GTX 660Ti (3GB)
                        // but it requires 3GB memory on the card!
                        optimal_blocks = (int)(optimal_blocks * 1.7261905);
                        WARPS_PER_BLOCK = 2;
                    }
                    else
                    {
                        // this results in 148x2 configuration on GTX 660Ti (2GB)
                        optimal_blocks = (int)(optimal_blocks * 0.8809524);
                        WARPS_PER_BLOCK = 2;
                    }
                }
                else if (props.minor == 5) // GK110 (Tesla K20X, K20, GeForce GTX TITAN)
                {
                    // TODO: what to do with Titan and Tesla K20(X)?
                    // for now, do the same as for GTX 660Ti (2GB)
                    optimal_blocks = (int)(optimal_blocks * 0.8809524);
                    WARPS_PER_BLOCK = 2;
                }
            }
            // 1st generation Fermi (compute 2.0) GF100, GF110
            else if (props.major == 2 && props.minor == 0)
            {
                // this results in a 60x4 configuration on GTX 570
                optimal_blocks = 4 * device_cores / WU_PER_WARP;
                WARPS_PER_BLOCK = 4;
            }
            // 2nd generation Fermi (compute 2.1) GF104,106,108,114,116
            else if (props.major == 2 && props.minor == 1)
            {
                // this results in a 56x2 configuration on GTX 460
                optimal_blocks = props.multiProcessorCount * 8;
                WARPS_PER_BLOCK = 2;
            }
            // G80, G92, GT2xx
            else if (props.major == 1)
            {
                if (props.minor == 0)  // G80
                {
                    // TODO: anyone knowing good settings for G80?
                    optimal_blocks = props.multiProcessorCount;
                    WARPS_PER_BLOCK = 4;
                }
                else if (props.minor == 1)  // G92
                {
                    // e.g. my 9600M works best at 4x4
                    optimal_blocks = props.multiProcessorCount;
                    WARPS_PER_BLOCK = 4;
                }
                else if (props.minor == 2)  // GT218, GT216, GT215
                {
                    // TODO: anyone knowing good settings for Compute 1.2?
                    // for now I assume performance is identical to compute 1.3
                    optimal_blocks = props.multiProcessorCount;
                    WARPS_PER_BLOCK = 3;
                }
                if (props.minor == 3)  // GT200
                {
                    // my GTX 260 works best at S27x3
                    optimal_blocks = props.multiProcessorCount;
                    WARPS_PER_BLOCK = 3;
                }
            }

            // in case we run out of memory with the automatically chosen configuration,
            // first back off with WARPS_PER_BLOCK, then reduce optimal_blocks.
            if (WARPS_PER_BLOCK==3 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
                WARPS_PER_BLOCK = 2;
            while (optimal_blocks > 0 && optimal_blocks * WARPS_PER_BLOCK > MAXWARPS[thr_id])
                optimal_blocks--;
        }
    }

    applog(LOG_INFO, "GPU #%d: using launch configuration %c%dx%d", device_map[thr_id], kernel->get_identifier(), optimal_blocks, WARPS_PER_BLOCK);

    if (device_singlememory[thr_id])
    {
        if (MAXWARPS[thr_id] != optimal_blocks * WARPS_PER_BLOCK)
        {
            MAXWARPS[thr_id] = optimal_blocks * WARPS_PER_BLOCK;
            if (device_texturecache[thr_id] == 1)
                kernel->unbindtexture_1D();
            else if (device_texturecache[thr_id] == 2)
                kernel->unbindtexture_2D();
            checkCudaErrors(cudaFree(d_V)); d_V = NULL;

            cudaGetLastError(); // clear the error state
            checkCudaErrors(cudaMalloc((void **)&d_V, SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t)));
            if (cudaGetLastError() == cudaSuccess) {
                for (int i=0; i < MAXWARPS[thr_id]; ++i)
                    h_V[thr_id][i] = d_V + SCRATCH * WU_PER_WARP * i;

                if (device_texturecache[thr_id] == 1)
                {
                    // bind linear memory to a 1D texture reference
                    if (kernel->get_texel_width() == 2)
                        kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t));
                    else
                        kernel->bindtexture_1D(d_V, SCRATCH * WU_PER_WARP * MAXWARPS[thr_id] * sizeof(uint32_t));
                }
                else if (device_texturecache[thr_id] == 2)
                {
                    // bind pitch linear memory to a 2D texture reference
                    if (kernel->get_texel_width() == 2)
                        kernel->bindtexture_2D(d_V, SCRATCH/2, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
                    else
                        kernel->bindtexture_2D(d_V, SCRATCH/4, WU_PER_WARP * MAXWARPS[thr_id], SCRATCH*sizeof(uint32_t));
                }

                // update pointers to scratch buffer in constant memory after reallocation
                kernel->set_scratchbuf_constants(MAXWARPS[thr_id], h_V[thr_id]);
            }
            else
            {
                applog(LOG_ERR, "GPU #%d: Unable to allocate enough memory for launch config '%s'.", device_map[thr_id], device_config[thr_id]);
            }
        }
    }
    else
    {
        // back off unnecessary memory allocations to have some breathing room
        while (MAXWARPS[thr_id] > 0 && MAXWARPS[thr_id] > optimal_blocks * WARPS_PER_BLOCK) {
            (MAXWARPS[thr_id])--;
            checkCudaErrors(cudaFree(h_V[thr_id][MAXWARPS[thr_id]]-h_V_extra[thr_id][MAXWARPS[thr_id]]));
            h_V[thr_id][MAXWARPS[thr_id]] = NULL; h_V_extra[thr_id][MAXWARPS[thr_id]] = 0;
        }
    }

    return optimal_blocks;
}

cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id)
{
    cudaError_t result = cudaSuccess;
    static double tsum[3][8] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    double tsync = 0.0;
    double tsleep = 0.95 * tsum[situation][thr_id];
    if (cudaStreamQuery(stream) == cudaErrorNotReady)
    {
#ifdef WIN32
        Sleep((DWORD)(1000*tsleep));
#else
        usleep((useconds_t)(1e6*tsleep));
#endif
        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);
        checkCudaErrors(result = cudaStreamSynchronize(stream));
        gettimeofday(&tv_end, NULL);
        tsync = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec);
    }
    if (tsync >= 0) tsum[situation][thr_id] = 0.95 * tsum[situation][thr_id] + 0.05 * (tsleep+tsync);

    return result;
}

extern "C" void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream)
{
    int GRID_BLOCKS = context_blocks[thr_id];
    int WARPS_PER_BLOCK = context_wpb[thr_id];
    unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;

    // copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(context_idata[stream][thr_id], X, mem_size,
                               cudaMemcpyHostToDevice, context_streams[stream][thr_id]));
}

extern "C" void cuda_scrypt_serialize(int thr_id, int stream)
{
    // if the device can concurrently execute multiple kernels, then we must
    // wait for the serialization event recorded by the other stream
    if (context_concurrent[thr_id] || device_interactive[thr_id])
        checkCudaErrors(cudaStreamWaitEvent(context_streams[stream][thr_id], context_serialize[(stream+1)&1][thr_id], 0));
}

extern "C" void cuda_scrypt_done(int thr_id, int stream)
{
    // record the serialization event in the current stream
    checkCudaErrors(cudaEventRecord(context_serialize[stream][thr_id], context_streams[stream][thr_id]));
}

extern "C" void cuda_scrypt_flush(int thr_id, int stream)
{
    // flush the work queue (required for WDDM drivers)
    checkCudaErrors(cudaStreamQuery(context_streams[stream][thr_id]));
}

extern "C" void cuda_scrypt_core(int thr_id, int stream)
{
    int GRID_BLOCKS = context_blocks[thr_id];
    int WARPS_PER_BLOCK = context_wpb[thr_id];

    // setup execution parameters
    dim3  grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
    dim3  threads(WU_PER_BLOCK, 1, 1);

    if (device_interactive[thr_id]) {
//        checkCudaErrors(MyStreamSynchronize(context_streams[stream][thr_id], 2, thr_id));
#ifdef WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }

    context_kernel[thr_id]->run_kernel(grid, threads, WARPS_PER_BLOCK, thr_id, context_streams[stream][thr_id], context_idata[stream][thr_id], context_odata[stream][thr_id], device_interactive[thr_id], false, device_texturecache[thr_id]);
}

extern "C" void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream)
{
    int GRID_BLOCKS = context_blocks[thr_id];
    int WARPS_PER_BLOCK = context_wpb[thr_id];
    unsigned int mem_size = WU_PER_LAUNCH * sizeof(uint32_t) * 32;

    // copy result from device to host (asynchronously)
    checkCudaErrors(cudaMemcpyAsync(X, context_odata[stream][thr_id], mem_size,
                               cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
}

extern "C" void cuda_scrypt_sync(int thr_id, int stream)
{
    MyStreamSynchronize(context_streams[stream][thr_id], 0, thr_id);
}

extern "C" uint32_t* cuda_transferbuffer(int thr_id, int stream)
{
    return context_X[stream][thr_id];
}

extern "C" uint32_t* cuda_hashbuffer(int thr_id, int stream)
{
    return context_H[stream][thr_id];
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set on the CPU
//! @param idata      input data as provided to device
//! @param reference  reference data, computed but preallocated
//! @param V          scrypt scratchpad
////////////////////////////////////////////////////////////////////////////////
static void xor_salsa8(uint32_t * const B, const uint32_t * const C);

extern "C" void
computeGold(uint32_t *idata, uint32_t *reference, uint32_t *V)
{
	uint32_t X[32];
	int i,j,k;

	for (k = 0; k < 32; k++)
		X[k] = idata[k];
	
	for (i = 0; i < 1024; i++) {
		memcpy(&V[i * 32], X, 128);
		xor_salsa8(&X[0], &X[16]);
		xor_salsa8(&X[16], &X[0]);
	}
	for (i = 0; i < 1024; i++) {
		j = 32 * (X[16] & 1023);
		for (k = 0; k < 32; k++)
			X[k] ^= V[j + k];
		xor_salsa8(&X[0], &X[16]);
		xor_salsa8(&X[16], &X[0]);
	}
	for (k = 0; k < 32; k++)
		reference[k] = X[k];
}

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

static void xor_salsa8(uint32_t * const B, const uint32_t * const C)
{
    uint32_t x0 = (B[ 0] ^= C[ 0]), x1 = (B[ 1] ^= C[ 1]), x2 = (B[ 2] ^= C[ 2]), x3 = (B[ 3] ^= C[ 3]);
    uint32_t x4 = (B[ 4] ^= C[ 4]), x5 = (B[ 5] ^= C[ 5]), x6 = (B[ 6] ^= C[ 6]), x7 = (B[ 7] ^= C[ 7]);
    uint32_t x8 = (B[ 8] ^= C[ 8]), x9 = (B[ 9] ^= C[ 9]), xa = (B[10] ^= C[10]), xb = (B[11] ^= C[11]);
    uint32_t xc = (B[12] ^= C[12]), xd = (B[13] ^= C[13]), xe = (B[14] ^= C[14]), xf = (B[15] ^= C[15]);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);
        
    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);
        
    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    B[ 0] += x0; B[ 1] += x1; B[ 2] += x2; B[ 3] += x3; B[ 4] += x4; B[ 5] += x5; B[ 6] += x6; B[ 7] += x7;
    B[ 8] += x8; B[ 9] += x9; B[10] += xa; B[11] += xb; B[12] += xc; B[13] += xd; B[14] += xe; B[15] += xf;
}

//
//  =============== SHA256 part ======================
//

static const uint32_t host_sha256_h[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static const uint32_t host_sha256_k[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Elementary functions used by SHA256 */
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define ROTR(x, n)      ((x >> n) | (x << (32 - n)))
#define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

/* SHA256 round function */
#define RND(a, b, c, d, e, f, g, h, k) \
	do { \
		t0 = h + S1(e) + Ch(e, f, g) + k; \
		t1 = S0(a) + Maj(a, b, c); \
		d += t0; \
		h  = t0 + t1; \
	} while (0)

/* Adjusted round function for rotating state */
#define RNDr(S, W, i) \
	RND(S[(64 - i) % 8], S[(65 - i) % 8], \
	    S[(66 - i) % 8], S[(67 - i) % 8], \
	    S[(68 - i) % 8], S[(69 - i) % 8], \
	    S[(70 - i) % 8], S[(71 - i) % 8], \
	    W[i] + sha256_k[i])

static const uint32_t host_keypad[12] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000280
};

static const uint32_t host_innerpad[11] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x000004a0
};

static const uint32_t host_outerpad[8] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0x00000300
};

static const uint32_t host_finalblk[16] = {
	0x00000001, 0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000620
};

//
// CUDA code
//

__constant__ uint32_t sha256_h[8];
__constant__ uint32_t sha256_k[64];
__constant__ uint32_t keypad[12];
__constant__ uint32_t innerpad[11];
__constant__ uint32_t outerpad[8];
__constant__ uint32_t finalblk[16];
__constant__ uint32_t pdata[20];
__constant__ uint32_t midstate[8];

__device__ void mycpy16(uint32_t *d, const uint32_t *s) {
#pragma unroll 4
    for (int k=0; k < 4; k++) d[k] = s[k];
}

__device__ void mycpy32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
    for (int k=0; k < 8; k++) d[k] = s[k];
}

__device__ void mycpy44(uint32_t *d, const uint32_t *s) {
#pragma unroll 11
    for (int k=0; k < 11; k++) d[k] = s[k];
}

__device__ void mycpy48(uint32_t *d, const uint32_t *s) {
#pragma unroll 12
    for (int k=0; k < 12; k++) d[k] = s[k];
}

__device__ void mycpy64(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
    for (int k=0; k < 16; k++) d[k] = s[k];
}

__device__ void mycpy76(uint32_t *d, const uint32_t *s) {
#pragma unroll 19
    for (int k=0; k < 19; k++) d[k] = s[k];
}

__device__ void mycpy128(uint32_t *d, const uint32_t *s) {
#pragma unroll 32
    for (int k=0; k < 32; k++) d[k] = s[k];
}

__device__ void cuda_sha256_init(uint32_t *state)
{
	mycpy32(state, sha256_h);
}

__device__ uint32_t cuda_swab32(uint32_t x)
{
    return ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
          | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu));
}

/*
 * SHA256 block compression function.  The 256-bit state is transformed via
 * the 512-bit input block to produce a new state.
 */
__device__ void cuda_sha256_transform(uint32_t *state, const uint32_t *block, int swap)
{
	uint32_t W[64];
	uint32_t S[8];
	uint32_t t0, t1;
	int i;

	/* 1. Prepare message schedule W. */
	if (swap) {
#pragma unroll 16
		for (i = 0; i < 16; i++)
			W[i] = cuda_swab32(block[i]);
	} else
		mycpy64(W, block);
#pragma unroll 24
	for (i = 16; i < 64; i += 2) {
		W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}

	/* 2. Initialize working variables. */
	mycpy32(S, state);

	/* 3. Mix. */
	RNDr(S, W,  0);
	RNDr(S, W,  1);
	RNDr(S, W,  2);
	RNDr(S, W,  3);
	RNDr(S, W,  4);
	RNDr(S, W,  5);
	RNDr(S, W,  6);
	RNDr(S, W,  7);
	RNDr(S, W,  8);
	RNDr(S, W,  9);
	RNDr(S, W, 10);
	RNDr(S, W, 11);
	RNDr(S, W, 12);
	RNDr(S, W, 13);
	RNDr(S, W, 14);
	RNDr(S, W, 15);
	RNDr(S, W, 16);
	RNDr(S, W, 17);
	RNDr(S, W, 18);
	RNDr(S, W, 19);
	RNDr(S, W, 20);
	RNDr(S, W, 21);
	RNDr(S, W, 22);
	RNDr(S, W, 23);
	RNDr(S, W, 24);
	RNDr(S, W, 25);
	RNDr(S, W, 26);
	RNDr(S, W, 27);
	RNDr(S, W, 28);
	RNDr(S, W, 29);
	RNDr(S, W, 30);
	RNDr(S, W, 31);
	RNDr(S, W, 32);
	RNDr(S, W, 33);
	RNDr(S, W, 34);
	RNDr(S, W, 35);
	RNDr(S, W, 36);
	RNDr(S, W, 37);
	RNDr(S, W, 38);
	RNDr(S, W, 39);
	RNDr(S, W, 40);
	RNDr(S, W, 41);
	RNDr(S, W, 42);
	RNDr(S, W, 43);
	RNDr(S, W, 44);
	RNDr(S, W, 45);
	RNDr(S, W, 46);
	RNDr(S, W, 47);
	RNDr(S, W, 48);
	RNDr(S, W, 49);
	RNDr(S, W, 50);
	RNDr(S, W, 51);
	RNDr(S, W, 52);
	RNDr(S, W, 53);
	RNDr(S, W, 54);
	RNDr(S, W, 55);
	RNDr(S, W, 56);
	RNDr(S, W, 57);
	RNDr(S, W, 58);
	RNDr(S, W, 59);
	RNDr(S, W, 60);
	RNDr(S, W, 61);
	RNDr(S, W, 62);
	RNDr(S, W, 63);

	/* 4. Mix local working variables into global state */
#pragma unroll 8
	for (i = 0; i < 8; i++)
		state[i] += S[i];
}

//
// Original scrypt.cpp HMAC SHA256 functions
//

__device__ void cuda_HMAC_SHA256_80_init(const uint32_t *key,
	uint32_t *tstate, uint32_t *ostate)
{
	uint32_t ihash[8];
	uint32_t pad[16];
	int i;

	/* tstate is assumed to contain the midstate of key */
	mycpy16(pad, key + 16);
	mycpy48(pad + 4, keypad);
	cuda_sha256_transform(tstate, pad, 0);
	mycpy32(ihash, tstate);

	cuda_sha256_init(ostate);
#pragma unroll 8
	for (i = 0; i < 8; i++)
		pad[i] = ihash[i] ^ 0x5c5c5c5c;
#pragma unroll 8
	for (i=8; i < 16; i++)
		pad[i] = 0x5c5c5c5c;
	cuda_sha256_transform(ostate, pad, 0);

	cuda_sha256_init(tstate);
#pragma unroll 8
	for (i = 0; i < 8; i++)
		pad[i] = ihash[i] ^ 0x36363636;
#pragma unroll 8
	for (i=8; i < 16; i++)
		pad[i] = 0x36363636;
	cuda_sha256_transform(tstate, pad, 0);
}

__device__ void cuda_PBKDF2_SHA256_80_128(const uint32_t *tstate,
	const uint32_t *ostate, const uint32_t *salt, uint32_t *output)
{
	uint32_t istate[8], ostate2[8];
	uint32_t ibuf[16], obuf[16];
	int j;

	mycpy32(istate, tstate);
	cuda_sha256_transform(istate, salt, 0);
	
	mycpy16(ibuf, salt + 16);
	mycpy44(ibuf + 5, innerpad);
	mycpy32(obuf + 8, outerpad);

	mycpy32(obuf, istate);
	ibuf[4] = 1;
	cuda_sha256_transform(obuf, ibuf, 0);

	mycpy32(ostate2, ostate);
	cuda_sha256_transform(ostate2, obuf, 0);
#pragma unroll 8
	for (j = 0; j < 8; j++)
		output[0 + j] = cuda_swab32(ostate2[j]); // TODO: coalescing!

	mycpy32(obuf, istate);
	ibuf[4] = 2;
	cuda_sha256_transform(obuf, ibuf, 0);

	mycpy32(ostate2, ostate);
	cuda_sha256_transform(ostate2, obuf, 0);
#pragma unroll 8
	for (j = 0; j < 8; j++)
		output[8 + j] = cuda_swab32(ostate2[j]); // TODO: coalescing!

	mycpy32(obuf, istate);
	ibuf[4] = 3;
	cuda_sha256_transform(obuf, ibuf, 0);

	mycpy32(ostate2, ostate);
	cuda_sha256_transform(ostate2, obuf, 0);
#pragma unroll 8
	for (j = 0; j < 8; j++)
		output[16 + j] = cuda_swab32(ostate2[j]); // TODO: coalescing!

	mycpy32(obuf, istate);
	ibuf[4] = 4;
	cuda_sha256_transform(obuf, ibuf, 0);

	mycpy32(ostate2, ostate);
	cuda_sha256_transform(ostate2, obuf, 0);
#pragma unroll 8
	for (j = 0; j < 8; j++)
		output[24 + j] = cuda_swab32(ostate2[j]); // TODO: coalescing!
}

extern "C" void prepare_sha256(int thr_id, uint32_t host_pdata[20], uint32_t host_midstate[8])
{
    static bool init[8] = {false, false, false, false, false, false, false, false};
    if (!init[thr_id])
    {
        cudaMemcpyToSymbol(sha256_h, host_sha256_h, sizeof(host_sha256_h), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(sha256_k, host_sha256_k, sizeof(host_sha256_k), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(keypad, host_keypad, sizeof(host_keypad), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(innerpad, host_innerpad, sizeof(host_innerpad), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(outerpad, host_outerpad, sizeof(host_outerpad), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(finalblk, host_finalblk, sizeof(host_finalblk), 0, cudaMemcpyHostToDevice);
        init[thr_id] = true;
    }
    cudaMemcpyToSymbol(pdata, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(midstate, host_midstate, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__global__ void cuda_pre_sha256(uint32_t inp[32], uint32_t tstate_ext[8], uint32_t ostate_ext[8], uint32_t nonce)
{
	nonce += (blockIdx.x * blockDim.x) + threadIdx.x; 
	inp += 32 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	tstate_ext += 8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	ostate_ext += 8 * ((blockIdx.x * blockDim.x) + threadIdx.x);

	uint32_t ldata[20], tstate[8], ostate[8];
	mycpy76(ldata, pdata); ldata[19] = nonce;
	mycpy32(tstate, midstate);
	cuda_HMAC_SHA256_80_init(ldata, tstate, ostate);
	cuda_PBKDF2_SHA256_80_128(tstate, ostate, ldata, inp);

	// TODO: coalescing would be desired
	mycpy32(tstate_ext, tstate);
	mycpy32(ostate_ext, ostate);
}

extern "C" void pre_sha256(int thr_id, int stream, uint32_t nonce, int throughput)
{
    dim3 block(32);
    dim3 grid((throughput+31)/32);

    cuda_pre_sha256<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_idata[stream][thr_id], context_tstate[stream][thr_id], context_ostate[stream][thr_id], nonce);
}

__global__ void cuda_post_sha256(uint32_t output[8], uint32_t tstate_ext[8], uint32_t ostate_ext[8], uint32_t salt_ext[32])
{
	output += 8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	tstate_ext += 8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	ostate_ext += 8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
	salt_ext += 32 * ((blockIdx.x * blockDim.x) + threadIdx.x);

	uint32_t tstate[16];
	uint32_t salt[32];
	uint32_t buf[16];
	uint32_t ostate[16];
	int i;
	
	// TODO: coalescing would be desired
	mycpy32(tstate, tstate_ext);
	mycpy32(ostate, ostate_ext);
	mycpy128(salt, salt_ext);
	
	cuda_sha256_transform(tstate, salt, 1);
	cuda_sha256_transform(tstate, salt + 16, 1);
	cuda_sha256_transform(tstate, finalblk, 0);
	mycpy32(buf, tstate);
	mycpy32(buf + 8, outerpad);

	cuda_sha256_transform(ostate, buf, 0);
#pragma unroll 8
	for (i = 0; i < 8; i++)
		output[i] = cuda_swab32(ostate[i]); // TODO: coalescing
}

extern "C" void post_sha256(int thr_id, int stream, uint32_t hash[8], int throughput)
{
    dim3 block(32);
    dim3 grid((throughput+31)/32);

    cuda_post_sha256<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_hash[stream][thr_id], context_tstate[stream][thr_id], context_ostate[stream][thr_id], context_odata[stream][thr_id]);

    unsigned int mem_size = throughput * sizeof(uint32_t) * 8;

    // copy device memory to host
    checkCudaErrors(cudaMemcpyAsync(hash, context_hash[stream][thr_id], mem_size,
                    cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
}
