#ifndef SALSA_KERNEL_H
#define SALSA_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// from salsa_kernel.cu
int cuda_num_devices();
void cuda_shutdown(int thr_id);
extern int device_map[8];
extern int device_interactive[8];
extern int device_texturecache[8];
extern int device_singlememory[8];
extern char *device_config[8];
extern char *device_name[8];
extern bool autotune;

// CUDA externals
int cuda_throughput(int thr_id);
uint32_t *cuda_transferbuffer(int thr_id, int stream);
void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream, bool flush);
void cuda_scrypt_core(int thr_id, int stream, bool flush);
void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream, bool flush);
void cuda_scrypt_sync(int thr_id, int stream);
void computeGold(uint32_t *idata, uint32_t *reference, uint32_t *V);

#ifdef __cplusplus
}
#endif

#endif // #ifndef SALSA_KERNEL_H
