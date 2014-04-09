#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>

extern "C" void prepare_sha256(int thr_id, uint32_t cpu_pdata[20], uint32_t cpu_midstate[8]);
extern "C" void pre_sha256(int thr_id, int stream, uint32_t nonce, int throughput);
extern "C" void post_sha256(int thr_id, int stream, int throughput);

#endif // #ifndef SHA256_H
