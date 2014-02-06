#ifndef KECCAK_H
#define KEKKAC_H

extern "C" void prepare_keccak512(int thr_id, const uint32_t host_pdata[20]);
extern "C" void pre_keccak512(int thr_id, int stream, uint32_t nonce, int throughput);
extern "C" void post_keccak512(int thr_id, int stream, uint32_t nonce, uint32_t hash[8], int throughput);

extern "C" void prepare_keccak256(int thr_id, const uint32_t host_pdata[20], const uint32_t ptarget[8]);
extern "C" uint32_t do_keccak256(int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h = false);

#endif // #ifndef KEKKAC_H
