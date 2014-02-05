#include "cpuminer-config.h"
#include "miner.h"

#include <string.h>
#include <stdint.h>

// an alternative SHA-3 implementation, easier portable to CUDA IMHO
#include "sha3.h"

int scanhash_keccak(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, struct timeval *tv_start, struct timeval *tv_end, unsigned long *hashes_done)
{
	gettimeofday(tv_start, NULL);

	uint32_t n = pdata[19] - 1;
	const uint32_t first_nonce = pdata[19];
	const uint32_t Htarg = ptarget[7];

	uint32_t endiandata[32];
	for (int kk=0; kk < 32; kk++) // why 32 and not 20?
		be32enc(&endiandata[kk], ((uint32_t*)pdata)[kk]);

	do {
		pdata[19] = ++n;

		uint32_t hash64[8];
		crypto_hash( (unsigned char*)hash64, (unsigned char*)&endiandata[0], 80 );

		if (hash64[7] <= Htarg && // if (((hash64[7]&0xFFFFFF00)==0) && 
				fulltest(hash64, ptarget)) {
			*hashes_done = n - first_nonce + 1;
			gettimeofday(tv_end, NULL);
			return true;
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	gettimeofday(tv_end, NULL);
	return 0;
}
