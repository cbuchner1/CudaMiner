/*
	scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane

	Public Domain or MIT License, whichever is easier
*/

#include "cpuminer-config.h"
#include "miner.h"
#include "salsa_kernel.h"

#include <string.h>

/* Hard-coded scrypt parameteres r and p - mikaelh */
#define SCRYPT_R 1
#define SCRYPT_P 1

/* Only the instrinsics versions are optimized for hard-coded values - mikaelh */
#define CPU_X86_FORCE_INTRINSICS

#include "scrypt-jane.h"
#include "code/scrypt-jane-portable.h"
#include "code/scrypt-jane-hash.h"
#include "code/scrypt-jane-romix.h"
#include "code/scrypt-jane-test-vectors.h"


#define scrypt_maxN 30  /* (1 << (30 + 1)) = ~2 billion */
#if (SCRYPT_BLOCK_BYTES == 64)
#define scrypt_r_32kb 8 /* (1 << 8) = 256 * 2 blocks in a chunk * 64 bytes = Max of 32kb in a chunk */
#elif (SCRYPT_BLOCK_BYTES == 128)
#define scrypt_r_32kb 7 /* (1 << 7) = 128 * 2 blocks in a chunk * 128 bytes = Max of 32kb in a chunk */
#elif (SCRYPT_BLOCK_BYTES == 256)
#define scrypt_r_32kb 6 /* (1 << 6) = 64 * 2 blocks in a chunk * 256 bytes = Max of 32kb in a chunk */
#elif (SCRYPT_BLOCK_BYTES == 512)
#define scrypt_r_32kb 5 /* (1 << 5) = 32 * 2 blocks in a chunk * 512 bytes = Max of 32kb in a chunk */
#endif
#define scrypt_maxr scrypt_r_32kb /* 32kb */
#define scrypt_maxp 25  /* (1 << 25) = ~33 million */

#include <stdio.h>
#include <malloc.h>

static void
scrypt_fatal_error_default(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

static scrypt_fatal_errorfn scrypt_fatal_error = scrypt_fatal_error_default;

void
scrypt_set_fatal_error_default(scrypt_fatal_errorfn fn) {
	scrypt_fatal_error = fn;
}

typedef struct scrypt_aligned_alloc_t {
	uint8_t *mem, *ptr;
} scrypt_aligned_alloc;

#if defined(SCRYPT_TEST_SPEED)
static uint8_t *mem_base = (uint8_t *)0;
static size_t mem_bump = 0;

/* allocations are assumed to be multiples of 64 bytes and total allocations not to exceed ~1.01gb */
static scrypt_aligned_alloc
scrypt_alloc(uint64_t size) {
	scrypt_aligned_alloc aa;
	if (!mem_base) {
		mem_base = (uint8_t *)malloc((1024 * 1024 * 1024) + (1024 * 1024) + (SCRYPT_BLOCK_BYTES - 1));
		if (!mem_base)
			scrypt_fatal_error("scrypt: out of memory");
		mem_base = (uint8_t *)(((size_t)mem_base + (SCRYPT_BLOCK_BYTES - 1)) & ~(SCRYPT_BLOCK_BYTES - 1));
	}
	aa.mem = mem_base + mem_bump;
	aa.ptr = aa.mem;
	mem_bump += (size_t)size;
	return aa;
}

static void
scrypt_free(scrypt_aligned_alloc *aa) {
	mem_bump = 0;
}
#else
static scrypt_aligned_alloc
scrypt_alloc(uint64_t size) {
	static const size_t max_alloc = (size_t)-1;
	scrypt_aligned_alloc aa;
	size += (SCRYPT_BLOCK_BYTES - 1);
	if (size > max_alloc)
		scrypt_fatal_error("scrypt: not enough address space on this CPU to allocate required memory");
	aa.mem = (uint8_t *)malloc((size_t)size);
	aa.ptr = (uint8_t *)(((size_t)aa.mem + (SCRYPT_BLOCK_BYTES - 1)) & ~(SCRYPT_BLOCK_BYTES - 1));
	if (!aa.mem)
		scrypt_fatal_error("scrypt: out of memory");
	return aa;
}

static void
scrypt_free(scrypt_aligned_alloc *aa) {
	free(aa->mem);
}
#endif


static int
scrypt_power_on_self_test() {
	const scrypt_test_setting *t;
	uint8_t test_digest[64];
	uint32_t i;
	int res = 7, scrypt_valid;
	scrypt_aligned_alloc YX, V;
	uint8_t *X, *Y;
	uint32_t N, chunk_bytes;
	const uint32_t r = SCRYPT_R;
	const uint32_t p = SCRYPT_P;

	if (!scrypt_test_mix()) {
#if !defined(SCRYPT_TEST)
		scrypt_fatal_error("scrypt: mix function power-on-self-test failed");
#endif
		res &= ~1;
	}

	if (!scrypt_test_hash()) {
#if !defined(SCRYPT_TEST)
		scrypt_fatal_error("scrypt: hash function power-on-self-test failed");
#endif
		res &= ~2;
	}

	for (i = 0, scrypt_valid = 1; post_settings[i].pw; i++) {
		t = post_settings + i;
		
		N = (1 << (t->Nfactor + 1));
		
		chunk_bytes = SCRYPT_BLOCK_BYTES * r * 2;
		V = scrypt_alloc((uint64_t)N * chunk_bytes);
		YX = scrypt_alloc((p + 1) * chunk_bytes);
		
		Y = YX.ptr;
		X = Y + chunk_bytes;
		
		scrypt_N_1_1((uint8_t *)t->pw, strlen(t->pw), (uint8_t *)t->salt, strlen(t->salt), N, test_digest, sizeof(test_digest), X, Y, V.ptr);
		scrypt_valid &= scrypt_verify(post_vectors[i], test_digest, sizeof(test_digest));
		
		scrypt_free(&V);
		scrypt_free(&YX);
	}
	
	if (!scrypt_valid) {
#if !defined(SCRYPT_TEST)
		scrypt_fatal_error("scrypt: scrypt power-on-self-test failed");
#endif
		res &= ~4;
	}

	return res;
}


void
scrypt_N_1_1(const uint8_t *password, size_t password_len, const uint8_t *salt, size_t salt_len, uint32_t N, uint8_t *out, size_t bytes, uint8_t *X, uint8_t *Y, uint8_t *V) {
	uint32_t chunk_bytes, i;
	const uint32_t r = SCRYPT_R;
	const uint32_t p = SCRYPT_P;

#if !defined(SCRYPT_CHOOSE_COMPILETIME)
	scrypt_ROMixfn scrypt_ROMix = scrypt_getROMix();
#endif

	chunk_bytes = SCRYPT_BLOCK_BYTES * r * 2;

	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_1(password, password_len, salt, salt_len, X, chunk_bytes * p);

	/* 2: X = ROMix(X) */
	for (i = 0; i < p; i++)
		scrypt_ROMix_1((scrypt_mix_word_t *)(X + (chunk_bytes * i)), (scrypt_mix_word_t *)Y, (scrypt_mix_word_t *)V, N);

	/* 3: Out = PBKDF2(password, X) */
	scrypt_pbkdf2_1(password, password_len, X, chunk_bytes * p, out, bytes);

#ifdef SCRYPT_PREVENT_STATE_LEAK
	/* This is an unnecessary security feature - mikaelh */
	scrypt_ensure_zero(Y, (p + 1) * chunk_bytes);
#endif
}


// yacoin: increasing Nfactor gradually
unsigned char GetNfactor(unsigned int nTimestamp) {
	int l = 0;

	unsigned int Nfactor = 0;

	// Yacoin defaults
	unsigned int Ntimestamp = 1367991200;
	unsigned int minN = 4;
	unsigned int maxN = 30;
	
	if (strlen(jane_params) > 0) {
		if (!strcmp(jane_params, "YAC") || !strcasecmp(jane_params, "Yacoin")) {} // No-Op
		else if (!strcmp(jane_params, "YBC") || !strcasecmp(jane_params, "YBCoin")) {
			// YBCoin:   1372386273, minN:  4, maxN: 30
			Ntimestamp = 1372386273; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "ZZC") || !strcasecmp(jane_params, "ZZCoin")) {
			// ZcCoin:   1375817223, minN: 12, maxN: 30
			Ntimestamp = 1375817223; minN= 12; maxN= 30;
		} else if (!strcmp(jane_params, "FEC") || !strcasecmp(jane_params, "FreeCoin")) {
			// FreeCoin: 1375801200, minN:  6, maxN: 32
			Ntimestamp = 1375801200; minN=  6; maxN= 32;
		} else if (!strcmp(jane_params, "ONC") || !strcasecmp(jane_params, "OneCoin")) {
			// OneCoin:  1371119462, minN:  6, maxN: 30
			Ntimestamp = 1371119462; minN=  6; maxN= 30;
		} else if (!strcmp(jane_params, "QQC") || !strcasecmp(jane_params, "QQCoin")) {
			// QQCoin:   1387769316, minN:  4, maxN: 30
			Ntimestamp = 1387769316; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "GPL") || !strcasecmp(jane_params, "GoldPressedLatinum")) {
			// GoldPressedLatinum:   1377557832, minN:  4, maxN: 30
			Ntimestamp = 1377557832; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "MRC") || !strcasecmp(jane_params, "MicroCoin")) {
			// MicroCoin:1389028879, minN:  4, maxN: 30
			Ntimestamp = 1389028879; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "APC") || !strcasecmp(jane_params, "AppleCoin")) {
			// AppleCoin:1384720832, minN:  4, maxN: 30
			Ntimestamp = 1384720832; minN=  4; maxN= 30;
		} else {
			if (sscanf(jane_params, "%u,%u,%u", &Ntimestamp, &minN, &maxN) != 3)
			if (sscanf(jane_params, "%u", &Nfactor) == 1) return Nfactor; // skip bounding against minN, maxN
			else applog(LOG_INFO, "Unable to parse scrypt-jane parameters: '%s'. Defaulting to Yacoin.", jane_params);
		}
	}
	// determination based on the constants determined above
	if (nTimestamp <= Ntimestamp)
		return minN;

	unsigned long int s = nTimestamp - Ntimestamp;
	while ((s >> 1) > 3) {
		l += 1;
		s >>= 1;
	}

	s &= 3;

	int n = (l * 170 + s * 25 - 2320) / 100;

	if (n < 0) n = 0;

	if (n > 255)
		printf("GetNfactor(%d) - something wrong(n == %d)\n", nTimestamp, n);

	Nfactor = n;
	if (Nfactor<minN) return minN;
	if (Nfactor>maxN) return maxN;
	return Nfactor;
}

#define bswap_32x4(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                     | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

int scanhash_scrypt_jane(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, struct timeval *tv_start, struct timeval *tv_end, unsigned long *hashes_done)
{
	const uint32_t Htarg = ptarget[7];
	static int s_Nfactor = 0;

	if (s_Nfactor == 0 && strlen(jane_params) > 0)
		applog(LOG_INFO, "Given scrypt-jane parameters: %s", jane_params);
	
	int Nfactor = GetNfactor(bswap_32x4(pdata[17]));
	if (Nfactor > scrypt_maxN) {
		scrypt_fatal_error("scrypt: N out of range");
	}
	
	N = (1 << (Nfactor + 1));

	if (Nfactor != s_Nfactor)
	{
		s_Nfactor = Nfactor;
		applog(LOG_INFO, "Nfactor is %d (N=%d)!", Nfactor, N);
	}

	parallel = 0;
	int throughput = cuda_throughput(thr_id);
	
	gettimeofday(tv_start, NULL);

	uint32_t *data[2] = { new uint32_t[20*throughput], new uint32_t[20*throughput] };
	uint32_t *hash = new uint32_t[8*throughput];

	uint32_t n = pdata[19] - 1;
//	int i;
	
#if !defined(SCRYPT_TEST)
	static int power_on_self_test = 0;
	if (!power_on_self_test) {
		power_on_self_test = 1;
		if (!scrypt_power_on_self_test())
			scrypt_fatal_error("scrypt: power on self test failed");
	}
#endif

	/* byte swap pdata into data[0]/[1] arrays */
	for (int k=0; k<2; ++k) {
		for(int z=0;z<20;z++) data[k][z] = bswap_32x4(pdata[z]);
		for(int i=1;i<throughput;++i) memcpy(&data[k][20*i], &data[k][0], 20*sizeof(uint32_t));
	}

	scrypt_aligned_alloc Xbuf[2] = { scrypt_alloc(128 * throughput), scrypt_alloc(128 * throughput) };
	scrypt_aligned_alloc Vbuf = scrypt_alloc((uint64_t)N * 128);
	scrypt_aligned_alloc Ybuf = scrypt_alloc(128);

	uint32_t nonce[2];
	uint32_t* cuda_X[2]      = { cuda_transferbuffer(thr_id,0), cuda_transferbuffer(thr_id,1) };

#if !defined(SCRYPT_CHOOSE_COMPILETIME)
	scrypt_ROMixfn scrypt_ROMix = scrypt_getROMix();
#endif

	int cur = 0, nxt = 1;

	nonce[cur] = n+1;

	for(int i=0;i<throughput;++i) {
		uint32_t tmp_nonce = ++n;
		data[cur][20*i + 19] = bswap_32x4(tmp_nonce);
	}

	/* 1: X = PBKDF2(password, salt) */
	for(int i=0;i<throughput;++i)
		scrypt_pbkdf2_1((unsigned char *)&data[cur][20*i], 80, (unsigned char *)&data[cur][20*i], 80, Xbuf[cur].ptr + 128 * i, 128);

	/* 2: X = ROMix(X) in CUDA */
	memcpy(cuda_X[cur], Xbuf[cur].ptr, 128 * throughput);
	cuda_scrypt_HtoD(thr_id, cuda_X[cur], cur);
	cuda_scrypt_serialize(thr_id, cur);
	cuda_scrypt_core(thr_id, cur, N);
	cuda_scrypt_done(thr_id, cur);
	cuda_scrypt_DtoH(thr_id, cuda_X[cur], cur);
	cuda_scrypt_flush(thr_id, cur);

	do {
		nonce[nxt] = n+1;

		for(int i=0;i<throughput;++i) {
			uint32_t tmp_nonce = ++n;
			data[nxt][20*i + 19] = bswap_32x4(tmp_nonce);
		}

		/* 1: X = PBKDF2(password, salt) */
		for(int i=0;i<throughput;++i)
			scrypt_pbkdf2_1((unsigned char *)&data[nxt][20*i], 80, (unsigned char *)&data[nxt][20*i], 80, Xbuf[nxt].ptr + 128 * i, 128);

		/* 2: X = ROMix(X) in CUDA */
		memcpy(cuda_X[nxt], Xbuf[nxt].ptr, 128 * throughput);
		cuda_scrypt_HtoD(thr_id, cuda_X[nxt], nxt);
		cuda_scrypt_serialize(thr_id, nxt);
		cuda_scrypt_core(thr_id, nxt, N);
		cuda_scrypt_done(thr_id, nxt);
		cuda_scrypt_DtoH(thr_id, cuda_X[nxt], nxt);
		cuda_scrypt_flush(thr_id, nxt);

		cuda_scrypt_sync(thr_id, cur);

#define VERIFY_ALL 0
#if VERIFY_ALL
		{
			/* 2: X = ROMix(X) */
			for(int i=0;i<throughput;++i)
				scrypt_ROMix_1((scrypt_mix_word_t *)(Xbuf[cur].ptr + 128 * i), (scrypt_mix_word_t *)Ybuf.ptr, (scrypt_mix_word_t *)Vbuf.ptr, N);

			unsigned int err = 0;
			for(int i=0;i<throughput;++i) {
				unsigned char *ref = (Xbuf[cur].ptr + 128 * i);
				unsigned char *dat = (unsigned char*)(cuda_X[cur] + 32 * i);
				if (memcmp(ref, dat, 128) != 0)
				{
					err++;
#if 0
					uint32_t *ref32 = (uint32_t*) ref;
					uint32_t *dat32 = (uint32_t*) dat;
					for (int j=0; j<32; ++j) {
						if (ref32[j] != dat32[j])
						fprintf(stderr, "ref32[i=%d][j=%d] = $%08x / $%08x\n", i, j, ref32[j], dat32[j]);
					}
#endif
				}
			}
			if (err > 0) fprintf(stderr, "%d out of %d hashes differ.\n", err, throughput);
		}
#endif
		memcpy(Xbuf[cur].ptr, cuda_X[cur], 128 * throughput);

		/* 3: Out = PBKDF2(password, X) */
		for(int i=0;i<throughput;++i)
			scrypt_pbkdf2_1((unsigned char *)&data[cur][20*i], 80, Xbuf[cur].ptr + 128 * i, 128, (unsigned char *)&hash[8*i], 32);

		for(int i=0;i<throughput;++i) {
			volatile unsigned char *hashc = (unsigned char *)&hash[8*i];

			if (hash[8*i+7] <= Htarg && fulltest(&hash[8*i], ptarget)) {

				uint32_t tmp_nonce = nonce[cur]+i;
					
				uint32_t thash[8], tdata[20];
				for(int z=0;z<20;z++) tdata[z] = bswap_32x4(pdata[z]);
				tdata[19] = bswap_32x4(tmp_nonce);
				scrypt_pbkdf2_1((unsigned char *)tdata, 80, (unsigned char *)tdata, 80, Xbuf[cur].ptr + 128 * i, 128);
				scrypt_ROMix_1((scrypt_mix_word_t *)(Xbuf[cur].ptr + 128 * i), (scrypt_mix_word_t *)(Ybuf.ptr), (scrypt_mix_word_t *)(Vbuf.ptr), N);
				scrypt_pbkdf2_1((unsigned char *)tdata, 80, Xbuf[cur].ptr + 128 * i, 128, (unsigned char *)thash, 32);
				if (memcmp(thash, &hash[8*i], 32) == 0)
				{
					*hashes_done = (n-throughput) - pdata[19] + 1;
					pdata[19] = tmp_nonce;
					scrypt_free(&Vbuf);
					scrypt_free(&Ybuf);
					scrypt_free(&Xbuf[0]); scrypt_free(&Xbuf[1]);
					delete[] data[0]; delete[] data[1];
					delete[] hash;
					gettimeofday(tv_end, NULL);
					return 1;
				}
				else
				{
					applog(LOG_INFO, "GPU #%d: %s result does not validate on CPU (i=%d, s=%d)!", device_map[thr_id], device_name[thr_id], i, cur);
				}
			}
		}
		cur = (cur+1)&1; nxt = (nxt+1)&1;
	} while ((n-throughput) < max_nonce && !work_restart[thr_id].restart);
	
	scrypt_free(&Vbuf);
	scrypt_free(&Ybuf);
	scrypt_free(&Xbuf[0]); scrypt_free(&Xbuf[1]);
	delete[] data[0]; delete[] data[1];
	delete[] hash;
	
	*hashes_done = (n-throughput) - pdata[19] + 1;
	pdata[19] = (n-throughput);
	gettimeofday(tv_end, NULL);
	return 0;
}
