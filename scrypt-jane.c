/*
	scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane

	Public Domain or MIT License, whichever is easier
*/

#include "cpuminer-config.h"
#include "miner.h"

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
const unsigned char minNfactor = 4;
const unsigned char maxNfactor = 30;

unsigned char GetNfactor(unsigned int nTimestamp) {
    int l = 0;

    if (nTimestamp <= 1367991200)
        return 4;

    unsigned long int s = nTimestamp - 1367991200;
    while ((s >> 1) > 3) {
      l += 1;
      s >>= 1;
    }

    s &= 3;

    int n = (l * 170 + s * 25 - 2320) / 100;

    if (n < 0) n = 0;

    if (n > 255)
        printf("GetNfactor(%d) - something wrong(n == %d)\n", nTimestamp, n);

    unsigned char N = (unsigned char)n;
    //printf("GetNfactor: %d -> %d %d : %d / %d\n", nTimestamp - nChainStartTime, l, s, n, min(max(N, minNfactor), maxNfactor));

//    return min(max(N, minNfactor), maxNfactor);

    if(N<minNfactor) return minNfactor;
    if(N>maxNfactor) return maxNfactor;
    return N;
}

int scanhash_scrypt_jane(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t data[20], hash[8], target_swap[8];
        volatile unsigned char *hashc = (unsigned char *) hash;
        volatile unsigned char *datac = (unsigned char *) data;
        volatile unsigned char *pdatac = (unsigned char *) pdata;
	uint32_t n = pdata[19] - 1;
	scrypt_aligned_alloc YX, V;
	uint8_t *X, *Y;
	uint32_t N, chunk_bytes;
	const uint32_t r = SCRYPT_R;
	const uint32_t p = SCRYPT_P;
	int i;
	
#if !defined(SCRYPT_TEST)
	static int power_on_self_test = 0;
	if (!power_on_self_test) {
		power_on_self_test = 1;
		if (!scrypt_power_on_self_test())
			scrypt_fatal_error("scrypt: power on self test failed");
	}
#endif

        /* byte swap it */
        for(int z=0;z<20;z++) {
            datac[(z*4)  ] = pdatac[(z*4)+3];
            datac[(z*4)+1] = pdatac[(z*4)+2];
            datac[(z*4)+2] = pdatac[(z*4)+1];
            datac[(z*4)+3] = pdatac[(z*4)  ];
        }

	int Nfactor = GetNfactor(data[17]);
	if (Nfactor > scrypt_maxN) {
		scrypt_fatal_error("scrypt: N out of range");
	}
	
	N = (1 << (Nfactor + 1));
	
	chunk_bytes = SCRYPT_BLOCK_BYTES * r * 2;
	V = scrypt_alloc((uint64_t)N * chunk_bytes);
	YX = scrypt_alloc((p + 1) * chunk_bytes);
	
	Y = YX.ptr;
	X = Y + chunk_bytes;

	do {
		data[19] = ++n;

		scrypt_N_1_1((unsigned char *)data, 80, 
                       (unsigned char *)data, 80, 
                       N, (unsigned char *)hash, 32, X, Y, V.ptr);

		if (hashc[31] == 0 && hashc[30] == 0) {

#if 0
                    for(int z=7;z>=0;z--)
                       fprintf(stderr, "%08x ", hash[z]);
                    fprintf(stderr, "\n");

                    for(int z=7;z>=0;z--)
                       fprintf(stderr, "%08x ", ptarget[z]);
                    fprintf(stderr, "\n");
#endif

                    if(fulltest(hash, ptarget)) {
			*hashes_done = n - pdata[19] + 1;
			pdatac[76] = datac[79];
                        pdatac[77] = datac[78];
                        pdatac[78] = datac[77];
                        pdatac[79] = datac[76];
			
			scrypt_free(&V);
			scrypt_free(&YX);
			return 1;
                   }
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);
	
	scrypt_free(&V);
	scrypt_free(&YX);
	
	*hashes_done = n - pdata[19] + 1;
	pdata[19] = n;
	return 0;
}
