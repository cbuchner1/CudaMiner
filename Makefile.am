
if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS		= compat

bin_PROGRAMS	= cudaminer

cudaminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  cpu-miner.c util.c \
			  wrapnvml.cu wrapnvml.h \
			  sha2.c \
			  scrypt.cpp \
			  maxcoin.cpp \
			  blakecoin.cpp \
			  sha3.cpp sha3.h \
			  scrypt-jane.cpp scrypt-jane.h \
			  salsa_kernel.cu salsa_kernel.h \
			  sha256.cu sha256.h \
			  keccak.cu keccak.h \
			  blake.cu blake.h blake.c sph_blake.h sph_types.h \
			  fermi_kernel.cu fermi_kernel.h \
			  kepler_kernel.cu kepler_kernel.h \
			  test_kernel.cu test_kernel.h \
			  nv_kernel.cu nv_kernel.h \
			  nv_kernel2.cu nv_kernel2.h \
			  titan_kernel.cu titan_kernel.h

cudaminer_LDFLAGS	= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
cudaminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ -ldl
cudaminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

.cu.o:
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_10 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

sha256.o: sha256.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_10 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

keccak.o: keccak.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_10 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

blake.o: blake.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_10 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

# NOTE: now compiling for compute 1.0 again, as it's using less power and runs way faster on Linux
fermi_kernel.o: fermi_kernel.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_10 --maxrregcount=64 $(JANSSON_INCLUDES) -o $@ -c $<

kepler_kernel.o: kepler_kernel.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=sm_30 --maxrregcount=32 $(JANSSON_INCLUDES) -o $@ -c $<

titan_kernel.o: titan_kernel.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=sm_35 --maxrregcount=32 $(JANSSON_INCLUDES) -o $@ -c $<

# NOTE: now compiling for compute 1.0 again, as it's using less power and runs way faster on Linux
test_kernel.o: test_kernel.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=compute_10 --maxrregcount=32 $(JANSSON_INCLUDES) -o $@ -c $<

nv_kernel.o: nv_kernel.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=sm_30 --maxrregcount=63 $(JANSSON_INCLUDES) -o $@ -c $<

nv_kernel2.o: nv_kernel2.cu
	$(NVCC) @CFLAGS@ -Xptxas "-abi=no -v" -arch=sm_35 --maxrregcount=80 $(JANSSON_INCLUDES) -o $@ -c $<
