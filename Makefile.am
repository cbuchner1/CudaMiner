
if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS		= compat

INCLUDES	= $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES)

bin_PROGRAMS	= cudaminer

cudaminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  cpu-miner.c util.c \
			  sha2.c \
			  scrypt.cpp \
			  salsa_kernel.cu salsa_kernel.h \
			  spinlock_kernel.cu spinlock_kernel.h \
			  legacy_kernel.cu legacy_kernel.h \
			  fermi_kernel.cu fermi_kernel.h \
			  titan_kernel.cu titan_kernel.h

cudaminer_LDFLAGS	= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
cudaminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
cudaminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@

.cu.o:
	$(NVCC) @CFLAGS@ -arch=compute_10 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

spinlock_kernel.o: spinlock_kernel.cu
	$(NVCC) @CFLAGS@ -arch=compute_11 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

titan_kernel.o: titan_kernel.cu
	$(NVCC) @CFLAGS@ -arch=compute_35 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<
