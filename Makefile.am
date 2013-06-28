if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autotools.sh configure.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters cudaminer.vcxproj.user \
			  compat/gettimeofday.c compat/getopt/getopt_long.c

SUBDIRS			= compat

INCLUDES		= $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES)

bin_PROGRAMS		= cudaminer

cudaminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  cuda-miner.cpp util.cpp \
			  sha2.cpp \
			  scrypt.cpp \
			  salsa_kernel.cu \
			  spinlock_kernel.cu \
			  titan_kernel.cu

cudaminer_LDFLAGS	= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
cudaminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
cudaminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@

.cu.o:
	$(NVCC) @CFLAGS@ -arch=compute_10 --maxrregcount=124 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

spinlock_kernel.o: spinlock_kernel.cu
	$(NVCC) @CFLAGS@ -arch=compute_11 --maxrregcount=124 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

titan_kernel.o: titan_kernel.cu
	$(NVCC) @CFLAGS@ -arch=compute_35 --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<
