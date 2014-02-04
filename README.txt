
CudaMiner release February 4th 2014 - beta release
---------------------------------------------------

***************************************************************
If you find this tool useful and like to support its continued 
          development, then consider a donation.

  LTC donation address: LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
  BTC donation address: 16hJF5mceSojnTD3ZTUDqdRhDyPJzoRakM
  YAC donation address: Y87sptDEcpLkLeAuex6qZioDbvy1qXZEj4
***************************************************************

>>> Introduction <<<

This is a CUDA accelerated mining application for most of your
AltCoin mining needs. Your nVidia cards can be very efficient
miners - don't believe the AMD fanboy myth that nVidia cards
suck at mining! We've recently had a watercooled 780Ti break
900 kHash/s at scrypt (N=1024) mining.

This application is currently supporting
1) scrypt mining with N=1024 (LiteCoin and many, many clones)
2) scrypt-jane mining (Yacoin and several clones)
3) scrypt mining with larger N (VertCoin)

You should see a notable speed-up compared to OpenCL based miners.

We're not supporting Quark, ProtoShares (Momentum) or any other
highly specialized "CPU-only" coin. And certainly no BitCoin: 
This train has left the station quite some time ago!


>>> Command Line Interface <<<

This code is based on the pooler cpuminer 2.3.2 release and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use (default is scrypt)
                          scrypt       scrypt Salsa20/8(1024, 1, 1), PBKDF2(SHA2)
                          scrypt:N     scrypt Salsa20/8(N, 1, 1), PBKDF2(SHA2)
                          scrypt-jane  scrypt Chacha20/8(N, 1, 1), PBKDF2(Keccak)
                          scrypt-jane:Coin
                                       Coin must be one of the supported coins.
                          scrypt-jane:Nfactor
                                       scrypt-chacha20/8(2*2^Nfactor, 1, 1)
                          scrypt-jane:StartTime,Nfmin,Nfmax
                                       like above nFactor derived from Unix time.
                          sha256d      SHA-256d (don't use this! No GPU acceleration)
  -o, --url=URL         URL of mining server (default: " DEF_RPC_URL ")
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of processors)
  -r, --retries=N       number of times to retry if a network call fails
                          (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 15)
  -T, --timeout=N       network timeout, in seconds (default: 270)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 5)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
  -q, --quiet           disable per-thread hashmeter output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
  -B, --background      run the miner in the background
      --benchmark       run in offline benchmark mode
  -c, --config=FILE     load a JSON-format configuration file
  -V, --version         display version information and exit
  -h, --help            display this help text and exit


Additional cudaminer specific command line options are:

--no-autotune    disables the built-in autotuning feature for
                 maximizing CUDA kernel efficiency and uses some
                 heuristical guesswork, which might not be optimal.

--devices        [-d] gives a comma separated list of CUDA device IDs
                 to operate on. Device IDs start counting from 0!
                 Alternatively give string names of your card like
                 gtx780ti or gt640#2 (matching 2nd gt640 in the PC).

--launch-config  [-l] specify the kernel launch configuration per device.
                 This replaces autotune or heuristic selection. You can
                 pass the strings "auto" or just a kernel prefix like
                 F or K or T to autotune for a specific card generation
                 or a kernel prefix plus a lauch configuration like F28x8
                 if you know what kernel runs best (from a previous
                 autotune).

--interactive    [-i] list of flags (0 or 1) to enable interactive
                 desktop performance on individual cards. Use this
                 to remove lag at the cost of some hashing performance.
                 Do not use large launch configs for devices that shall
                 run in interactive mode - it's best to use autotune!

--batchsize      [-b] comma separated list of max. scrypt iterations that
                 are run in one kernel invocation. Default is 1024. Best to
                 use powers of 2 here. Increase for better performance in
                 scrypt-jane with high N-factors. Lower for more interactivity
                 of your video display especially when using the interactive
                 mode.

--texture-cache  [-C] list of flags (0 or 1 or 2) to enable use of the 
                 texture cache for reading from the scrypt scratchpad.
                 1 uses a 1D cache, whereas 2 uses a 2D texture layout.
                 Cached operation has proven to be slightly faster than
                 noncached operation on most GPUs.

--single-memory  [-m] list of flags (0 or 1) to make the devices
                 allocate their scrypt scratchpad in a single,
                 consecutive memory block. On Windows Vista, 7/8
                 this may lead to a smaller memory size being used.
                 When using the texture cache this option is implied.

--hash-parallel  [-H] scrypt also has a small SHA256 or Keccak component:
                      0 hashes this single threaded on the CPU.
                      1 to enable multithreaded hashing on the CPU.
                      2 offloads everything to the GPU (default)

--lookup-gap     [-L] values > 1 enable a tradeoff between memory
                 savings and extra computation effort, in order to
                 improve efficiency with high N-factor scrypt-jane
                 coins. Defaults to 1.

--time-limit     Exit the miner after given number of seconds mining.
                 Useful for doing some round robin pool (or worker)
                 hopping from an external controls script.


>>> Examples <<<


Example for Litecoin Mining on coinotron pool with GTX 660 Ti

cudaminer -d gtx660ti -l K28x32 -C 2 -i 0 -o stratum+tcp://coinotron.com:3334 -O workername:password


Example for Yacoin Mining on yac.coinmine.pl pool with GTX 780

cudaminer -s 10 --algo=scrypt-jane -d gtx780 -L 3 -l T9x21 -b 4096 -C 0 -i 1 -m 0 -o stratum+tcp://yac.coinmine.pl:9088 -O workername:password


Example for VertCoin mining on vtcpool.co.uk with GTX 660 Ti (assuming N=2048, changes over time)

cudaminer --algo=scrypt:2048 -d gtx660ti -l K14x32 -C 2 -i 0 -o stratum+tcp://vtcpool.co.uk:3333 -O workername:password


You will have to adjust the -d parameter. In most cases a -d 0 will work
for you. Specifying video cards by name is best when you often swap your
video cards. The device IDs tend to change a lot, whereas the names 
are much more consistent.

If you are not sure what configuration your video card might need, then
leave away the -l option and let cudaminer autotune.

For scrypt-jane coins with high N factor using a lookup gap with values
greater than 1 will likely boost your performance. Best to try  -L 1
first and work your way up.

For solo-mining you typically use -o 127.0.0.1:xxxx where xxxx represents
the RPC portnumber specified in your wallet's .conf file and you have to
pass the same username and password with -O as specified in the wallet's
.conf file. The wallet must also be started with the -server option and
the server flag in the wallet's .conf file set to 1



>>> About CUDA Kernels <<<

CUDA kernels do all the computation. Which one we select and in which
configuration it is run greatly affects performance. The CUDA kernel
launch configurations are given as a character string, e.g. F27x3

             Prefix    Blocks   x   Warps per block

Available kernel prefixes are:
F or f - Fermi and Legacy cards (Compute 1.x and 2.x)
K or k - Kepler cards (Compute 3.0)
T or t - Titan, GTX 780 and GK208 based cards (Compute 3.5)

Upper case kernel prefixes mean high register count kernels.
Lower case kernel prefixes mean low register count kernels.

so F27x3 means: use Fermi kernel with high register count
                run 27 blocks in total
                each block consisting of 3 warps or 96 threads
                (a warp is a group of 32 threads)

You will want to pick lower case letters for scrypt-jane based coins
with a high N-factor (N being 12 and above...) because the performance
can be much better.

If you do not specify a kernel to use, autotune will pick a kernel
that might be best for your hardware and selected algorithm.

You can also override the autotune's automatic kernel selection,
e.g. pass

-l F
or
-l K
or
-l T

in order to autotune the Fermi, Kepler or Titan kernels overriding
the automatic selection.



>>> Table of CUDA Kernels <<<

Different CUDA kernels are identified by their Prefix letter. In some cases
an alias letter is also accepted, to ensure the existing launch configs
from the December 2013/January 2014 development versions are still running.

Prefix  Alias  Compute Req.  Registers   use for
F       L      1.0           64          scrypt & low N-factor scrypt-jane
K       Y      3.0           63          scrypt & low N-factor scrypt-jane
T       Z      3.5           80          scrypt & low N-factor scrypt-jane

f       X      1.0           32          high N-factor scrypt-jane
k              3.0           32          high N-factor scrypt-jane
t              3.5           32          high N-factor scrypt-jane

the old "Legacy" kernel has been replaced with the F kernel, which will also
be faster on Compute 1.0 legacy devices in many cases. Therefore the F kernel
has been compiled to require only Compute 1.0 capability.



>>> scrypt-jane Specifics <<<

scrypt-jane coins are designed to become more memory-hard over time. Some
of the older coins like Yacoin already approach very high N-factors. Yacoin
at the time of writing requires 4 MB per hash at N-factor 14. This means that
a card with 4 GB video RAM can only compute 1024 hashes in parallel. This
is a way too low number to fully occupy all the card's computational
resources.

Hence it is best on these cards to use kernels that run 4 threads per hash.
These are the low register count kernels, all starting with lower case
prefix letters (f, k, t).

Additionally GPUs with enough computational reserves benefit from enabling
the lookup-gap feature with the -L option and passing values > 1. This
cuts the memory use per hash and allows us to run more hashes (threads)
simultaneously. However additional computations have to be made to compensate
for the reduced lookup tables: any missing intermediate values have to be
recomputed "on the fly".

Use -H 2 with any low N-factor (below 12) scrypt-jane coins, otherwise your
CPU performance may be seriously limiting your hash rates.


The following coin parameters have been hardcoded into cudaminer and can
be given as a coin specifier with the --algo=scrypt-jane option

[YBC] YBCoin, [ZZC] ZZCoin, [FEC] FreeCoin, [ONC] OneCoin, [QQC] QQCoin,
[GPL] GoldPressedLatinum, [MRC] MicroCoin, [APC] AppleCoin, [CPR] Copperbars,
[CACH] CacheCoinm, [UTC] UltraCoin, [VEL] VelocityCoin, [ITC] InternetCoin,
[RAD] RadioactiveCoin

e.g. --algo=scrypt-jane:YBC or --algo=scrypt-jane:YBCoin

To mine new coins with different chain start time and minimum and maximum
N-factors, you can pass the parameters to the --algo option like this:

-algo=scrypt-jane:1389196388,4,30



>>> Additional Notes <<<

This code should be running on nVidia GPUs ranging from compute capability
1.0 up to compute capability 3.5. Just don't expect any hashing miracles
from your old clunkers.

Compute 1.0 through 1.3 devices seem to run faster on Windows XP or Linux
because these OS'es use a more efficient driver model.

Some coins mine a bit faster with the 32 bit cudaminer versions, other mine
faster with the 64 bit cudaminer version. If your computer runs a 64 bit OS,
try running both versions and compare the mining speeds!

To see what autotuning does, enable the debug option (-D) switch.
You will get a table of kHash/s for a variety of launch configurations.
You may only want to do this when running on a single GPU, otherwise
the autotuning output of multiple cards will get all mixed up.

Note that mining through N-factor changes is risky and might fail. Typically
a new N factor requires a different kernel launch configuration. I am trying
to compensate for an N-factor increase by doubling the current lookup-gap
value. This will allow the miner to keep working with the same memory buffers,
but the additional lookup-gap requires more computations to be made. It is
best to re-tune your kernel configuration after every N-factor change.



>>> RELEASE HISTORY <<<

  The February 4th release fixes a problem with very apparently incorrect
  autotune measurements and it also repairs the multi-GPU support. So you
  can again use a single cudaminer to drive all your GPUs. It wasn't the
  driver's fault after all, I was sloppy about some initialization of
  constant memory on the GPUs.

  The Febrary 2nd 2014 release supports scrypt-jane for the first time
  and includes faster scrypt kernels kindly submitted by nVidia.
  Most Dave Andersen and nVidia derived kernels now support -C 1 and -C 2
  texture caching options providing a speed benefit in some cases.

  The December 18th 2013 milestone transitions cudaminer to CUDA 5.5, which
  makes it require newer nVidia drivers unfortunately. However users of
  Kepler devices will see a significant speed boost of 30% for Compute 3.0
  devices and around 10% for Compute 3.5 devices. This was made possible
  by David Andersen who posted his more efficient miner code under Apache
  license. This release marks a first step of integrating his work.

   ... some history removed ...

  April 4th 2013 initial release.



>>> TODO <<<

Usability Improvements:
- better CUDA error handling and recovery from errors
- add failover support between different pools
- smarter autotune algorithm
- temperature and GPU utilization control features
- an external API for system monitoring like that of CGMiner

Further Optimization:
- fix a problem in CUDA issue order that prevents overlapping of
  memory transfers and kernel launches (this could bring 5% more
  speed when fixed!)
- check the hashes against target on the GPU, saving memory
  transfers over the PCI-x bus
- reduce the thread divergence in the lookup-gap feature by
  sorting threads by loop trip count (requires a swapping of
  the thread state in the entire thread block)
- allow for more optimized Keccak or SHA2 implementations for
  specific hardware (like Compute 3.5 using the funnel shifter)
- make a direct port of the nv_kernel.cu to Fermi, using warp
  shuffle emulation with shared memory.


>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner (Germany): original CUDA implementation

Alexis Provos (Greece): submitted a faster Salsa 20/8 round function.

David Andersen (USA, Carnegie Mellon University): designed a low
                        register count scrypt Kepler kernel with
                        improved memory access. Currently best
                        performing with high N-factor scrypt-jane.

Alexey Panteleev (USA, nVidia): submitted a kernel with improved
                        memory access functions for Kepler devices 
                        providing the fastest scrypt performance.

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler) !


Source code is included to satisfy GNU GPL V2 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
