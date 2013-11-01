
CudaMiner release October 10th 2013 - alpha release
---------------------------------------------------

this is a CUDA accelerated mining application for litecoin only.
The most computationally heavy parts of the scrypt algorithm (the
Salsa 20/8 iterations) are run on the GPU.

You should see a notable speed-up compared to OpenCL based miners.
Some numbers from my testing:

GTX 260:    44  kHash/sec (maybe more on Linux/WinXP)
GTX 640:    40  kHash/sec
GTX 460:   100  kHash/sec
GTX 560Ti: 140  kHash/sec
GTX 660Ti: 176  kHash/sec (or 225 kHash/sec on the 448 core edition)

NOTE: Compute 1.0 through 1.3 devices seem to run faster on Windows XP
or Linux.

Your nVidia cards will now suck a little less for mining! This tool
will automatically use all nVidia GPUs found in your system, but the
used device count can be limited to a lower number using the "-t"
option, or even selected individually with the "-d" option

This code is based on the pooler cpuminer 2.3.2 release and inherits
its command line interface and options.

Additional command line options are:

--no-autotune    disables the built-in autotuning feature for
                 maximizing CUDA kernel efficiency and uses some
                 heuristical guesswork, which might not be optimal.

--devices        [-d] gives a list of CUDA device IDs to operate on.
                 Device IDs start counting from 0!

--launch-config  [-l] specify the kernel launch configuration per device.
                 This replaces autotune or heuristic selection. You can
                 pass the strings "auto" or just a kernel prefix like
                 L or F or K or T to autotune for a specific card generation
                 or a kernel prefix plus a lauch configuration like F28x8
                 if you know what kernel runs best (from a previous autotune).

--interactive    [-i] list of flags (0 or 1) to enable interactive
                 desktop performance on individual cards. Use this
                 to remove lag at the cost of some hashing performance.
                 Do not use large launch configs for devices that shall
                 run in interactive mode - it's best to use autotune!

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

--hash-parallel  [-H] 1 to enable parallel hashing on the CPU. May
                 use more CPU but distributes hashing load neatly
                 across all CPU cores. Use 0 otherwise, which is now
                 the default.

>>> Example command line options, advanced use <<<

cudaminer.exe -H 1 -d 0,1,2 -i 1,0,0 -l auto,F27x3,K28x4 -C 0,2,1
-o stratum+tcp://coinotron.com:3334 -O workername:password  

The option -H 1 distributes the CPU load across all available cores.

I instruct cudaminer to use devices 0,1 and 2. Because I have the display
attached to device 0, I set that device to run in interactive mode so
it is fully responsive for desktop use while mining.

Device 0 performs autotune and runs in interactive mode because I explicitly
set the launch config to auto and the corresponding interactive flag is 1.
Device 1 will use kernel launch configuration F27x3 (for Fermi) and device 2
uses K28x4 (for Kepler) both in non-interactive mode.

I turn on the use of the texture cache to 2D for device 1, 1D for device
2 and off for devices 0.

The given -o/-O settings mine on the coinotron pool using the stratum
protocol.


>>> Additional Notes <<<

The HMAC SHA-256 parts of scrypt are still executed on the CPU, and so
any BitCoin mining will NOT be GPU accelerated. This tool is for LTC.

There is also some SHA256 hashing required to do Scrypt hashing, and
this part is also done by the CPU, in parallel to the work done by
the GPU(s).

This code should be fine on nVidia GPUs ranging from compute
capability 1.1 up to compute capability 3.5. The Geforce Titan has
received experimental and untested support.

To see what autotuning does, enable the debug option (-D) switch.
You will get a table of kHash/s for a variety of launch configurations.
You may only want to do this when running on a single GPU, otherwise
the autotuning output of multiple cards will mix.


>>> RELEASE HISTORY <<<

- the November 1st release finally fixes the stratum protocol
  hang for good. Root cause analysis: The ssize_t didn't wasn't
  a signed type in my Windows port, causing the stratum_send_line
  function to enter an infinite loop whenever the connection was
  lost, while holding the socket mutex.

- the October 10th release may fix some infrequent hanging in
  the stratum protocol. Or maybe not. Please test.

  I also turned of the parallel SHA256 computations on the CPU
  because they seem to load the CPU a little more in most cases.
  use -H 1 to get the previous behavior.

- the July 13th release adds support for the Stratum protocol,
  by making a fresh fork of pooler's cpuminer code (and any future
  updates of pooler's code will be easier to integrate).

- the April 30th release fixes a minor problem in the CPU SHA-256
  parallelization that might have lead to inflated CPU use.

  Modified the CUDA API command issue order to get 99-100%
  utilization out of my Kepler card on Windows.

  The old "S" kernels have been replaced with kernels that seem
  to slightly improve performance on Kepler cards. Just prepend
  your previous Kepler launch config (e.g. 28x8) with an S prefix
  to see if you get any performance gains. Works for me! ;)

- the April 22th release fixes Linux 64 bit compilation and reintroduces
  memory access optimizations in the Titan kernel.

- the April 17th release fixes the texture cache feature (yay!) but
  the even Kepler cards currently see no real benefits yet (boo!).

  Ctrl-C will now also interrupt the autotuning loop, and pressing
  Ctrl-C a second time will always result in a hard exit.

  The Titan kernel was refactored into a write-to-scratchpad phase and
  into a read-from-scratchpad case using const __restrict__ pointers,
  which makes the Titan automatically use the 48kb texture cache in each
  SMX during the read phase. No need to use the -C flag with Titan.

  CPU utilization seems lower than in previous releases, especially in
  interactive mode. In fact I barely see cudaminer.exe consuming CPU
  resources all ;)

- the April 14th release lowers the CPU use dramatically. I also fixed the
  Windows specific driver crash on CTRL-C problem. You still should not
  click the close button on the DOS box, as this does not leave the
  program enough time for cleanly shutting down.

- the April 13th release turns the broken texture cache feature OFF by
  default, as now also seems detrimental to performance. So what remains of
  yesterday's update is just the interactive mode and the restored
  Geforce Titan support.

  I also added a validation of GPU results by the CPU.

- the April 12th update boosts Kepler performance by 15-20% by enabling
  the texture cache on these devices to do its scrypt scratchpad lookups.
  You can also override the use of the texture cache from command line.

  I also add an interactive mode for cards that drive monitors, so you
  can be almost lag-free when using the desktop. It costs some performance
  though. In interactive mode autotuning, smaller kernel launch configs
  are selected. Try not to override this with huge launch configs, or the
  effect of interactive mode would be negated.  

  Put Titan support back to its original state. I suspect that a CUDA
  compiler bug made the kernel crash when I applied the same optimizations
  that work so nicely on Compute 1.0 trough 3.0 devices.

- the April 10th update speeds up the CUDA kernels SIGNIFICANTLY by using
  larger memory transactions (yay!!!)

- the April 9th update fixes an autotune problem and adds Linux autotools
  support.

- the April 8th release adds CUDA kernel optimizations that may get up to
  20% more kHash out of newer cards (Fermi generation and later...).

  It also adds UNTESTED Geforce Titan support.

  I also use Microsoft's parallel patterns library to split up the CPU
  HMAC SHA256 workload over several CPU cores. This was a limiting factor
  for some GPUs before.

- the April 6th release adds an auto-tuning feature that determines the
  best kernel launch configuration per GPU. It takes up to a few minutes
  while the GPU's memory and host CPU may be pegged a bit. You can disable
  this tuning with the --no-autotune switch

- April 4th initial release.


>>> About CUDA Kernels <<<

CUDA kernels do the computation. Which one we select and in which
configuration it is run greatly affects performance. CUDA kernel
launch configurations are given as a character string, e.g. F27x3

                       prefix blocks x warps

Available kernel prefixes are:
L - Legacy cards (compute 1.x)
F - Fermi cards (Compute 2.x)
K - Kepler cards (Compute 3.0). The letter S (for "spinlock") also works
T - Titan and GK208 based cards (Compute 3.5)

Examples:

e.g. L27x3 is a launch configuration that works well on GTX 260
     F28x4 is a launch configuration that works on Geforce GTX 460
     K290x2 is a launch configuration that works on Geforce GTX 660Ti

You should wait through autotune to see what kernel is found best for
your current hardware configuration. You can also override the autotune's
automatic device generation selection, e.g. pass

-l F
or
-l K
or
-l T

in order to autotune the Fermi kernel on a Legacy, Kepler or Titan device

>>> TODO <<<

Usability Improvements:
- add reasonable error checking for CUDA API calls
- a compiled 64 bit version also for Windows
- add failover support between different pools
- investigate why on some machine the legacy kernel fails,
  and on other machines the Fermi kernel fails.

Further Optimization:
- consider use of some inline assembly in CUDA
- investigate benefits of a LOOKUP_GAP implementation
- optimize more for compute 3.5 devices like newer GT640 cards
  and the Geforce Titan.


***************************************************************
If you find this tool useful and like to support its continued 
        development, then consider a donation in LTC.

  The donation address is LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
***************************************************************

Source code is included to satisfy GNU GPL V2 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
