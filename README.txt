
CudaMiner release April 22th 2013 - alpha release
-------------------------------------------------

this is a CUDA accelerated mining application for litecoin only.
The most computationally heavy parts of the scrypt algorithm (the
Salsa 20/8 iterations) are run on the GPU.

You should see a notable speed-up compared to OpenCL based miners.
Some numbers from my testing:

GTX 260:    44  kHash/sec  (OpenCL: 20)
GTX 640:    39  kHash/sec
GTX 460:   101  kHash/sec
GTX 560Ti: 140  kHash/sec
GTX 660Ti: 156  kHash/sec  (OpenCL: 60-70)

NOTE: Compute 1.0 through 1.3 devices seem to run faster on Windows XP
or Linux.

Your nVidia cards will now suck a little less for mining! This tool
will automatically use all nVidia GPUs found in your system, but the
used device count can be limited to a lower number using the "-t"
option, or even selected individually with the "-d" option

This code is based on the pooler cpuminer 2.2.3 release and inherits
its command line interface and options.

Additional command line options are:

--no-autotune    disables the built-in autotuning feature for
                 maximizing CUDA kernel efficiency and uses some
                 heuristical guesswork, which might not be optimal.

--devices        [-d] gives a list of CUDA device IDs to operate on.
                 Device IDs start counting from 0!

--launch-config  [-l] specify the kernel launch configuration per device.
                 This replaces autotune or heuristic selection.

--interactive    [-i] list of flags (0 or 1) to enable interactive
                 desktop performance on individual cards. Use this
                 to remove lag at the cost of some hashing performance.
                 Do not use large launch configs for devices that shall
                 run in interactive mode - it's best to use autotune!

--texture-cache  [-C] list of flags (0 or 1 or 2) to enable use of the 
                 texture cache for reading from the scrypt scratchpad.
                 1 uses a 1D cache, whereas 2 uses a 2D texture layout.
                 This is very experimental and may hurt performance
                 on some cards.

--single-memory  [-m] list of flags (0 or 1) to make the devices
                 allocate their scrypt scratchpad in a single,
                 consecutive memory block. On Windows Vista, 7/8
                 this may lead to a smaller memory size being used.


>>> Example command line options, advanced use <<<

cudaminer.exe -d 0,1,2 -i 1,0,0 -l auto,S27x3,28x4 -C 0,2,1
-o http://ltc.kattare.com:9332 -O myworker.1:mypass

I tell cudaminer to use devices 0,1 and 2. Because I have the monitor
attached to device 0, I set that device to run in interactive mode so
it is fully responsive for desktop use while mining.

Device 0 performs autotune for interactive mode because I explicitly
set it to auto. Device 1 will use kernel launch configuration S27x3 and
device 2 uses 28x4.

I turn on the use of the texture cache to 2D for device 1, 1D for device
2 and off for the other devices.

The given -o/-O settings mine on Burnside's pool, on which I happen to have
an account.


>>> Additional Notes <<<

The HMAC SHA-256 parts of scrypt are still executed on the CPU, and so
any BitCoin mining will NOT be GPU accelerated. This tool is for LTC.

This does not support the Stratum protocol. To do stratum mining
you have to run a local proxy.

This code should be fine on nVidia GPUs ranging from compute
capability 1.1 up to compute capability 3.5. The Geforce Titan has
received experimental and untested support.

To see what autotuning does, enable the debug option (-D) switch.
You will get a table of kHash/s for a variety of launch configurations.
You may only want to do this when running on a single GPU, otherwise
the autotuning output of multiple cards will mix.


>>> RELEASE HISTORY <<<

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
launch configurations are given as a character string, e.g. S27x3

                       prefix blocks x warps

Currently there is just one prefix, which is "S". Later releases may
see the introduction of more kernel variants with using other letters.

Examples:

e.g. S27x3 is a launch configuration that works well on GTX 260
      28x4 is a launch configuration that works on Geforce GTX 460
     290x2 is a launch configuration that works on Geforce GTX 660Ti

You should wait through autotune to see what kernel is found best for
your current hardware configuration.

The choice between Non-Titan and Titan CUDA kernels is automatically
made based on your device's compute capability. Titans cost around
a thousand dollars, so you probably don't have one.


Prefix  | Non-Titan          | Titan
-------------------------------------------------------
 <none> | low shared memory  | default kernel
        | optimized kernel   | with funnel shifter
        |                    |
   S    | special kernel     | spinlock kernel
        | for older GPUs     | with funnel shifter


>>> TODO <<<

Usability Improvements:
- add reasonable error checking for CUDA API calls
- add Stratum support
- add failover support

Further Optimization:
- consider use of some inline assembly in CUDA
- investigate benefits of a LOOKUP_GAP implementation


***************************************************************
If you find this tool useful and like to support its continued 
        development, then consider a donation in LTC.

  The donation address is LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
***************************************************************

Source code is included to satisfy GNU GPL V2 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
