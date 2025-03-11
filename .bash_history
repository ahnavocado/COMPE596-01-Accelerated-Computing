ls
pwd
htop
btop
snap install btop
sudo snap install btop
ls
mkdir workspace
cd workspace/
ls
mkdir 250124
cd 250124/
ls
vim hi.c
ort OMP_NUM_THREADS=8
exoort OMP_NUM_THREADS=8
export OMP_NUM_THREADS=8
./hi
gcc -fopenmp hi.c -o hi
ls
unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./hi)
(unset OMP_NUM_THREADS; export OMP_NUM_THREADS=4; ./hi)
ls
cd workspace/
mkdir 250127
cd 25012
cd 250127
ls
nvtop
htop
ls
mkdir P01
cd P
cd P01/
ls
vim test_0.c
gcc -fopenmp test_0.c -o test_0
export OMP_NUM_THREADS=8
./test_0 
vim test_0
vim test_0.c 
gcc -fopenmp test_0.c -o test_0
./test_0 
vim test_0.c 
gcc -fopenmp test_0.c -o test_0
./test_0 
ls
cd workspace/
ls
htop
cd P01/
ls
./test_0 
vim test_0.c 
gcc -fopenmp test_0.c -o test_0
./test_0 
ls
cd workspace/
l
cd P01/
ls
vim final.c
gcc -fopenmp final.c -o final
./final 
vim final.c
gcc -fopenmp final.c -o final
./final 
vim final.c
gcc -fopenmp final.c -o final
./final 
htop
nvtop
btop
ls
clear
btop
#grep 'physical id' /proc/cpuinfo | uniq | wc -l
#cat /proc/cpuinfo
grep 'physical id' /proc/cpuinfo | uniq | wc -l
grep 'cpu cores' /prco/cupinfo | uniq 
grep 'cpu cores' /prco/cpuinfo | uniq 
grep 'cpu cores' /proc/cpuinfo | uniq 
htop
top
./final 
htop
ls
cd workspace/
ls
mkdir P02
cd P02
ls
cd workspace/
ls
cd P02
ls
vim Makefile
cat Makefile 
vim Node_single.c
vim Node.c
ls
vim Node.h
vim Node_single.c 
vim Node.c
ls
make run1    # Runs with 1 thread
make run2    # Runs with 2 threads
make run4    # Runs with 4 threads
make run16   # Runs with 16 threads
make run_s1  # Runs serial with 1 thread
make
date
touch Makefile Node_single.c Node.h Node.c
make clean
make
make clean .
vim Makefile 
make clean 
vim Makefile make run1    # Runs with 1 thread
make run1    # Runs with 1 thread
make run2    # Runs with 2 threads
make run4    # Runs with 4 threads
make run16   # Runs with 16 threads
ls
vim Makefile 
make benchmark_serial
ls
cd workspace/
ls
cd P0
cd P02
ls
make benchmark_serial
ls
cat Makefile 
ls
vim Node.c 
vim Node_single.c 
vim Makefile 
make benchmark_serial
vim Makefile 
make benchmark_serial
ls
vim Node.c
cd ../P01
ls
cat final.c 
ls
./final 
htop
ls
cd ../P02
ls
vim Node.c
cat Node.c
cat Node_single.c 
ls
vim Node.c
cat Makefile 
./run1
run run1
cat Makefile 
more Makefile 
make run1
vim Node.c 
make run1
vim Node.c 
htop
ls
cd workspace/P02
ls
cat Node.c
cat Node_single.c 
cat Node.h
cal
make run_parallel
vim Makefile 
ls
vim Makefile 
make run_parallel
vim Node.c
make run_parallel
make
make run_parallel
vim Makefile 
cat Makefile 
vim Makefile 
make
make run1    # 1개 스레드 실행
make run2    # 2개 스레드 실행
make run4    # 4개 스레드 실행
make run8    # 8개 스레드 실행
make run16   # 16개 스레드 실행
make run32   # 32개 스레드 실행
make run64   # 64개 스레드 실행
make run128  # 128개 스레드 실행
make run256  # 256개 스레드 실행
vim Makefile 
make run256  # 256개 스레드 실행make run1    # 1개 스레드 실행
make run2    # 2개 스레드 실행
make run4    # 4개 스레드 실행
make run8    # 8개 스레드 실행
make run16   # 16개 스레드 실행
make run32   # 32개 스레드 실행
make run64   # 64개 스레드 실행
make run128  # 128개 스레드 실행
make run256  # 256개 스레드 실행
make
make run1    # 1개 스레드 실행
make run2    # 2개 스레드 실행
make run4    # 4개 스레드 실행
make run8    # 8개 스레드 실행
make run16   # 16개 스레드 실행
make run32   # 32개 스레드 실행
make run64   # 64개 스레드 실행
make run128  # 128개 스레드 실행
make run256  # 256개 스레드 실행
make
vim Makefile 
cat Node.c
export OMP_NUM_THREADS=256
make
make run_parallel
ls
cd workspace/P02
ls
cat Makefile 
vim Node.c
make benchmark_serial
vim Node_single.c 
make benchmark_serial
ls
vim Makefile 
vim Node.c
vim Node_single.c 
vim Node.c
vim Makefile 
make run_serial
make run_parallel
ls
chmod +x node_single node_parallel
chmod +x node_single Node
chmod +x node_single node
ls
make run_parallel
make run_serial
vim Makefile 
make
make run_serial
ls
cd workspace/P02/
ls
vim Node.c
cat Node.c
ls
vim Node_auto.c
vim Makefile 
ls
vim Makefile 
cat Node_single.c 
vim node_single.c
vim Node_single.c 
gcc -Wall -o node_serial Node_serial.c
gcc -Wall -o node_serial Node_single.c
./node_serial
ls
ls
cd workspace/P02/
ls
make
vim Makefile 
make run_parallel 
export OMP_NUM_THREADS=4  && ./node_parallel
export OMP_NUM_THREADS=8  && ./node_parallel
export OMP_NUM_THREADS=16 && ./node_parallel
export OMP_NUM_THREADS=32 && ./node_parallel
export OMP_NUM_THREADS=4  && ./node
export OMP_NUM_THREADS=8  && ./node
export OMP_NUM_THREADS=16 && ./node
export OMP_NUM_THREADS=32 && ./node
vim Node.c
cat Node.c
vim Node.c
gcc -Wall -fopenmp -o node_parallel Node.c
vim Node.c
gcc -Wall -fopenmp -o node_parallel Node.c
vim Node.c
gcc -Wall -fopenmp -o node_parallel Node.c
./node_parallel
export OMP_NUM_THREADS=8 && ./node_parallel
export OMP_NUM_THREADS=256 && ./node_parallel
cat Makefile 
vim Makefile 
cat Makefile make run_parallel
make run_parallel
make
make run_parallel
vim Makefile 
make
make run_parallel
cat Makefile
cat Node.c
vim Node_single.c 
cat Node_single.c 
lstopo
ls
cd workspace/
l
mkdir P03
ls
cd P03
ls
l
ls
cd workspace/
ls
cd P03
clear
ls
cd workspace/P03
l
ls
vim imp_0.c
gcc -fopenmp imp_0.c -o imp_0
vim imp_0.c
gcc -fopenmp imp_0.c -o imp_0
vim imp_0.c 
gcc -fopenmp imp_0.c -o imp_0
vim imp_0.c 
gcc -fopenmp imp_0.c -o imp_0
ls
vim imp_0.c 
gcc -fopenmp imp_0.c -o imp_0 -lm
ls
./imp_0 
vim imp_0.c 
gcc -fopenmp imp_0.c -o imp_0 -lm
./imp_0 
vim imp_0.c 
gcc -fopenmp imp_0.c -o imp_0 -lm
./imp_0 
vim imp_0.c 
gcc -fopenmp imp_0.c -o imp_0 -lm
./imp_0 
vim orign_0.c
gcc -fopenmp orign_0.c -o origin_0 -lm
./origin_0 
ls
vim orign_0.c 
gcc -fopenmp orign_0.c -o orign_0 -lm
gcc -fopenmp orign_0.c -o origin_0 -lm
vim orign_0.c 
gcc -fopenmp orign_0.c -o orign_0 -lm
export OMP_NUM_THREADS=2
./orign_0 
./orign_0 2
./orign_0 8
rm -rf origin_0 
vim orign_0.c 
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 8
ls
cd workspace/P03
ls
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define NTHRDS 8  // OpenMP 스레드 개수
#define N 1000000  // 구간 개수 (짝수여야 함)
// 적분할 함수 정의
double f(double x) {
}
int main() {
}
ls
cat orign_0.c
./orign_0 
ls
./orign_0 4
./origin_0 
ls
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
./orign_0 4
vim orign_0.c
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
ls
vim orign_0.c 
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
vim orign_0.c 
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
./imp_0 
vim orign_0.c 
gcc -fopenmp orign_0.c -o orign_0 -lm
vim orign_0.c
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
./imp_0 
cat imp_0.c
vim orign_0.c
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
./orign_0 8
vim orign_0.c
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
./orign_0 8
vim orign_0.c
gcc -fopenmp orign_0.c -o orign_0 -lm
./orign_0 
./orign_0 16
vim orign_1.c
gcc -fopenmp orign_1.c -o orign_1 -lm
./orign_1
./orign_1 16 100000000
./orign_1 64 1000000000
./orign_1 64 10000000000
./orign_1 128 10000000000
./orign_1 128 1000000
./orign_1 128 10000000
./orign_1 128 100000000
./orign_1 128 1000000000
./orign_1 128 10000000000
./orign_1 128 100000000000
clear
./orign_1 128 100
./orign_1 128 1000
./orign_1 128 10000
./orign_1 128 100000
./orign_1 128 1000000
./orign_1 128 10000000
./orign_1 128 100000000
./orign_1 128 1000000000
./orign_1 128 10000000000
./orign_1 128 100000000000
ls
./orign_1 128 10
./orign_1 128 100
./orign_1 128 1000
./orign_1 128 10000
./orign_1 128 100000
./orign_1 128 1000000
./orign_1 128 10000000
./orign_1 128 100000000
./orign_1 128 1000000000
./orign_1 128 10000000000
./orign_1 128 100000000000
./orign_1 128 1000000000000
vim orign_2.c
gcc -fopenmp orign_2.c -o orign_2 -lm
gcc clear
gcc -fopenmp orign_2.c -o orign_2 -lm
ls
vim orign_2.c
gcc -fopenmp orign_2.c -o orign_2 -lm
./orign_2
./orign_2 128 10
./orign_2 128 100
./orign_2 128 1000
./orign_2 128 10000
./orign_2 128 100000
./orign_2 128 1000000
./orign_2 128 10000000
./orign_2 128 100000000
./orign_2 128 1000000000
./orign_2 128 10000000000
./orign_2 128 100000000000
./orign_2 128 10
./orign_2 128 100
./orign_2 128 1000
./orign_2 128 10000
./orign_2 128 100000
./orign_2 128 1000000
./orign_2 128 10000000
./orign_2 128 100000000
./orign_2 128 1000000000
./orign_2 128 10000000000
./orign_2 128 100000000000
./orign_1 128 10
./orign_1 128 100
./orign_1 128 1000
./orign_1 128 10000
./orign_1 128 100000
./orign_1 128 1000000
./orign_1 128 10000000
./orign_1 128 100000000
./orign_1 128 1000000000
./orign_1 128 10000000000
./orign_1 128 10
./orign_1 128 100
./orign_1 128 1000
./orign_1 128 10000
./orign_1 128 100000
./orign_1 128 1000000
./orign_1 128 10000000
./orign_1 128 100000000
./orign_1 128 1000000000
./orign_1 128 10000000000
./orign_1 128 100000000000
./orign_2 128 10
./orign_2 128 100
./orign_2 128 1000
./orign_2 128 10000
./orign_2 128 100000
./orign_2 128 1000000
./orign_2 128 10000000
./orign_2 128 100000000
./orign_2 128 1000000000
./orign_2 128 10000000000
htop
./orign_2 128 10
./orign_2 128 100
./orign_2 128 1000
./orign_2 128 10000
./orign_2 128 100000
./orign_2 128 1000000
./orign_2 128 10000000
./orign_2 128 100000000
./orign_2 128 1000000000
./orign_2 128 10000000000
./orign_2 128 10
./orign_2 128 100
./orign_2 128 1000
./orign_2 128 10000
./orign_2 128 100000
./orign_2 128 1000000
./orign_2 128 10000000
./orign_2 128 100000000
./orign_2 128 1000000000
./orign_2 128 10000000000
./orign_2 128 100000000000
./orign_2 128 1000000000000
./orign_2 1 1000000
./orign_2 2 1000000
./orign_2 4 1000000
./orign_2 8 1000000
./orign_2 16 1000000
./orign_2 32 1000000
./orign_2 64 1000000
./orign_2 128 1000000
./orign_2 256 1000000
./orign_2 1 10000000
./orign_2 2 10000000
./orign_2 4 10000000
./orign_2 8 10000000
./orign_2 16 10000000
./orign_2 32 10000000
./orign_2 64 10000000
./orign_2 128 10000000
./orign_2 256 10000000
./orign_2 1 100000000
./orign_2 2 100000000
./orign_2 4 100000000
./orign_2 8 100000000
./orign_2 16 100000000
./orign_2 32 100000000
./orign_2 64 100000000
./orign_2 128 100000000
./orign_2 256 100000000
cat orign_2.c
cat orign_0
cat orign_0.c
w
module load nvhpc/22.11
cuda-memcheck
cuda-memcheck  lspci | grep -i nvidia
uname -m && cat /etc/*release
uname -r
ls
pwd
cd workspace/
pwd
ls
cd P03
ls
orign_2
./orign_2
./orign_2 128 10000000
./orign_2 128 100000000
./orign_2 128 1000000000
finger
finger pollinate@
xeyes
Last login: Wed Feb 19 23:56:35 on ttys000
;MMMMMMMMMMMMMMMMMMMMMMMM:       Resolution: 1512x982
:MMMMMMMMMMMMMMMMMMMMMMMM:       DE: Aqua
.MMMMMMMMMMMMMMMMMMMMMMMMX.      WM: Quartz Compositor
sahn@dgx.sdsu.edu's password:
Welcome to NVIDIA DGX Server Version 5.2.0 (GNU/Linux 5.4.0-205-generic x86_64)

  System information as of Fri 21 Feb 2025 10:13:42 AM PST

  System load:                 0.86
  Usage of /:                  76.5% of 1.72TB
  Memory usage:                2%
  Swap usage:                  0%
  Processes:                   2651
  Users logged in:             4
  IPv4 address for docker0:    172.17.0.1
  IPv4 address for enp225s0f0: 130.191.49.91
  IPv6 address for enp225s0f0: 2607:f380:a67:e1:63f:72ff:fece:6faa
  IPv4 address for enp226s0:   192.168.1.91
  IPv4 address for ibp12s0:    10.1.0.91
  IPv4 address for ibp18s0:    10.1.1.91

The system has 0 critical alerts and 1 warnings. Use 'sudo nvsm show alerts' for more details.


********** ********** ********** ********** ********** ********** **********

Please invoke long-running jobs through the SLURM scheduler using the sbatch
command. Slurm Workload Manager Tutorials are available via URL
https://slurm.schedmd.com/tutorials.html

********** ********** ********** ********** ********** ********** **********

Last login: Wed Feb 19 20:48:30 2025 from 10.130.210.184
sahn@dgx:~$ w
 10:13:46 up 17 days, 40 min, 12 users,  load average: 1.03, 0.92, 0.83
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
pathey   pts/0    10.130.87.103    09:57   42.00s  0.09s  0.00s nano P04_Serial
holland  pts/1    tmux(2364571).%1 08Feb25 22:19m  0.46s  0.46s -bash
njohnson pts/2    10.130.188.8     10:09    3:32   0.07s  0.07s -bash
pathey   pts/5    10.130.87.103    10:02   10:42   0.08s  0.00s nano P03_Parall
paolini  pts/6    97.130.194.74    05:11    5:01m  0.03s  0.03s -bash
paolini  pts/7    97.130.194.74    05:11    5:01m  0.03s  0.03s -bash
pathey   pts/8    10.130.87.103    10:04   50.00s  0.11s  0.11s -bash
holland  pts/12   tmux(2364571).%0 05Feb25  4days  0.20s  0.20s -bash
sahn     pts/17   10.130.210.184   10:13    1.00s  0.10s  0.03s w
paolini  pts/14   tmux(798764).%0  Wed15   16:45m  0.14s  5.90s tmux
paolini  pts/15   10.130.93.221    10:09   25.00s  0.09s  0.09s -bash
holland  pts/33   tmux(2364571).%2 09Feb25  4days  0.14s  0.14s -bash
sahn@dgx:~$ module load nvhpc/22.11
sahn@dgx:~$ cuda-memcheck
/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.0/bin/cuda-memcheck: Nothing to check
Usage: cuda-memcheck [options] [your-program] [your-program-options]
Options:
 --binary-patching <yes|no>  [Default : yes]
                       Control the binary patching of the device code. This is enabled by default.
                       Disabling this option will result in a loss of precision for error reporting.
 --check-api-memory-access <yes|no> [Default : yes]
                       Check cudaMemcpy/cudaMemset for accesses to device memory
 --check-deprecated-instr <yes|no>  [Default : no]
                       Check for usage of deprecated instructions.
                       If deprecated instruction usage is found, an error will be reported.
                       Which instructions are checked might depend on the selected tool.
                       This is disabled by default.
 --check-device-heap <yes|no>  [Default : yes]
                       Check allocations on the device heap. This is enabled by default.
 --demangle <full|simple|no>  [Default : full]
                       Demangle function names
                       full   : Show full name and prototype
                       simple : Show only device kernel name
                       no     : Show mangled names
 --destroy-on-device-error <context|kernel>   [Default : context]
                       Behavior of cuda-memcheck on a precise device error.
                       NOTE: Imprecise errors  will always destroy the context.
                       context : CUDA Context is terminated with an error.
                       kernel  : Kernel is terminated. Subsequent kernel launches are still allowed.
 --error-exitcode <number> [Default : 0]
                       When this is set, memcheck will return the given exitcode when any errors are detected
 --filter key1=val1,key2=val2,...
                       The filter option can be used to control the kernels that will be checked by the tool
                       Multiple filter options can be defined. Each option is additive, so kernels matching
                       any specified filter will be checked
                       Filters are specified as key value pairs, with each pair separated by a ','
                       Keys have both a long form, and a shorter form for convenience
                       Valid values for keys are:
                           kernel_name, kne      : The value is the full demangled name of the kernel
                           kernel_substring, kns : The value is a substring present in the demangled name of the kernel
                       NOTE: The name and substring keys cannot be simultaneously specified
 --flush-to-disk <yes|no>   [Default : no]
                       Flush errors to disk. This can be enabled to ensure all errors are flushed down
 --force-blocking-launches <yes|no>   [Default : no]
                       Force launches to be blocking.
 -h | --help           Show this message.
 --help-debug          Show information about debug only flags
 --language <c|fortran> [Default : c]
                       This option can be used to enable language specific behavior. When set to fortan, the thread and block indices
                       of messages printed by cuda-memcheck will start with 1-based offset to match Fortran semantics.
 --log-file <string>   File where cuda-memcheck will write all of its text output. If not specified, memcheck output is written to stdout.
                       The sequence %p in the string name will be replaced by the pid of the cuda-memcheck application.
                       The sequence %q{FOO} will be replaced by the value of the environment variable FOO. If the environment variable
                       is not defined, it will be replaced by an empty string.
                       The sequence %% is replaced with a literal % in the file name.
                       Any other character following % will cause the entire string to be ignored.
                       If the file cannot be written to for any reason including an invalid path, insufficient permissions or disk being full
                       the output will go to stdout
 --leak-check <full|no> [Default : no]
                       Print leak information for CUDA allocations.
                       NOTE: Program must end with cudaDeviceReset() for this to work.
 --prefix <string>     Changes the prefix string displayed by cuda-memcheck.
 --print-level <info|warn|error|fatal> [Default : warn]
                       Set the minimum level of errors to print
 --print-limit <number> [Default is : 10000]
                       When this is set, memcheck will stop printing errors after reaching the given number
                       of errors. Use 0 for unlimited printing.
 --read <file>         Reads error records from a given file.
 --racecheck-report <all|hazard|analysis>  [Default : analysis]
                       The reporting mode that applies to racecheck.
                       all      : Report all hazards and race analysis reports.
                       hazard   : Report only hazards.
                       analysis : Report only race analysis results.
 --report-api-errors <all|explicit|no> [Default : explicit]
                       Print errors if any API call fails
                       all      : Report all CUDA API errors, including those APIs invoked implicitly
                       explicit : Report errors in explicit CUDA API calls only
                       no       : Disable reporting of CUDA API errors
 --save <file>         Saves the error record to file.
                       The sequence %p in the string name will be replaced by the pid of the cuda-memcheck application.
                       The sequence %q{FOO} will be replaced by the value of the environment variable FOO. If the environment variable
                       is not defined, it will be replaced by an empty string.
                       The sequence %% is replaced with a literal % in the file name.
                       Any other character following % will cause an error.
 --show-backtrace <yes|host|device|no> [Default : yes]
                       Display a backtrace on error.
                       no     : No backtrace shown
                       host   : Only host backtrace shown
                       device : Only device backtrace shown for precise errors
                       yes    : Host and device backtraces shown
                       See the manual for more information
 --tool <memcheck|racecheck|synccheck|initcheck>  [Default : memcheck]
                       Set the tool to use.
                       memcheck    : Memory access checking
                       racecheck   : Shared memory hazard checking
                       Note : This disables memcheck, so make sure the app is error free.
                       synccheck   : Synchronization checking
                       initcheck   : Global memory initialization checking
 --track-unused-memory <yes|no> [Default : no]
                       Check for unused memory allocations. This requires initcheck tool.
 -V | --version        Print the version of cuda-memcheck.

Please see the cuda-memcheck manual for more information.

sahn@dgx:~$ cuda-memcheck
.bash_history  .bashrc        .config/       .profile       workspace/
.bash_logout   .cache/        .nv/           .viminfo
sahn@dgx:~$ cuda-memcheck
.bash_history  .bashrc        .config/       .profile       workspace/
.bash_logout   .cache/        .nv/           .viminfo
sahn@dgx:~$ cuda-memcheck  lspci | grep -i nvidia
07:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
0f:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
47:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
4e:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
87:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
90:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
b7:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
bd:00.0 3D controller: NVIDIA Corporation Device 20b0 (rev a1)
c4:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)
c5:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)
c6:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)
c7:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)
c8:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)
c9:00.0 Bridge: NVIDIA Corporation Device 1af1 (rev a1)
sahn@dgx:~$ uname -m && cat /etc/*release
x86_64
DGX_NAME="DGX Server"
DGX_PRETTY_NAME="NVIDIA DGX Server"
DGX_SWBUILD_DATE="2021-08-18-20-11-12"
DGX_SWBUILD_VERSION="5.1.0"
DGX_COMMIT_ID="6933cd7"
DGX_PLATFORM="DGX Server for DGX A100"
DGX_SERIAL_NUMBER="1573020000041"

DGX_OTA_VERSION="5.2.0"
DGX_OTA_DATE="Thu 24 Mar 2022 01:43:22 PM PDT"
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=20.04
DISTRIB_CODENAME=focal
DISTRIB_DESCRIPTION="Ubuntu 20.04.4 LTS"
NAME="Ubuntu"
VERSION="20.04.4 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 20.04.4 LTS"
VERSION_ID="20.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=focal
UBUNTU_CODENAME=focal
sahn@dgx:~$ uname -r
5.4.0-205-generic
sahn@dgx:~$ ls
workspace
sahn@dgx:~$ pwd
/mnt/beegfs/dgx/sahn
sahn@dgx:~$ cd workspace/
sahn@dgx:~/workspace$ pwd
/mnt/beegfs/dgx/sahn/workspace
sahn@dgx:~/workspace$ ls
250124  250127  P01  P02  P03
sahn@dgx:~/workspace$ cd P03
sahn@dgx:~/workspace/P03$ ls
imp_0  imp_0.c  orign_0  orign_0.c  orign_1  orign_1.c  orign_2  orign_2.c
sahn@dgx:~/workspace/P03$ orign_2
orign_2: command not found
sahn@dgx:~/workspace/P03$ ./orign_2
Usage: ./orign_2 <num_threads> <num_intervals>
sahn@dgx:~/workspace/P03$ ./orign_2 128 10000000
Requesting 128 threads
Using 10000000 intervals
Approximated integral result: 2.056167583560283
Exact integral solution: 2.056167583560283
Numerical error: 0.000000000000000e+00
Execution time: 0.073278 seconds
sahn@dgx:~/workspace/P03$ ./orign_2 128 100000000
Requesting 128 threads
Using 100000000 intervals
Approximated integral result: 2.056167583560277
Exact integral solution: 2.056167583560283
Numerical error: 5.773159728050814e-15
Execution time: 0.126754 seconds
sahn@dgx:~/workspace/P03$ ./orign_2 128 1000000000
Requesting 128 threads
Using 1000000000 intervals
Approximated integral result: 2.056167583560271
Exact integral solution: 2.056167583560283
Numerical error: 1.154631945610163e-14
Execution time: 0.903710 seconds
sahn@dgx:~/workspace/P03$ finger
Login     Name                   Tty      Idle  Login Time   Office     Office Phone
adutta    Arkajit Dutta          pts/24      2  Feb 21 10:43 (10.130.194.32)
asanchez  Anabel Sanchez         pts/25     15  Feb 21 10:28 (10.130.173.147)
ascott    Anthonie Scott         pts/20     12  Feb 21 10:35 (10.130.187.178)
atamim    Abdul Karim Tamim      pts/18     10  Feb 21 10:14 (10.130.140.186)
fgonzalez Fausto Saavedra Gonza  pts/21      3  Feb 21 10:31 (10.130.164.206)
holland   Gregory Holland        pts/1   22:54  Feb  8 11:32 (tmux(2364571).%1)
holland   Gregory Holland        pts/12     4d  Feb  5 13:00 (tmux(2364571).%0)
holland   Gregory Holland        pts/33     4d  Feb  9 20:22 (tmux(2364571).%2)
jrathi    Jiya Santosh Rathi     pts/22     18  Feb 21 10:28 (10.130.207.55)
njohnson  Nathan Johnson         pts/2       7  Feb 21 10:27 (10.130.188.8)
paolini   Christopher Paolini    pts/6    5:36  Feb 21 05:11 (97.130.194.74)
paolini   Christopher Paolini    pts/7    5:36  Feb 21 05:11 (97.130.194.74)
paolini   Christopher Paolini    pts/14  17:20  Feb 19 15:14 (tmux(798764).%0)
paolini   Christopher Paolini    pts/15     21  Feb 21 10:09 (10.130.93.221)
paolini   Christopher Paolini    pts/26         Feb 21 10:29 (10.130.93.221)
pathey    Peter Athey            pts/0      19  Feb 21 09:57 (10.130.87.103)
pathey    Peter Athey            pts/5      45  Feb 21 10:02 (10.130.87.103)
pathey    Peter Athey            pts/8      35  Feb 21 10:04 (10.130.87.103)
ppagaria  Parshav Pagaria        pts/23     16  Feb 21 10:28 (10.130.120.187)
sahn      Sukyum Ahn             pts/17         Feb 21 10:13 (10.130.210.184)
ls
cd workspace/
ls
mkdir P04
cd P0
cd P04
ls
vim 0_serial.c
vim 2_gpu.c
vim 3_simd.c
vim 1_parallel.c
ls
nvc -mp=gpu -gpu=cc80 -o 2_gpu 2_gpu.c 
which nvc
sudo apt update
apt install nvidia-hpc-sdk
echo $PATH | grep nvidia
which nvcc
nvcc --version
cd /opt/nvidia/hpc_sdk/Linux_x86_64
ls
cd 22.11/
ls
cd .
cd /
ls
cd ~
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/lib:$LD_LIBRARY_PATH
cd workspace/P04/
ls
nvcc --version
which nvc
nvc -mp=gpu -gpu=cc80 -o 2_gpu 2_gpu.c 
ls
2to3-2.7 
./2_gpu 
./2_gpu 100 100
gcc -o 0_serial 0_serial.c -lm -fopenmp
./0_serial 
./0_serial 1 1
./0_serial 10 10
./0_serial 100 100
./0_serial 100 1000
./0_serial 1000 1000
./0_serial 100 100
./0_serial 10 10
./0_serial 200 1000 >> a.txt
cat a.txt 
./0_serial 50 1000 >> a.txt
cat a.txt 
vim a.txt 
./0_serial 10 1000 >> a.txt
cat a.txt 
T10 = [...
1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 ; ...
1.000000 0.500000 0.303616 0.214466 0.171259 0.153082 0.153082 0.171259 0.214466 0.303616 0.500000 1.000000 ; ...
1.000000 0.696383 0.500000 0.382989 0.317488 0.287987 0.287987 0.317488 0.382989 0.500000 0.696383 1.000000 ; ...
1.000000 0.785534 0.617011 0.500000 0.427719 0.393390 0.393390 0.427719 0.500000 0.617011 0.785534 1.000000 ; ...
./0_serial 10 1000 >> a.txt
cat a.txt 
ls
cd workspace/P04
ls
vim 0_serial.c
gcc -o 0_serial -lm -fopenmp 0_serial.c 
./0_serial 10 10
./0_serial 1000 10
./0_serial 1000 100
./0_serial 1000 1000
./0_serial 1000 10000
./0_serial 1000 100000 >> output.txt
export OMP_NUM_THREADS=2
./1_parallel 1000 1000000000 >> 1_09_2.txt
export OMP_NUM_THREADS=2
./1_parallel 1000 1000000000 >> 1_09_2_1.txt
cd workspace/P04
ls
htop
ls
vim 1_parallel.c 
nvc -mp=gpu -gpu=cc80 -o 1_parallel 1_parallel.c 
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/lib:$LD_LIBRARY_PATH
nvc -mp=gpu -gpu=cc80 -o 1_parallel 1_parallel.c 
vim 1_parallel.c
ls
vim 0_serial.c
gcc -o 0_serial 0_serial.c -lm -fopenmp
l
ls -l
nvc -mp=gpu -gpu=cc80 -o 3_simd 3_simd.c 
export OMP_NUM_THREADS=128
ls
./3_simd 1000 10000000
./3_simd 1000 1000000
./3_simd 1000 1000
./3_simd 1000 10000
export OMP_NUM_THREADS=512
./3_simd 1000 10000
htop
nvc -mp=gpu -gpu=cc80 -o 4_extra 4_extra.c
nvc -mp=gpu -gpu=cc80 -o 5_gpu 5_gpu.c
mv 4_extra.c 6_simd.c
nvc -mp=gpu -gpu=cc80 -o 6_simd 6_simd.c 
htop
./6_simd 1000 10000
vim test.sh
chmod +x test.sh 
./test.sh 
ls
cd workspace/P04
ls
export OMP_NUM_THREADS=128
./2_gpu 100 100000
./2_gpu 10 100000
./2_gpu 1000 100
./1_parallel 1000 100000
./1_parallel 100 100
./1_parallel 10 1000000
vim 4_extra.c
nvc -mp=gpu -gpu=cc80 -o 4_extra 4_extra.c 
./4_extra 10 1000000
htop
vim 5_gpu.c
nvc -mp=gpu -gpu=cc80 -o 5_gpu 5_gpu.c 
ls
rm -rf 4_extra 
ls
vim 4_parallel.c
gcc -o 4_parallel 4_parallel.c -lm -fopenmp
./6_simd 1000 1000000
./0_serial 1000 100000 >> output.txt
./0_serial 1000 10000000 >> 0_07.txt
htop
ls
cd workspace/
ls
cd P04
ls
cat 1_09_2.txt 
cat 1_09_2_1.txt 
ls
htop
cd workspace/P04/
pwd 
export OMP_NUM_THREADS=2
./1_parallel 100 1000000000 
./1_parallel 100 1000
./1_parallel 100 10000
export OMP_NUM_THREADS=128
./1_parallel 100 10000
cat 1_parallel.c
cat 4_parallel.c
cat 1_parallel.c
export OMP_NUM_THREADS=128
./4_parallel 200 100000
vim 7_parallel.c
gcc -o 7_parallel -lm -fopenmp 7_parallel.c 
./7_parallel 200 100000
export OMP_NUM_THREADS=2
./4_parallel 200 100000
export OMP_NUM_THREADS=4
./4_parallel 200 100000
export OMP_NUM_THREADS=8
./4_parallel 200 100000
export OMP_NUM_THREADS=16
./4_parallel 200 100000
export OMP_NUM_THREADS=32
./4_parallel 200 100000
ps
ps -l
cd workspace/P04
ps -l
ls
tasklist
ls
htop
cd workspace/P04
ls
ls -l
./0_serial 100 1000000 >> 0_06.txt
./0_serial 100 10000000 >> 0_07.txt
./0_serial 100 100000000 >> 0_08.txt
./0_serial 100 1000000000 >> 0_09.txt
./0_serial 100 10000000000 >> 0_10.txt
ls
ls -l
export OMP_NUM_THREADS=128
./1_parallel 100 1000000000 >> 1_09_128_1.txt
export OMP_NUM_THREADS=128
./1_parallel 100 1000000000 
./6_simd 1000 1000000000
./6_simd 100 1000000000
./1_parallel 100 1000000000 
./4_parallel 100 1000000000 
./0_serial 100 10000000000
vim 0_serial.c
gcc -o 0_serial -lm -fopen 0_serial.c
gcc -o 0_serial -lm -fopenmp 0_serial.c
./0_serial 100 10
./0_serial 100 100 
./0_serial 100 1000 
./0_serial 100 10000 
./0_serial 100 100000 
./0_serial 100 1000000
./0_serial 100 10000000 
./0_serial 100 100000000 
./0_serial 100 1000000000 
./0_serial 200 10
./0_serial 200 100 
./0_serial 200 1000 
./0_serial 200 10000 
./0_serial 200 10
./0_serial 200 100 
./0_serial 200 1000 
./0_serial 200 10000 
./0_serial 200 100000 
./0_serial 200 1000000
./0_serial 500 1000000
./0_serial 500 10
./0_serial 500 100 
./0_serial 500 1000 
./0_serial 500 10000 
./0_serial 500 100000 
./0_serial 200 100000 
vim 0_serial.c
gcc -o 0_serial -lm -fopenmp 0_serial.c
./0_serial 200 100000 >> plot.txt
export OMP_NUM_THREADS=64
./2_gpu 200 100000
./3_simd 200 100000
./5_gpu 200 100000
./6_simd 200 100000
export OMP_NUM_THREADS=128
./2_gpu 200 100000
./3_simd 200 100000
./5_gpu 200 100000
./6_simd 200 100000
export OMP_NUM_THREADS=16
./2_gpu 200 100000
./3_simd 200 100000
./5_gpu 200 100000
./6_simd 200 100000
export OMP_NUM_THREADS=32
./2_gpu 200 100000
./3_simd 200 100000
./5_gpu 200 100000
./6_simd 200 100000
cd workspace/P04
ls
export OMP_NUM_THREADS=32
./4_parallel 200 100000
ls
cd workspace/P04/
ls
cat 4_parallel.c
nvltop
nvtop
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/lib:$LD_LIBRARY_PATH
nvltop
sudo apt install nvtop
htop
