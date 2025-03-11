export OMP_NUM_THREADS=2
./1_parallel 1000 1000000000 >> 1_09_2.txt
./4_parallel 1000 1000000000 >> 4_09_2.txt
export OMP_NUM_THREADS=4
./1_parallel 1000 1000000000 >> 1_09_4.txt
./4_parallel 1000 1000000000 >> 4_09_4.txt
export OMP_NUM_THREADS=8
./1_parallel 1000 1000000000 >> 1_09_8.txt
./4_parallel 1000 1000000000 >> 4_09_8.txt
export OMP_NUM_THREADS=16
./1_parallel 1000 1000000000 >> 1_09_16.txt
./4_parallel 1000 1000000000 >> 4_09_16.txt
export OMP_NUM_THREADS=32
./1_parallel 1000 1000000000 >> 1_09_32.txt
./4_parallel 1000 1000000000 >> 4_09_32.txt
export OMP_NUM_THREADS=64
./1_parallel 1000 1000000000 >> 1_09_64.txt
./4_parallel 1000 1000000000 >> 4_09_64.txt
export OMP_NUM_THREADS=128
./1_parallel 1000 1000000000 >> 1_09_128.txt
./4_parallel 1000 1000000000 >> 4_09_128.txt
export OMP_NUM_THREADS=256
./1_parallel 1000 1000000000 >> 1_09_256.txt
./4_parallel 1000 1000000000 >> 4_09_256.txt


export OMP_NUM_THREADS=64
./2_gpu 1000 1000000000 >> 2_09_64.txt
./3_simd 1000 1000000000 >> 3_09_64.txt
./5_gpu 1000 1000000000 >> 5_09_64.txt
./6_simd 1000 1000000000 >> 6_09_64.txt
export OMP_NUM_THREADS=128
./2_gpu 1000 1000000000 >> 2_09_128.txt
./3_simd 1000 1000000000 >> 3_09_128.txt
./5_gpu 1000 1000000000 >> 5_09_128.txt
./6_simd 1000 1000000000 >> 6_09_128.txt

