Benchmark examples for AD on normal distributions and its scaling with the number of times the derivative is evaluated.

## To compile and run:

### CPU Benchmark: 

```
bin/clang-9  -Xclang -load -Xclang ./etc/cling/plugins/lib/clad.so  -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn  -I./etc/cling/plugins/include/ -lstdc++ benchmarking_gauss_exec.cpp  -o benchmarking_gauss_exec  -ldl -lrt -x c++ -std=c++11 -O0 -lm -ldl
```

### GPU Benchmark:

```
bin/clang-9  -Xclang -load -Xclang ./etc/cling/plugins/lib/clad.so  -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn   -I./etc/cling/plugins/include/ -lstdc++ benchmarking_gauss_clad.cu -o benchmarking_gauss_clad  -L/usr/local/cuda-10.0/lib64 -lcudart_static -ldl -lrt -pthread --cuda-path=/usr/local/cuda-10.0 -x c++ -std=c++11 -lcuda -lm
```

## To profile GPU Benchmark:

```
nvprof --unified-memory-profiling off ./benchmarking_gauss_clad
```