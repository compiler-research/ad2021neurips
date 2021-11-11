
  - Setup llvm and clad

  - Setup Tapenade
    - Make the ad function for our gaussian function (details to be added)
    - A few helper functions are needed

```
cd tap/tapenade_3.16/ADFirstAidKit
clang -x c++ -std=c++11 -O2 -fpic -c adStack.c adBuffer.c
cd ../../../
ar -rv libadfirst.a tap/tapenade_3.16/ADFirstAidKit/adBuffer.o tap/tapenade_3.16/ADFirstAidKit/adStack.o
```

  - Compile the binary

```
clang -DCLAD_NO_NUM_DIFF -x c++ -std=c++11 -O2 -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn -Xclang -load -Xclang $PWD/t2/inst/lib/clad.so ngaus.cc -I $PWD/miniconda3/envs/clad_test/include -lstdc++ -lm -ladfirst -g -I $PWD/tap/tapenade_3.16/ADFirstAidKit -L.
```

  - Run it
```
./a.out 1000 5120

(1000 trials with 5120 dimensions of the normal distribution)