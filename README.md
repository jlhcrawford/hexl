# NTT-avx

This repository provides an efficient implementation of the negacyclic number-theoretic transform (NTT). Currently, we only support 64-bit unsigned integers. We provide three implementations:

1) A default implementation
2) An AVX-512-accelerated implementation
3) An AVX512-IFMA-accelearated implementation for prime moduli < 52 bits

The library will automatically choose the best implementation for the given hardware. Implementation 3) is most preferred, followed by implementation 2), followed by implementation 1).

We may end up merging this into intel-palisade-development, or keep it as a separate repo.

# Build
```bash

mkdir build
cd build
# Other compilers may work; we find best performance with clang-10
# For debugging with heavy performance hit, include -DNTT_DEBUG=ON
cmake .. -DCMAKE_CXX_COMPILER=clang++-10 -DCMAKE_C_COMPILER=clang-10

make -j
```

# Test
```bash
make test
```

# Benchmarking
```bash
make bench
```

# Contributing
Before contributing, please run
```bash
make check
```
and make sure all unit-test pass and the pre-commit checks pass.
