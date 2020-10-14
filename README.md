# Intel-lattice
This repository provides an efficient implementation of integer arithmetic on Galois fields. Such arithmetic is prevalent in cryptography, particularly in homomorphic encryption (HE) schemes based on the ring learning with errors (RLWE) problem.

In such applications, the underlying prime `p` is typically 40-60 bits. Currently, Intel-lattice provides an API for 64-bit unsigned integers.

Intel lattice implements the following functions:
-  The negacyclic number-theoretic transform (NTT), with the following implementations:
   - 1) A default implementation in native C++
   - 2) An AVX512-DQ-accelerated implementation
   - 3) An AVX512-IFMA-accelerated implementation for `p` < 50 bits

- Element-wise vector-vector modular multiplication, with the following implementations;
  - 1) A default implementation in native C++
  - 2) An AVX512-DQ accelerated implementation
  - 3) An AVX512-IFMA-accelerated implementation for `p`  < 50 bits

-  The inverse negacyclic number-theoretic transform (NTT), with the following implementations:
   - 1) A default implementation in native C++
   - 2) An AVX512-IFMA-accelerated implementation for `p` > 50 bits bits
   - 3) An AVX512-DQ-accelerated implementation for `p` < 50 bits

In each case, the library will automatically choose the best implementation for the given CPU AVX512 feature set.

The functions are currently optimized for performance on Intel ICX servers. Performance may suffer on non-ICX servers.

For additional functionality, see the public headers, located in `include/intel-lattice`


# Build

## Dependencies
  - CMake >= 3.13
  - A modern compiler supporting C++17, e.g. clang-10
  - Operating system: Currently, we have tested the code on Ubuntu 18.04 and macOS 10.15

To build intel-lattice, call
```bash
mkdir build
cd build
# Other compilers may work; we find best performance with clang-10
# For debugging with heavy performance hit, include -DLATTICE_DEBUG=ON
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

## Repository layout
Public headers reside in the `intel-lattice/include` folder.
Private headers, e.g. those containing AVX512 code should not be put in this folder.
