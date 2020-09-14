# Intel-lattice
** We plan to rename this repository to Intel-lattice. **

This repository provides an efficient implementation of integer arithmetic on Galois fields. We only support 64-bit unsigned integers.

We currently provide three primary functions:

-  The negacyclic number-theoretic transform (NTT), with the following implementations:
   - 1) A default implementation in native C++
   - 2) An AVX512-DQ-accelerated implementation
   - 3) An AVX512-IFMA-accelerated implementation for prime moduli < 50 bits

  The library will automatically choose the best implementation for the given hardware. Implementation 3) is most preferred, followed by implementation 2), followed by implementation 1).

- Polynomial-polynomial modular multiplication, with the following implementations;
  - 1) A default implementation in native C++
  - 2) An AVX512-DQ accelerated implementation
  - 3) An AVX512-IFMA-accelerated implementation for prime moduli < 50 bits

  The library will automatically choose the best implementation for the given hardware. Implementation 3) is most preferred, followed by implementation 2), followed by implementation 1).

-  The inverse negacyclic number-theoretic transform (NTT), with the following implementations:
   - 1) A default implementation in native C++
   - 2) An AVX512-DQ-accelerated implementation
   - 3) An AVX512-IFMA-accelerated implementation for prime moduli < 50 bits

  The library will automatically choose the best implementation for the given hardware. Implementation 3) is most preferred, followed by implementation 2), followed by implementation 1).


# Build
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
