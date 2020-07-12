# NTT-avx

This is a separate repo to test implementations of the number-theoretic transform (NTT) using AVX512

We may end up merging this into intel-palisade-development, or keep it as a separate repo.


# Build
```bash

mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-10 -DCMAKE_C_COMPILER=clang-10
make -j
```

# Test
```bash
./test/unit-test
```

# Contributing
Before contributing, please run the pre-commit, i.e.
```bash
pre-commit install
pre-commit run --all-files
```
and make sure all unit-tests pass.
