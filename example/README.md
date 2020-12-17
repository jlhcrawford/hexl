# Example using Intel-lattice in an external application

To use IntelLattice in an external application, first build Intel-lattice with `LATTICE_EXPORT=ON`. Then, run `make install`.

Next, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(IntelLattice
    HINTS ${INTEL_LATTICE_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> intel_lattice)
```

If Intel-lattice is installed globally, `INTEL_LATTICE_HINT_DIR` is not needed. Otherwise, `INTEL_LATTICE_HINT_DIR` should be the directory containing  `IntelLatticeConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/`
