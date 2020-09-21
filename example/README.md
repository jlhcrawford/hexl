# Example in using Intel-lattice in external library

```bash
mkdir build
cd build
export INTEL_LATTICE=/path/to/intel-lattice
cmake .. -DINTEL_LATTICE_LIBRARY=$INTEL_LATTICE/build/intel-lattice/libintel_lattice.a -DINTEL_LATTICE_HEADER=$INTEL_LATTICE/intel-lattice/include

make -j

```
