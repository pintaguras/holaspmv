Hola SpMV: Globally Homogeneous, Locally Adaptive Sparse Matrix-Vector Multiplication on the GPU
========


---
### Introduction

Hola SpMV provides an efficient sparse matrix vector multiplication for NVIDIA GPUs.
It takes a CSR matrix (M) and a vector (a) as input and computes b=M*a.
Hola SpMV performs load balancing between thread blocks. For this load balancing it requires an additional GPU buffer.
This buffer must be allocated before running SpMV and will be filled as a first step during SpMV.
The code is currently under development and will be updated in terms of useability, readability and performance.

---
### Recent updates

 * 3.11.2018 CMake support
 * 3.11.2018 fix for loading symmetric and hermitian matrices
 * 27.6.2017 Initial upload of naive SpMV, naive SpMVT and Hola SpMV
 
### Expected updates
 * CMake build files for Linux
 * Transpose SpMV with Hola
 * Tuned parameters for different GPU generations
 * Performance optimizations for small matrices

---
### Build and usage

The code is split into source code and header files. The important code is found in `source/holaspmv.cu`.
The other classes and headers are mostly helpers to load matrices, convert data, compute ground thruth results etc.

Hola SpMV is built on CUDA. You need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and have an NVIDIA GPU with compute capability 3.0 or higher in your system.
It has been developed with CUDA Toolkit 8.0 and optimized for Maxwell and Pascal GPUs.

Hola SpMV uses the reduction found in [cub](https://nvlabs.github.io/cub/). Before building, make sure to clone cub in the `deps/`.
Under Windows you can use the provided `pull.bat`.

Currently, build files are included for Visual Studio 2015 under `build/vs2015`. In the near future we will include CMake files for building and running under Linux.

The executeable loads an mtx file (supplied as the first command argument), converts it to CSR, stores the converted matrix as a binary file for reuse.
It then computes the ground truth SpMV on the CPU before running `naiveSpMV` and `holaSpMV`. 
Hola SpMV requires an additional buffer for load balancing, which is to be executed on the GPU before launching SpMV.
To query the size of the buffer, call the HolaSpMV function with a nullptr. 

mtx matrices can be downloaded from [SuiteSparse](https://www.cise.ufl.edu/research/sparse/matrices/).
Hola has been evaluated on all reasonably sized SuiteSparse matrices.

---
### Resources


##### [Globally Homogeneous, Locally Adaptive Sparse Matrix-Vector Multiplication on the GPU](http://dl.acm.org/citation.cfm?id=3079086)
When you use the code in a scientific work, please cite our paper

Globally Homogeneous, Locally Adaptive Sparse Matrix-Vector Multiplication on the GPU  
Markus Steinberger,  Rhaleb Zayer and Hans-Peter Seidel  
Proceedings of the International Conference on Supercomputing, 2017

	:::
		@inproceedings{Steinberger:2017:GHL:3079079.3079086,
		 author = {Steinberger, Markus and Zayer, Rhaleb and Seidel, Hans-Peter},
		 title = {Globally Homogeneous, Locally Adaptive Sparse Matrix-vector Multiplication on the GPU},
		 booktitle = {Proceedings of the International Conference on Supercomputing},
		 series = {ICS '17},
		 year = {2017},
		 isbn = {978-1-4503-5020-4},
		 location = {Chicago, Illinois},
		 pages = {13:1--13:11},
		 articleno = {13},
		 numpages = {11},
		 url = {http://doi.acm.org/10.1145/3079079.3079086},
		 doi = {10.1145/3079079.3079086},
		 acmid = {3079086},
		 publisher = {ACM},
		 address = {New York, NY, USA},
		 keywords = {GPU, SpMV, linear algebra, sparse matrix},
		}



##### [How naive is naive SpMV on the GPU?](http://ieeexplore.ieee.org/document/7761634/)
The source code includes our naive SpMV and transpose SpMV implementation. If you use this code, please cite:


How naive is naive SpMV on the GPU?  
Markus Steinberger, Andreas Derler, Rhaleb Zayer and Hans-Peter Seidel  
IEEE High Performance Extreme Computing Conference, 2016

	:::
		@INPROCEEDINGS{7761634,
		author={Markus Steinberger and Andreas Derler and Rhaleb Zayer and Hans-Peter Seidel},
		booktitle={2016 IEEE High Performance Extreme Computing Conference (HPEC)},
		title={How naive is naive SpMV on the GPU?},
		year={2016},
		pages={1-8},
		keywords={cache storage;data handling;graphics processing units;matrix multiplication;parallel processing;sparse matrices;GPU hardware;cache performance;complex data format;data conversion;direct multiplication;fast hardware supported atomic operation;format conversion;graphics hardware;highly tuned parallel implementation;linear algebra computation;multiplication transposition;naive SpMV;sparse matrix vector multiplication;transpose operation;Bandwidth;Graphics processing units;Hardware;Instruction sets;Load management;Memory management;Sparse matrices},
		doi={10.1109/HPEC.2016.7761634},
		month={Sept},}


---
### Contact

[Markus Steinberger](http://www.markussteinberger.net) and [Rhaleb Zayer](http://people.mpi-inf.mpg.de/~rzayer/).

---
### Paper Graphs

The supplemental performance plots can be found here:

| Operation  | Size   | Format  | Link  |
| ---------- | ------ | ------- | ----- |
| SpMV       | small  | float   | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_float_small.pdf) |
| SpMV       | large  | float   | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_float_large.pdf) |
| SpMV       | small  | double  | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_double_small.pdf) |
| SpMV       | large  | double  | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_double_large.pdf) |
| SpMVT      | small  | float   | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_t_float_small.pdf) |
| SpMVT      | large  | float   | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_t_float_large.pdf) |
| SpMVT      | small  | double  | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_t_double_small.pdf) |
| SpMVT      | large  | double  | [pdf](https://bitbucket.org/gpusmack/holaspmv/raw/b5613f90aea7fc524498e1f82b0ca53bc580e589/graphs/spmv_comp_marker_t_double_large.pdf) |
