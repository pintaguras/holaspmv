CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(holaSpMV LANGUAGES CXX CUDA)



option(CUDA_BUILD_CC30 "Build with compute capability 3.0 support" FALSE)
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" TRUE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" TRUE)
option(CUDA_BUILD_CC70 "Build with compute capability 7.0 support" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)


if (CUDA_BUILD_CC30)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=sm_30 ")
endif ()
if (CUDA_BUILD_CC35)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute__35,code=sm_35 ")
endif ()
if (CUDA_BUILD_CC50)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50 ")
endif ()
if (CUDA_BUILD_CC52)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52 ")
endif ()
if (CUDA_BUILD_CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61 ")
endif ()
if (CUDA_BUILD_CC70)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70 ")
endif ()

if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo ")
endif ()



SET(holaDir ${PROJECT_SOURCE_DIR}/../../)

FILE(GLOB headers ${holaDir}/include/*.h)
FILE(GLOB cudaheaders ${holaDir}/include/*.cuh)
FILE(GLOB sources ${holaDir}/source/*.cpp)
FILE(GLOB cudasources ${holaDir}/source/*.cu)


INCLUDE_DIRECTORIES(${holaDir}/include/)


ADD_EXECUTABLE(holaSPMV ${headers} ${cudaheaders} ${sources} ${cudasources})
target_link_libraries(holaSPMV ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
