#include "../include/naivespmv.h"

#include <stdint.h>
#include <stdexcept>

template<typename T>
__device__ T myLoad(const T* d)
{
	return *d;
	//return __ldg(d);
}

template<typename ValueType, typename IndexType, typename OffsetType>
__global__ void spmv(uint32_t num_non_zeroes, uint32_t out_size, uint32_t num_other, 
	const ValueType* __restrict matrix, const IndexType* __restrict inIndex, const OffsetType*__restrict offsets, 
	const ValueType* __restrict inVec, ValueType* __restrict outVec)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= out_size)
		return;

	ValueType sum = 0;
	for (OffsetType j = myLoad(offsets + i); j < myLoad(offsets + i + 1); ++j)
	{
		IndexType ind = myLoad(inIndex + j);
		sum += myLoad(inVec + ind) * myLoad(matrix + j);
	}
	outVec[i] = sum;
}


//double atomic add hack for devices that do not support it in hardware
template<typename T>
__device__ inline T tempAtomicAdd(T* address, T val)
{
	return atomicAdd(address, val);
}
#if __CUDA_ARCH__ < 600
//http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
template<>
__device__ inline double tempAtomicAdd<double>(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old; old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	return __longlong_as_double(old);
}

#endif

template<typename ValueType, typename IndexType, typename OffsetType>
__global__ void spmvt(uint32_t num_non_zeroes, uint32_t out_size, uint32_t num_other, 
	const ValueType* __restrict matrix, const IndexType* __restrict inIndex, const OffsetType*__restrict offsets, 
	const ValueType* __restrict inVec, ValueType* __restrict outVec)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_other)
		return;

	ValueType inV = myLoad(inVec + i);
	for (OffsetType j = myLoad(offsets + i); j < myLoad(offsets + i + 1); ++j)
	{
		IndexType ind = myLoad(inIndex + j);
		ValueType res = inV * myLoad(matrix + j);
		tempAtomicAdd(outVec + ind, res);
	}
}



template<typename T>
void naive_spmv(dDenseVector<T>& res, const dCSR<T>& m, const dDenseVector<T>& v, bool transpose)
{
	if (transpose && v.size != m.rows)
		throw std::runtime_error("SPMV dimensions mismatch");
	if (!transpose && v.size != m.cols)
		throw std::runtime_error("SPMV dimensions mismatch");

	size_t outsize = transpose ? m.cols : m.rows;
	if (res.size < outsize)
		res.alloc(outsize);
	res.size = outsize;

	uint32_t blockSize = 256;
	if (transpose)
	{
		spmvt<T, unsigned int, unsigned int> <<<(m.cols + blockSize - 1) / blockSize, blockSize >>> (
			m.nnz, m.rows, m.cols,
			m.data, m.col_ids, m.row_offsets,
			v.data, res.data);
	}
	else
	{

		spmv<T, unsigned int, unsigned int><<<(m.rows + blockSize - 1) / blockSize, blockSize >>> (
			m.nnz, m.rows, m.cols,
			m.data, m.col_ids, m.row_offsets,
			v.data, res.data);
	}
}

template void naive_spmv<float>(dDenseVector<float>& res, const dCSR<float>& m, const dDenseVector<float>& v, bool transpose);
template void naive_spmv<double>(dDenseVector<double>& res, const dCSR<double>& m, const dDenseVector<double>& v, bool transpose);