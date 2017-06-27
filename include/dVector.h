#pragma once

#include "Vector.h"
#include <cuda_runtime.h>

template<typename T>
struct dDenseVector
{
	size_t size;
	T* data;

	dDenseVector() : size(0), data(nullptr) { }
	void alloc(size_t s)
	{
		if (data != nullptr)
			cudaFree(data);
		cudaMalloc(&data, sizeof(T)*s);
		size = s;
	}
	~dDenseVector()
	{
		if (data != nullptr)
			cudaFree(data);
	}
};

template<typename T>
void convert(dDenseVector<T> & dvec, const DenseVector<T>& vec, unsigned int padding = 0)
{
	dvec.alloc(vec.size+padding);
	dvec.size = vec.size;

	cudaMemcpy(dvec.data, &vec.data[0], dvec.size * sizeof(T), cudaMemcpyHostToDevice);
	if (padding)
		cudaMemset(dvec.data + dvec.size, 0, padding * sizeof(T));
}

template<typename T>
void convert(DenseVector<T> & vec, const dDenseVector<T>& dvec)
{
	vec.alloc(dvec.size);
	cudaMemcpy(&vec.data[0], dvec.data, dvec.size * sizeof(T), cudaMemcpyDeviceToHost);
}