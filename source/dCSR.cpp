#include "../include/dCSR.h"
#include "../include/CSR.h"

#include <cuda_runtime.h>

namespace
{
	template<typename T>
	void dealloc(dCSR<T>& mat)
	{
		if (mat.col_ids != nullptr)
			cudaFree(mat.col_ids);
		if (mat.data != nullptr)
			cudaFree(mat.data);
		if (mat.row_offsets != nullptr)
			cudaFree(mat.row_offsets);
		mat.col_ids = nullptr;
		mat.data = nullptr;
		mat.row_offsets = nullptr;
	}
}

template<typename T>
void dCSR<T>::alloc(size_t r, size_t c, size_t n)
{
	dealloc(*this);
	rows = r;
	cols = c;
	nnz = n;
	cudaMalloc(&data, sizeof(T)*n);
	cudaMalloc(&col_ids, sizeof(unsigned int)*n);
	cudaMalloc(&row_offsets, sizeof(unsigned int)*(r+1));
}
template<typename T>
dCSR<T>::~dCSR()
{
	dealloc(*this);
}


template<typename T>
void convert(dCSR<T>& dcsr, const CSR<T>& csr, unsigned int padding)
{
	dcsr.alloc(csr.rows + padding, csr.cols, csr.nnz + 8*padding);
	dcsr.rows = csr.rows; dcsr.nnz = csr.nnz;
	cudaMemcpy(dcsr.data, &csr.data[0], csr.nnz * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(dcsr.col_ids, &csr.col_ids[0], csr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dcsr.row_offsets, &csr.row_offsets[0], (csr.rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

	if (padding)
	{
		cudaMemset(dcsr.data + csr.nnz, 0, 8 * padding * sizeof(T));
		cudaMemset(dcsr.col_ids + csr.nnz, 0, 8 * padding * sizeof(unsigned int));
		cudaMemset(dcsr.row_offsets + csr.rows + 1, 0, padding * sizeof(unsigned int));
	}
}

template void dCSR<float>::alloc(size_t r, size_t c, size_t n);
template void dCSR<double>::alloc(size_t r, size_t c, size_t n);

template dCSR<float>::~dCSR();
template dCSR<double>::~dCSR();

template void convert(dCSR<float>& dcsr, const CSR<float>& csr, unsigned int);
template void convert(dCSR<double>& dcsr, const CSR<double>& csr, unsigned int);