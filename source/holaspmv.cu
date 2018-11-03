#include "../include/holaspmv.h"
#include "../include/common.cuh"
#include "../deps/cub/cub/cub.cuh"

#include <stdint.h>
#include <stdexcept>
#include <algorithm>

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




template<typename OFFSET_TYPE, uint32_t NNZ_PER_BLOCK, typename VALUE_TYPE3>
__global__ void DetermineBlockStarts(int num_other, const OFFSET_TYPE*__restrict offsets, uint32_t* startingIds, VALUE_TYPE3* __restrict outvec)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > num_other)
		return;

	int a = offsets[id];
	int b = offsets[min(id + 1, num_other)];

	int blocka = divup<int>(a, NNZ_PER_BLOCK);
	int blockb = (b - 1) / static_cast<int>(NNZ_PER_BLOCK);

	//iterate over all blocks that start with that row
	if(a != b)
		for (; blocka <= blockb; ++blocka)
			startingIds[blocka] = id;

	//write last
	if (id == num_other)
		startingIds[divup<int>(b, NNZ_PER_BLOCK)] = id - 1;
	else
		outvec[id] = 0;
}

template<typename OFFSET_TYPE, uint32_t NNZ_PER_BLOCK, typename VALUE_TYPE3>
__global__ void DetermineBlockStartsTranspose(int num_other, int num_this, const OFFSET_TYPE*__restrict offsets, uint32_t* startingIds, VALUE_TYPE3* __restrict outvec)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < num_this)
		outvec[id] = 0;

	if (id > num_other)
		return;

	int a = offsets[id];
	int b = offsets[min(id + 1, num_other)];

	int blocka = divup<int>(a, NNZ_PER_BLOCK);
	int blockb = (b - 1) / static_cast<int>(NNZ_PER_BLOCK);

	//iterate over all blocks that start with that row
	if (a != b)
		for (; blocka <= blockb; ++blocka)
			startingIds[blocka] = id;

	//write last
	if (id == num_other)
		startingIds[divup<int>(b, NNZ_PER_BLOCK)] = id - 1;
}


template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, uint32_t WARPWIDE_SWITCH, bool BLOCK_FLAG, bool ZERO_CHECK, bool SINGLE_ELEMENT>
class ThreadStarts
{
public:
	struct FlagSMem
	{
		uint32_t flags[THREADS + 1];// 1 block, THREADS/WARP_SIZE zero rows
		__device__ uint32_t zerorow()
		{
			if (ZERO_CHECK)
				return flags[1 + threadIdx.x / WARP_SIZE];
			else
				return 0;
		}

	};
	struct SMem
	{
		uint32_t starting_rows[THREADS];
		uint32_t end_row_bits[THREADS];
		int fetched_offsets[THREADS * NNZ_PER_THREAD + 2];

		__device__ __forceinline__ uint32_t startingRow()
		{
			return starting_rows[threadIdx.x];
		}
		__device__ __forceinline__ uint32_t bitmask()
		{
			return end_row_bits[threadIdx.x];
		}
		__device__ __forceinline__ int block_start_offset(FlagSMem& flagsmem)
		{
			if (WARPWIDE_SWITCH >= THREADS * NNZ_PER_THREAD)
				return flagsmem.flags[THREADS] - blockIdx.x * THREADS*NNZ_PER_THREAD;
			else
				return fetched_offsets[0];
		}
	};

	enum class Flags : uint32_t {
		Single = 0x0,
		OnePerThread = 0x1,
		ZeroFullDynamic = 0x0000FFFF,
		ZeroOffset = 0x8000,
	};

private:
	template<class OFFSET_TYPE>
	__device__ __forceinline__ uint32_t runSimple(SMem& smem, FlagSMem& flagmem, uint32_t blockRowStart, uint32_t blockNumRows, const OFFSET_TYPE* __restrict offsets)
	{

		uint32_t flag = 0;
		for (uint32_t r = threadIdx.x; r < blockNumRows; r += THREADS)
		{
			int ain = static_cast<int>(offsets[r + blockRowStart] - blockIdx.x * THREADS*NNZ_PER_THREAD);
			int bin = offsets[r + blockRowStart + 1] - blockIdx.x * THREADS*NNZ_PER_THREAD;

			if (BLOCK_FLAG)
			{
				uint32_t n = min(0x7FFF, bin - ain);
				flag = flag | n | ((~n) << 16);
			}

			int a = max(0, ain);
			int b = min(static_cast<int>(THREADS*NNZ_PER_THREAD), bin) - 1;

			int threada = divup<int>(a, static_cast<int>(NNZ_PER_THREAD));
			int threadb = b / static_cast<int>(NNZ_PER_THREAD);

			if (ZERO_CHECK && ain == bin)
				atomicMax(&flagmem.flags[1 + max(0,a-1) / (WARP_SIZE*NNZ_PER_THREAD)], r+2);

			//iterate over all threads that start with that row
			for (; threada <= threadb; ++threada)
				smem.starting_rows[threada] = r;

			uint32_t bitmask_value = 0x1 << (b - threadb*NNZ_PER_THREAD);
			atomicOr(&smem.end_row_bits[threadb], bitmask_value);

		}
		if (BLOCK_FLAG)
		{
			if (flag)
			{
				uint32_t f_other = __shfl(flag, 0);
				bool diff = __any(f_other != flag);
				if (laneid() == 0)
				{
					uint32_t offsetflag = (smem.fetched_offsets[0] == 0) ? static_cast<uint32_t>(Flags::ZeroOffset) : 0;
					flag = diff ? 0xFFFFFFFF : flag | offsetflag;
					atomicOr(flagmem.flags, flag);
				}
			}
			__syncthreads();
			if (((~flagmem.flags[0]) >> 16) == (flagmem.flags[0] & 0x7FFF))
			{
				//if all the warps reported same behavior, everything is deterministic, so we can quick out here
				return flagmem.flags[0] & 0xFFFF;
			}
		}
		return 0xFFFFFFFF;
	}


	template<class OFFSET_TYPE>
	__device__ __forceinline__ void runThread(SMem& smem, FlagSMem& flagmem, uint32_t blockRowStart, uint32_t blockNumRows, const OFFSET_TYPE* __restrict offsets)
	{
		for (uint32_t r = threadIdx.x; r < blockNumRows; r += THREADS)
		{
			int ain = smem.fetched_offsets[r];
			int bin = smem.fetched_offsets[r + 1];

			uint32_t a = max(0, ain);
			uint32_t threada = divup<uint32_t>(a, NNZ_PER_THREAD);

			if (ZERO_CHECK && ain == bin)
				flagmem.flags[1 + threada / WARP_SIZE] = 1;
			else
			{
				uint32_t b = min(THREADS*NNZ_PER_THREAD, bin) - 1;
				uint32_t threadb = b / NNZ_PER_THREAD;

				//iterate over all threads that start with that row
				for (; threada <= threadb; ++threada)
					smem.starting_rows[threada] = r;

				uint32_t bitmask_value = 0x1 << (b - threadb*NNZ_PER_THREAD);
				atomicOr(&smem.end_row_bits[threadb], bitmask_value);
			}
		}
	}

	template<class OFFSET_TYPE>
	__device__ __forceinline__ void runWarp(SMem& smem, FlagSMem& flagmem, uint32_t blockRowStart, uint32_t blockNumRows, const OFFSET_TYPE* __restrict offsets)
	{
		for (uint32_t r = threadIdx.x / WARP_SIZE; r < blockNumRows; r += THREADS / WARP_SIZE)
		{
			int ain = smem.fetched_offsets[r];
			int bin = smem.fetched_offsets[r + 1];
			uint32_t a = max(0, ain);
			uint32_t threada = divup<uint32_t>(a, NNZ_PER_THREAD);

			if (ZERO_CHECK && ain == bin)
				flagmem.flags[1 + threada / WARP_SIZE] = 1;
			else
			{
				uint32_t b = min(THREADS*NNZ_PER_THREAD, bin) - 1;
				uint32_t threadb = b / NNZ_PER_THREAD;

				//iterate over all threads that start with that row
				for (uint32_t i = threada + laneid(); i <= threadb; i += WARP_SIZE)
					smem.starting_rows[i] = r;

				if (laneid() == 0)
				{
					uint32_t bitmask_value = 0x1 << (b - threadb*NNZ_PER_THREAD);
					atomicOr(&smem.end_row_bits[threadb], bitmask_value);
				}
			}
		}
	}
public:
	template<class OFFSET_TYPE>
	__device__ __forceinline__ void initSingle(SMem& smem, FlagSMem& flagsmem, uint32_t blockRowStart, uint32_t blockNumRows, const OFFSET_TYPE* __restrict offsets)
	{
		if (BLOCK_FLAG && WARPWIDE_SWITCH >= THREADS * NNZ_PER_THREAD)
			flagsmem.flags[THREADS] = offsets[blockRowStart];
	}
	__device__ __forceinline__ void init(SMem& smem, FlagSMem& flagmem)
	{
		flagmem.flags[threadIdx.x] = 0;
		smem.end_row_bits[threadIdx.x] = 0;
	}
	template<class OFFSET_TYPE>
	__device__ __forceinline__ uint32_t run(SMem& smem, FlagSMem& flagmem, uint32_t blockRowStart, uint32_t blockNumRows, const OFFSET_TYPE* __restrict offsets)
	{
		if (SINGLE_ELEMENT && blockNumRows == 1)
		{
			//check for only a single row and fast out
			return static_cast<uint32_t>(Flags::Single);
		}
		if (BLOCK_FLAG && !ZERO_CHECK && blockNumRows == THREADS*NNZ_PER_THREAD + 1)
		{
			//if we do not support zeros, we can simply check for row count being
			// equal to the nonzeros the block will do and fast out.
			return static_cast<uint32_t>(Flags::OnePerThread);
		}

		if (ZERO_CHECK && blockNumRows > THREADS * NNZ_PER_THREAD + 1)
		{
			//if we have more rows than we will do nnz then there must be some non zero rows 
			//runSimple(smem, blockRowStart, blockNumRows, offsets);
			return static_cast<uint32_t>(Flags::ZeroFullDynamic);
		}

		if (WARPWIDE_SWITCH >= THREADS * NNZ_PER_THREAD)
		{
			return runSimple(smem, flagmem, blockRowStart, blockNumRows, offsets);
		}
		else
		{
			uint32_t flag = 0;
			//prefetch offsets to shared, will need them at least twice, potentially multiple times.
			// if we only needed them twice and had a proper memory access pattern, L1 would help us out a lot
			for (uint32_t r = threadIdx.x; r < blockNumRows + (BLOCK_FLAG ? 0 : 1); r += THREADS)
			{
				int ain = static_cast<int>(offsets[r + blockRowStart] - blockIdx.x * THREADS*NNZ_PER_THREAD);
				smem.fetched_offsets[r] = ain;
				if (BLOCK_FLAG)
				{
					int bin = static_cast<int>(offsets[r + blockRowStart + 1] - blockIdx.x * THREADS*NNZ_PER_THREAD);
					smem.fetched_offsets[r + 1] = bin;
					//default behavior, figuring out if there is a common state accross the entire block

					uint32_t n = min(0x7FFF, bin - ain);
					flag = flag | n | ((~n) << 16);
				}
			}

			if (BLOCK_FLAG)
			{
				if (flag)
				{
					uint32_t f_other = __shfl(flag, 0);
					bool diff = __any(f_other != flag);
					if (laneid() == 0)
					{
						uint32_t offsetflag = (smem.fetched_offsets[0] == 0) ? static_cast<uint32_t>(Flags::ZeroOffset) : 0;
						flag = diff ? 0xFFFFFFFF : flag | offsetflag;
						atomicOr(flagmem.flags, flag);
					}
				}
			}

			__syncthreads();

			if (BLOCK_FLAG)
			{
				if (((~flagmem.flags[0]) >> 16) == (flagmem.flags[0] & 0x7FFF))
				{
					//if all the warps reported same behavior, everything is deterministic, so we can quick out here
					return flagmem.flags[0] & 0xFFFF;
				}
			}


			//if (BLOCK_FLAG)
			//{
			//	default behavior, figuring out if there is a common state accross the entire block
			//	uint32_t flag = 0;
			//	for (uint32_t r = threadIdx.x; r < blockNumRows; r += THREADS)
			//	{
			//		int ain = smem.fetched_offsets[r];
			//		int bin = smem.fetched_offsets[(r + 1)];
			//		uint32_t n = min(0x7FFF, bin - ain);
			//		flag = flag | n | ((~n) << 16);
			//	}
			//	if (flag)
			//	{
			//		uint32_t f_other = __shfl(flag, 0);
			//		bool diff = __any(f_other != flag);
			//		if (laneid() == 0)
			//		{
			//			uint32_t offsetflag = (smem.fetched_offsets[0] == 0) ? static_cast<uint32_t>(Flags::ZeroOffset) : 0;
			//			flag = diff ? 0xFFFFFFFF : flag | offsetflag;
			//			atomicOr(flagmem.flags, flag);
			//		}
			//	}
			//	__syncthreads();
			//	if (((~flagmem.flags[0]) >> 16) == (flagmem.flags[0] & 0x7FFF))
			//	{
			//		if all the warps reported same behavior, everything is deterministic, so we can quick out here
			//		return flagmem.flags[0] & 0xFFFF;
			//	}
			//}

			//no special case, lets determine the the state for every thread
			if (blockNumRows <= WARPWIDE_SWITCH)
				//if there are only few rows, using an entire warp per row is better, we can scatter the data faster
				runWarp(smem, flagmem, blockRowStart, blockNumRows, offsets);
			else
				//a large number of rows means that probably every thread needs a different state, so using one thread per
				// input pair should result in a good performance
				runThread(smem, flagmem, blockRowStart, blockNumRows, offsets);
		}

		return 0xFFFFFFFF;
	}
};

template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, typename VALUE_TYPE3>
struct OutputBuffer
{
	VALUE_TYPE3 buffer[THREADS*NNZ_PER_THREAD];

	template<class COUNTER_SMEM>
	__device__ __forceinline__ void writeBufferTwoAtLeast(VALUE_TYPE3* __restrict outVec, COUNTER_SMEM& counterSmem)
	{
		writeInner(outVec, counterSmem);

		//write first and last atomically
		if (threadIdx.x < 2)
		{
			uint32_t address = threadIdx.x*(counterSmem.bNum - 1);
			tempAtomicAdd(outVec + counterSmem.bStart + address, buffer[address]);
		}
	}

	template<class COUNTER_SMEM>
	__device__ __forceinline__ void writeBufferTwoChecked(VALUE_TYPE3* __restrict outVec, COUNTER_SMEM& counterSmem, bool writeSecond)
	{
		writeInner(outVec, counterSmem);

		//write first and last atomically
		if (threadIdx.x == 0 || writeSecond && threadIdx.x == 1)
		{
			uint32_t address = threadIdx.x*(counterSmem.bNum - 1);
			tempAtomicAdd(outVec + counterSmem.bStart + address, buffer[address]);
		}
	}

	template<class COUNTER_SMEM>
	__device__ __forceinline__ void writeBuffer(VALUE_TYPE3* __restrict outVec, COUNTER_SMEM& counterSmem, bool last)
	{
		writeInner(outVec, counterSmem);

		//write first and last atomically
		if (threadIdx.x == 0 || last)
		{
			uint32_t address = (threadIdx.x == 0 ? 0 : (counterSmem.bNum - 1));
			tempAtomicAdd(outVec + counterSmem.bStart + address, buffer[address]);
		}
	}

	template<class COUNTER_SMEM>
	__device__ __forceinline__ void writeInner(VALUE_TYPE3* __restrict outVec, COUNTER_SMEM& counterSmem)
	{
		uint32_t blockOut = counterSmem.bStart;
		uint32_t wStart = blockOut / 32 * 32;
		uint32_t shift = blockOut - wStart;
		uint32_t num = shift + counterSmem.bNum - 1;

		//write all but first and last while makeing sure we never run across cache lines
		if (threadIdx.x > shift && threadIdx.x < num)
			outVec[wStart + threadIdx.x] = buffer[threadIdx.x - shift];
		if (threadIdx.x + THREADS < num)
			ConditionalIteration<NNZ_PER_THREAD>::iterate([&](uint32_t i) {
			uint32_t offset = threadIdx.x + (i + 1)*THREADS;
			outVec[wStart + offset] = buffer[offset - shift];
			return (offset + THREADS) < num;
		});

	}
};

template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, bool PADDEDLOAD, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, int SORTED_LOAD>
struct ValueLoader;

template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE>
struct ValueLoader<THREADS, NNZ_PER_THREAD, true, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, 0>
{
	using SMem = int;

	__device__ __forceinline__
		static void syncSmem()
	{
	}

	__device__ __forceinline__
		static void loadAndPremultiply(SMem& smem, VALUE_TYPE3(&values)[NNZ_PER_THREAD], uint32_t nnz, const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const VALUE_TYPE2* __restrict inVec)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);
		const int VecSize = static_min<MaxVecLoadIndex, MaxVecLoadMat, NNZ_PER_THREAD>::value;

		//full warp load case
		INDEX_TYPE index_in[NNZ_PER_THREAD];
		warp_load_vectorized<VecSize>(index_in, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));

		VALUE_TYPE2 vec_in[NNZ_PER_THREAD];
		#pragma unroll
		for (int i = 0; i < NNZ_PER_THREAD; ++i)
			vec_in[i] = inVec[index_in[i]];

		VALUE_TYPE1 mat_in[NNZ_PER_THREAD];
		warp_load_vectorized<VecSize>(mat_in, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));

		#pragma unroll
		for (int i = 0; i < NNZ_PER_THREAD; ++i)
			values[i] = vec_in[i] * mat_in[i];

		vectorized_to_blocked<VecSize>(values);
	}

	__device__
		static void loadWithIndices(SMem& smem, VALUE_TYPE3(&values)[NNZ_PER_THREAD], INDEX_TYPE(&indices)[NNZ_PER_THREAD], const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);

		warp_load_vectorized<MaxVecLoadMat>(values, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));
		vectorized_to_blocked<MaxVecLoadMat>(values);

		warp_load_vectorized<MaxVecLoadIndex>(indices, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));
		vectorized_to_blocked<MaxVecLoadIndex>(indices);

	}

	__device__
	static void loadValues(VALUE_TYPE3(&values)[NNZ_PER_THREAD], const VALUE_TYPE1* matrix, uint32_t nnz)
	{
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);
		warp_load_vectorized<MaxVecLoadMat>(values, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));
		vectorized_to_blocked<MaxVecLoadMat>(values);
	}

	__device__
	static void loadIndices(INDEX_TYPE(&indices)[NNZ_PER_THREAD], const INDEX_TYPE* inIndex, uint32_t nnz)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		warp_load_vectorized<MaxVecLoadIndex>(indices, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));
		vectorized_to_blocked<MaxVecLoadIndex>(indices);
	}
};

template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE>
struct ValueLoader<THREADS, NNZ_PER_THREAD, false, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, 0>
{
	using SMem = int;

	__device__ __forceinline__
		static void syncSmem()
	{
	}

	__device__ __forceinline__
		static void loadAndPremultiply(SMem& smem, VALUE_TYPE3(&values)[NNZ_PER_THREAD], uint32_t nnz, const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const VALUE_TYPE2* __restrict inVec)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);
		const int VecSize = static_min<MaxVecLoadIndex, MaxVecLoadMat, NNZ_PER_THREAD>::value;

		uint32_t warp_offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x / WARP_SIZE*WARP_SIZE*NNZ_PER_THREAD;
		if (warp_offset + WARP_SIZE*NNZ_PER_THREAD <= nnz)
		{
			//full warp load case
			INDEX_TYPE index_in[NNZ_PER_THREAD];
			warp_load_vectorized<VecSize>(index_in, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));

			VALUE_TYPE2 vec_in[NNZ_PER_THREAD];
			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				vec_in[i] = inVec[index_in[i]];

			VALUE_TYPE1 mat_in[NNZ_PER_THREAD];
			warp_load_vectorized<VecSize>(mat_in, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));

			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				values[i] = vec_in[i] * mat_in[i];

			vectorized_to_blocked<VecSize>(values);
		}
		else if (warp_offset < nnz)
		{
			//warp is partially filled... load it in a naive way
			uint32_t offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x*NNZ_PER_THREAD;

			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (offset + i < nnz)
					values[i] = inVec[inIndex[offset + i]] * matrix[offset + i];
				else
					values[i] = 0;
			}
		}
		else
		{
			//full warp out of bounds -> fill with 0
			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				values[i] = 0;
		}
	}

	__device__
		static void loadValues(VALUE_TYPE3(&values)[NNZ_PER_THREAD], const VALUE_TYPE1* matrix, uint32_t nnz)
	{
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);

		uint32_t warp_offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x / WARP_SIZE * WARP_SIZE*NNZ_PER_THREAD;
		if (warp_offset + WARP_SIZE * NNZ_PER_THREAD <= nnz)
		{
			//full warp load case
			warp_load_vectorized<MaxVecLoadMat>(values, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));
			vectorized_to_blocked<MaxVecLoadMat>(values);
		}
		else if (warp_offset < nnz)
		{
			//warp is partially filled... load it in a naive way
			uint32_t offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x*NNZ_PER_THREAD;

			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (offset + i < nnz)
					values[i] = matrix[offset + i];
				else
					values[i] = 0;
			}
		}
		else
		{
			//full warp out of bounds -> fill with 0
			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				values[i] = 0;
		}
	}

	__device__
		static void loadIndices(INDEX_TYPE(&indices)[NNZ_PER_THREAD], const INDEX_TYPE* inIndex, uint32_t nnz)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		uint32_t warp_offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x / WARP_SIZE * WARP_SIZE*NNZ_PER_THREAD;
		if (warp_offset + WARP_SIZE * NNZ_PER_THREAD <= nnz)
		{
			//full warp load case
			warp_load_vectorized<MaxVecLoadIndex>(indices, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));
			vectorized_to_blocked<MaxVecLoadIndex>(indices);
		}
		else if (warp_offset < nnz)
		{
			//warp is partially filled... load it in a naive way
			uint32_t offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x*NNZ_PER_THREAD;

			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (offset + i < nnz)
					indices[i] = inIndex[offset + i];
				else
					indices[i] = 0;
			}
		}
		else
		{
			//full warp out of bounds -> fill with 0
			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				indices[i] = 0;
		}
	}
};

template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE>
struct ValueLoader<THREADS, NNZ_PER_THREAD, false, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, 1>
{
	struct SMem
	{
		VALUE_TYPE2 inputBuffer[THREADS*NNZ_PER_THREAD];
	};

	__device__ __forceinline__
		static void syncSmem()
	{
		__syncthreads();
	}

	__device__ __forceinline__
		static void loadAndPremultiply(SMem& smem, VALUE_TYPE3(&values)[NNZ_PER_THREAD], uint32_t nnz, const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const VALUE_TYPE2* __restrict inVec)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);
		const int VecSize = static_min<MaxVecLoadIndex, MaxVecLoadMat, NNZ_PER_THREAD>::value;

		uint32_t warp_offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x / WARP_SIZE*WARP_SIZE*NNZ_PER_THREAD;
		if (warp_offset + WARP_SIZE*NNZ_PER_THREAD <= nnz)
		{
			//full warp load case

			INDEX_TYPE index_in[NNZ_PER_THREAD];
			warp_load_vectorized<VecSize>(index_in, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));

			int tpos[NNZ_PER_THREAD];
			for (int j = 0; j < NNZ_PER_THREAD; ++j)
				tpos[j] = j;

			threadOddEvenMergeSort<SortAscending>(index_in, tpos);


			#pragma unroll
			for (int j = 0; j < NNZ_PER_THREAD; ++j)
				smem.inputBuffer[threadIdx.x + tpos[j] * THREADS] = inVec[index_in[j]];

			VALUE_TYPE1 mat_in[NNZ_PER_THREAD];
			warp_load_vectorized<VecSize>(mat_in, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));


			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				values[i] = mat_in[i] * smem.inputBuffer[threadIdx.x + i*THREADS];

			vectorized_to_blocked<VecSize>(values);
		}
		else if (warp_offset < nnz)
		{
			//warp is partially filled... load it in a naive way
			uint32_t offset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x*NNZ_PER_THREAD;

			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (offset + i < nnz)
					values[i] = inVec[inIndex[offset + i]] * matrix[offset + i];
				else
					values[i] = 0;
			}
		}
		else
		{
			//full warp out of bounds -> fill with 0
			#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				values[i] = 0;
		}
	}
};

template<uint32_t THREADS, uint32_t NNZ_PER_THREAD, typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE>
struct ValueLoader<THREADS, NNZ_PER_THREAD, true, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, 1>
{
	struct SMem
	{
		VALUE_TYPE2 inputBuffer[THREADS*NNZ_PER_THREAD];
	};

	__device__ __forceinline__
		static void syncSmem()
	{
		__syncthreads();
	}

	__device__ __forceinline__
		static void loadAndPremultiply(SMem& smem, VALUE_TYPE3(&values)[NNZ_PER_THREAD], uint32_t nnz, const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const VALUE_TYPE2* __restrict inVec)
	{
		const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
		const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);
		const int VecSize = static_min<MaxVecLoadIndex, MaxVecLoadMat, NNZ_PER_THREAD>::value;

		INDEX_TYPE index_in[NNZ_PER_THREAD];
		warp_load_vectorized<VecSize>(index_in, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));

		int tpos[NNZ_PER_THREAD];
		for (int j = 0; j < NNZ_PER_THREAD; ++j)
			tpos[j] = j;

		threadOddEvenMergeSort<SortAscending>(index_in, tpos);


#pragma unroll
		for (int j = 0; j < NNZ_PER_THREAD; ++j)
			smem.inputBuffer[threadIdx.x + tpos[j] * THREADS] = inVec[index_in[j]];

		VALUE_TYPE1 mat_in[NNZ_PER_THREAD];
		warp_load_vectorized<VecSize>(mat_in, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));


#pragma unroll
		for (int i = 0; i < NNZ_PER_THREAD; ++i)
			values[i] = mat_in[i] * smem.inputBuffer[threadIdx.x + i*THREADS];

		vectorized_to_blocked<VecSize>(values);
	}
};

template<class ThreadStarts, uint32_t THREADS, uint32_t NNZ_PER_THREAD, bool PADDEDLOAD,
	typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE,
	bool ZEROCHECK, bool SCALE, int SORTED_LOAD,
	class ToHandleOptimizations, class HandledOptimizations>
	class Multiplication;


template<class ThreadStarts, uint32_t THREADS, uint32_t NNZ_PER_THREAD, bool PADDEDLOAD,
	typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE,
	bool ZEROCHECK, bool SCALE, int SORTED_LOAD, int HANDLED>
	struct Multiplication<ThreadStarts, THREADS, NNZ_PER_THREAD, PADDEDLOAD,
	VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, OFFSET_TYPE,
	ZEROCHECK, SCALE, SORTED_LOAD, sequence<>, sequence<HANDLED>>
{
public:
	using ValueLoader = ValueLoader<THREADS, NNZ_PER_THREAD, PADDEDLOAD, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, SORTED_LOAD>;
	using OutputBuffer = ::OutputBuffer<THREADS, NNZ_PER_THREAD, VALUE_TYPE3>;
	struct SMem
	{
		union {
			typename ValueLoader::SMem valueloader;
			OutputBuffer outputBuffer;
			uint32_t dynamicRowStarts[THREADS*NNZ_PER_THREAD];
		};

	};

	template<class COUNTER_SMEM, class THREADINFO_SMEM>
	__device__ __forceinline__
		static void run(uint32_t flag,
			const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const OFFSET_TYPE* __restrict offsets, const VALUE_TYPE2* __restrict inVec,
			VALUE_TYPE3* __restrict outVec, VALUE_TYPE3 scale, uint32_t nnz,
			COUNTER_SMEM& counterSmem, THREADINFO_SMEM& threadInfoSmem, SMem& smem)
	{
		//general implementation
		__syncthreads();
		uint32_t my_row = min(threadInfoSmem.startingRow(), counterSmem.bNum);
		uint32_t my_bitmask = threadInfoSmem.bitmask();

		ValueLoader::syncSmem();

		//if (counterSmem.bStart >= 73649 && counterSmem.bStart < 73980)
		//	printf("ok22");

		VALUE_TYPE3 values[NNZ_PER_THREAD];
		ValueLoader::loadAndPremultiply(smem.valueloader, values, nnz, matrix, inIndex, inVec);
		__syncthreads();

		VALUE_TYPE3 res = 0;

		if (ZEROCHECK && flag == static_cast<uint32_t>(ThreadStarts::Flags::ZeroFullDynamic))
		{
			//we have got many zero rows, we cannot even buffer the output, so everything needs to
			//be done dynamically

			//set row starts to -1 for every element
#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				smem.dynamicRowStarts[i*THREADS + threadIdx.x] = 0xFFFFFFFF;
			__syncthreads();
			for (uint32_t r = threadIdx.x; r < counterSmem.bNum; r += THREADS)
			{
				uint32_t ain = offsets[r + counterSmem.bStart];
				//fast out for empty rows at the end of the marix...
				if (ain == nnz)
					break;
				int bin = offsets[r + counterSmem.bStart + 1] - blockIdx.x * THREADS*NNZ_PER_THREAD;
				int a = max(0, static_cast<int>(ain - blockIdx.x * THREADS*NNZ_PER_THREAD));
				int b = min(static_cast<int>(THREADS*NNZ_PER_THREAD), bin) - 1;

				uint32_t grow = counterSmem.bStart + r;
				if (a <= b)
					smem.dynamicRowStarts[b / NNZ_PER_THREAD + (b%NNZ_PER_THREAD)*THREADS] = grow;
				//mark last elements for all threads that share that row
				int thread_ends = (a / static_cast<int>(NNZ_PER_THREAD) + 1)*static_cast<int>(NNZ_PER_THREAD) - 1;
				for (; thread_ends < b; thread_ends += NNZ_PER_THREAD)
					smem.dynamicRowStarts[thread_ends / NNZ_PER_THREAD + (thread_ends%NNZ_PER_THREAD)*THREADS] = grow;
			}
			__syncthreads();
#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (SCALE)
					res += values[i] * scale;
				else
					res += values[i];
				uint32_t trow = smem.dynamicRowStarts[threadIdx.x + i*THREADS];
				if (trow != 0xFFFFFFFF)
				{
					tempAtomicAdd(outVec + trow, res);
					res = 0;
				}
			}
			return;
		}
		else if (ZEROCHECK && counterSmem.flags.zerorow() != 0)
		{
			//we have zero element rows in this warp, need to work over them dynamically

			//within the warp zero everything from the first row of the warp until the end of the warps region
			uint32_t first_start = __shfl(my_row, 0);
			for (uint32_t r = first_start + laneid(); r < counterSmem.flags.zerorow() - 1; r += WARP_SIZE)
			{
				smem.outputBuffer.buffer[r] = 0;
				//if (counterSmem.bStart + r == 73649)
				//	printf("ok");
			}

			__threadfence_block();

			//every threads needs to do its part dynamically now
			OFFSET_TYPE el_offset = blockIdx.x * (THREADS*NNZ_PER_THREAD) + threadIdx.x * NNZ_PER_THREAD;
			OFFSET_TYPE next_row_offset = offsets[counterSmem.bStart + my_row + 1];

#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (el_offset == next_row_offset)
				{
					smem.outputBuffer.buffer[my_row] = res;
					next_row_offset = offsets[counterSmem.bStart + my_row + 2];
					++my_row;
					res = 0;
					while (el_offset == next_row_offset)
					{
						next_row_offset = offsets[counterSmem.bStart + my_row + 2];
						++my_row;
					}
				}
				if (SCALE)
					res += values[i] * scale;
				else
					res += values[i];
				++el_offset;
			}
			if (el_offset == next_row_offset || ((my_bitmask & (0x1 << NNZ_PER_THREAD - 1)) != 0))
			{
				smem.outputBuffer.buffer[my_row] = res;
				next_row_offset = offsets[counterSmem.bStart + my_row + 2];
				++my_row;
				res = 0;
				while (el_offset == next_row_offset)
				{
					next_row_offset = offsets[counterSmem.bStart + my_row + 2];
					++my_row;
				}
			}
		}
		else
		{
			//regular case
#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			{
				if (SCALE)
					res += values[i] * scale;
				else
					res += values[i];
				if ((my_bitmask & (0x1 << i)) != 0)
				{
					smem.outputBuffer.buffer[my_row] = res;
					++my_row;
					res = 0;
				}
			}
		}
		//if (threadIdx.x == THREADS - 1 && my_row == counterSmem.bNum - 1 && res == 0)
		//	smem.outputBuffer.buffer[my_row] = res;
		__syncthreads();
		//TODO: figure out the right point to switch to a reduction
		//do a per warp reduction in case we expect too many collisions
		if (__popc(__ballot(my_bitmask)) < 16)
		{
			//use a reduction to combine the elements
			bool pivot = my_bitmask != 0;
			bool pivot_right = laneid() == 31;

			//conditional reduction
#pragma unroll
			for (int offset = 1; offset < WARP_SIZE; offset *= 2)
			{
				VALUE_TYPE3 incoming_res = __shfl_down(pivot ? 0.0f : res, offset);
				if (!pivot_right)
					res += incoming_res;
				bool incoming_pivot = __shfl_down(pivot || pivot_right, offset);
				pivot_right = pivot_right || incoming_pivot;
			}
			if ((pivot || laneid() == 0) && res != 0)
				tempAtomicAdd(smem.outputBuffer.buffer + my_row, res);
		}
		else
		{
			//write the carryover with atomic
			if (res != 0)
				tempAtomicAdd(smem.outputBuffer.buffer + my_row, res);
		}
		
		__syncthreads();
		if (sequence_any<static_is_zero, sequence<HANDLED>>::value)
		{
			uint32_t nnz_block = nnz - blockIdx.x*(THREADS*NNZ_PER_THREAD);
			//smem.outputBuffer.writeBufferTwoAtLeast(outVec, counterSmem);
			//bool writelast = my_row == counterSmem.bNum  && ((blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x*NNZ_PER_THREAD < nnz));
			bool writelast = (my_row == counterSmem.bNum && (threadIdx.x*NNZ_PER_THREAD < nnz_block))
				|| (static_cast<int>(nnz_block / NNZ_PER_THREAD) - 1 == -threadIdx.x);
			// && (threadIdx.x*NNZ_PER_THREAD < max(NNZ_PER_THREAD + 1, (nnz - blockIdx.x*(THREADS*NNZ_PER_THREAD))));
			//				&& ((my_bitmask & (0x1 << NNZ_PER_THREAD - 1)) != 0);
			smem.outputBuffer.writeBuffer(outVec, counterSmem, writelast);
		}
			
		else
		{
			uint32_t nnz_block = nnz - blockIdx.x*(THREADS*NNZ_PER_THREAD);
			//bool writelast = my_row == counterSmem.bNum && my_row != 1 && ((blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x*NNZ_PER_THREAD < nnz));
			bool writelast = my_row != 1 && ((my_row == counterSmem.bNum && (threadIdx.x*NNZ_PER_THREAD < nnz_block)) 
				|| (static_cast<int>(nnz_block / NNZ_PER_THREAD) - 1 == -threadIdx.x));
			// && (threadIdx.x*NNZ_PER_THREAD < max(NNZ_PER_THREAD+1, (nnz - blockIdx.x*(THREADS*NNZ_PER_THREAD))));
//				&& ((my_bitmask & (0x1 << NNZ_PER_THREAD - 1)) != 0);
			smem.outputBuffer.writeBuffer(outVec, counterSmem, writelast);
		}
	}
};

template<class ThreadStarts, uint32_t THREADS, uint32_t NNZ_PER_THREAD,
	typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE,
	bool ZEROCHECK, bool SCALE, int SORTED_LOAD, int... UNHANDLED, int... HANDLED>
	struct Multiplication<ThreadStarts, THREADS, NNZ_PER_THREAD, true,
	VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, OFFSET_TYPE,
	ZEROCHECK, SCALE, SORTED_LOAD, sequence<0, UNHANDLED...>, sequence<HANDLED...>>
{
	using Next = Multiplication<ThreadStarts, THREADS, NNZ_PER_THREAD, true, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, OFFSET_TYPE, ZEROCHECK, SCALE, SORTED_LOAD, sequence<UNHANDLED...>, sequence<0, HANDLED...>>;
	using Reduction = cub::BlockReduce<VALUE_TYPE3, THREADS>;
	struct SMem
	{
		union
		{
			typename Next::SMem nextSmem;
			typename Reduction::TempStorage reductionStorage;
		};
	};

	template<class COUNTER_SMEM, class THREADINFO_SMEM>
	__device__ __forceinline__
		static void run(uint32_t flag,
			const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const OFFSET_TYPE* __restrict offsets, const VALUE_TYPE2* __restrict inVec,
			VALUE_TYPE3* __restrict outVec, VALUE_TYPE3 scale, uint32_t nnz,
			COUNTER_SMEM& counterSmem, THREADINFO_SMEM& threadInfoSmem, SMem& smem)
	{
		//special implementation for single row
		if (flag == static_cast<uint32_t>(ThreadStarts::Flags::Single))
		{


			VALUE_TYPE3 res = 0;
			uint32_t offset = blockIdx.x*(NNZ_PER_THREAD*THREADS) + threadIdx.x;
			//simply add all up
			#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			{
				INDEX_TYPE index = inIndex[offset + i*THREADS];
				VALUE_TYPE2 vec = inVec[index];
				VALUE_TYPE1 mat = matrix[offset + i*THREADS];
				res += vec * mat;
			}
			res = Reduction(smem.reductionStorage).Sum(res);
			if (threadIdx.x == 0)
			{
				if (SCALE) res = res * scale;
				tempAtomicAdd(outVec + counterSmem.bStart, res);
			}
		}
		else
			Next::run(flag, matrix, inIndex, offsets, inVec, outVec, scale, nnz, counterSmem, threadInfoSmem, smem.nextSmem);
	}
};
template<class ThreadStarts, uint32_t THREADS, uint32_t NNZ_PER_THREAD,
	typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE,
	bool ZEROCHECK, bool SCALE, int SORTED_LOAD, int... UNHANDLED, int... HANDLED>
	struct Multiplication<ThreadStarts, THREADS, NNZ_PER_THREAD, false,
	VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, OFFSET_TYPE,
	ZEROCHECK, SCALE, SORTED_LOAD, sequence<0, UNHANDLED...>, sequence<HANDLED...>>
{
	using Next = Multiplication<ThreadStarts, THREADS, NNZ_PER_THREAD, false, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, OFFSET_TYPE, ZEROCHECK, SCALE, SORTED_LOAD, sequence<UNHANDLED...>, sequence<0, HANDLED...>>;
	using Reduction = cub::BlockReduce<VALUE_TYPE3, THREADS>;
	struct SMem
	{
		union
		{
			typename Next::SMem nextSmem;
			typename Reduction::TempStorage reductionStorage;
		};
	};

	template<class COUNTER_SMEM, class THREADINFO_SMEM>
	__device__ __forceinline__
		static void run(uint32_t flag,
			const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const OFFSET_TYPE* __restrict offsets, const VALUE_TYPE2* __restrict inVec,
			VALUE_TYPE3* __restrict outVec, VALUE_TYPE3 scale, uint32_t nnz,
			COUNTER_SMEM& counterSmem, THREADINFO_SMEM& threadInfoSmem, SMem& smem)
	{
		//special implementation for single row
		if (flag == static_cast<uint32_t>(ThreadStarts::Flags::Single))
		{
			//simply add all up

			VALUE_TYPE3 res = 0;
			uint32_t offset = blockIdx.x*(NNZ_PER_THREAD*THREADS);
			if(offset + NNZ_PER_THREAD*THREADS <= nnz)
			{
				// full block case
				offset = offset + threadIdx.x;
				#pragma unroll
				for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
				{
					INDEX_TYPE index = inIndex[offset + i*THREADS];
					VALUE_TYPE2 vec = inVec[index];
					VALUE_TYPE1 mat = matrix[offset + i*THREADS];
					res += vec * mat;
				}
			}
			else
			{
				//this is the final block
				offset = offset + threadIdx.x;
				#pragma unroll
				for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
				{
					if(offset + i*THREADS < nnz)
					{ 
						INDEX_TYPE index = inIndex[offset + i*THREADS];
						VALUE_TYPE2 vec = inVec[index];
						VALUE_TYPE1 mat = matrix[offset + i*THREADS];
						res += vec * mat;
					}
				}
			}
			res = Reduction(smem.reductionStorage).Sum(res);
			if (threadIdx.x == 0)
			{
				if (SCALE) res = res * scale;
				tempAtomicAdd(outVec + counterSmem.bStart, res);
			}
		}
		else
			Next::run(flag, matrix, inIndex, offsets, inVec, outVec, scale, nnz, counterSmem, threadInfoSmem, smem.nextSmem);
	}
};


template<typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE,
	uint32_t NNZ_PER_BLOCK, uint32_t THREADS, uint32_t LAUNCHBOUNDS, bool PADDEDLOAD, bool ZERO_CHECK, bool SCALE, int SORTED_LOAD, uint32_t WARPWIDE_FLAG_SWITCH, int... SPECIALIZATIONS>
			__launch_bounds__(THREADS, LAUNCHBOUNDS)
	__global__ void MultiplyAlong(uint32_t num_non_zeroes, uint32_t out_size, uint32_t num_other,
		const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const OFFSET_TYPE* __restrict offsets, const VALUE_TYPE2* __restrict inVec,
		VALUE_TYPE3* __restrict outVec, VALUE_TYPE3 scale,
		uint32_t* __restrict startingIds)
{
	const int NNZ_PER_THREAD = NNZ_PER_BLOCK / THREADS;
	const bool SINGLEROWOPT = sequence_any<static_is_zero, sequence<SPECIALIZATIONS...>>::value;
	const bool BLOCK_FLAG = sizeof...(SPECIALIZATIONS) > (SINGLEROWOPT ? 1 : 0);
	const uint32_t WARPWIDE_SWITCH = BLOCK_FLAG ? WARPWIDE_FLAG_SWITCH : NNZ_PER_BLOCK;
	static_assert(NNZ_PER_THREAD * THREADS == NNZ_PER_BLOCK, "NNZ_PER_BLOCK must be evenly devisible by THREADS");

	using ThreadStarts = ::ThreadStarts < THREADS, NNZ_PER_THREAD, WARPWIDE_SWITCH, BLOCK_FLAG, ZERO_CHECK, SINGLEROWOPT>;
	using Multiplication = ::Multiplication<ThreadStarts, THREADS, NNZ_PER_THREAD, PADDEDLOAD, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, OFFSET_TYPE, ZERO_CHECK, SCALE, SORTED_LOAD, sequence<SPECIALIZATIONS...>, sequence<>>;

	struct SMem
	{
		uint32_t bStart, bNum;
		typename ThreadStarts::FlagSMem flags;
		union
		{
			typename ThreadStarts::SMem startsSmem;
			typename Multiplication::SMem multSmem;
		};
	};

	__shared__ SMem smem;

	ThreadStarts threadStarts;

	//if (threadIdx.x == 0 && blockIdx.x == 462 )
	//	printf("ok22");

	if (threadIdx.x == 0)
	{
		smem.bStart = startingIds[blockIdx.x];
		smem.bNum = startingIds[blockIdx.x + 1] - smem.bStart + 1;
		threadStarts.initSingle(smem.startsSmem, smem.flags, smem.bStart, smem.bNum, offsets);
	}

	//if (threadIdx.x == 0 && smem.bStart <= 399 && smem.bStart+smem.bNum >= 399)
	//	//if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1)
	//{
	//	printf("last");
	//}

	threadStarts.init(smem.startsSmem, smem.flags);
	__syncthreads();

	uint32_t flag = threadStarts.run(smem.startsSmem, smem.flags, smem.bStart, smem.bNum, offsets);
	Multiplication::run(flag, matrix, inIndex, offsets, inVec, outVec, scale, num_non_zeroes, smem, smem.startsSmem, smem.multSmem);
}




template<typename ACCESS_F, typename COLL_ACCESS_F, typename SEP_FUNCTION, class TYPE, class INDEXTYPE, uint32_t NNZ_PER_THREAD>
__device__ void sortedResolve(TYPE(&values)[NNZ_PER_THREAD], INDEXTYPE(&col_indices)[NNZ_PER_THREAD], ACCESS_F accessF, COLL_ACCESS_F collAccessF, SEP_FUNCTION sepF)
{
	// keep adding up
	TYPE out = values[0];
#pragma unroll
	for (uint32_t i = 1; i < NNZ_PER_THREAD; ++i)
	{
		if (col_indices[i - 1] != col_indices[i])
		{
			accessF(col_indices[i - 1], out);
			out = values[i];
		}
		else
			out = out + values[i];
	}
	sepF();
	INDEXTYPE lower_col = __shfl_up(col_indices[NNZ_PER_THREAD - 1], 1);
	bool pivot = lower_col != col_indices[NNZ_PER_THREAD - 1];
	if (__popc(__ballot(pivot)) >= 16)
	{
		//atomic out
		collAccessF(col_indices[NNZ_PER_THREAD - 1], out);
	}
	else
	{
		//conditional reduction
		bool pivot_right = laneid() == 31;
#pragma unroll
		for (int offset = 1; offset < WARP_SIZE; offset *= 2)
		{
			TYPE incoming_res = __shfl_down(pivot ? 0.0f : out, offset);
			if (!pivot_right)
				out += incoming_res;
			bool incoming_pivot = __shfl_down(pivot || pivot_right, offset);
			pivot_right = pivot_right || incoming_pivot;
		}
		if ((pivot || laneid() == 0))
			collAccessF(col_indices[NNZ_PER_THREAD - 1], out);
	}
}



template<typename VALUE_TYPE1, typename VALUE_TYPE2, typename VALUE_TYPE3, typename INDEX_TYPE, typename OFFSET_TYPE,
	uint32_t NNZ_PER_BLOCK, uint32_t THREADS, uint32_t LAUNCHBOUNDS, bool PADDEDLOAD, bool SCALE, bool SORTED_RESOLVE, bool BUFFERED_RESOLVE, int SORT_DIRECT, bool ROWCOUNT_DIRECT_THRESHOLD, uint32_t OUTPUT_BUFFER_SIZE_MIN, uint32_t OUTPUT_BUFFER_SIZE_MAX>
	__launch_bounds__(THREADS, LAUNCHBOUNDS)
	__global__ void MultiplyAlongTranspose(uint32_t num_non_zeroes, uint32_t out_size, uint32_t num_other,
		const VALUE_TYPE1* matrix, const INDEX_TYPE* inIndex, const OFFSET_TYPE* __restrict offsets, const VALUE_TYPE2* __restrict inVec,
		VALUE_TYPE3* __restrict outVec, VALUE_TYPE3 scale,
		uint32_t* __restrict startingIds)
{
	const int NNZ_PER_THREAD = NNZ_PER_BLOCK / THREADS;
	static_assert(NNZ_PER_THREAD * THREADS == NNZ_PER_BLOCK, "NNZ_PER_BLOCK must be evenly devisible by THREADS");

	using CubSorter = cub::BlockRadixSort<uint32_t, THREADS, NNZ_PER_THREAD, VALUE_TYPE3>;

	//automatically increase output buffer in accordance with already required smem and 100% occupancy
	const uint32_t OUTPUT_BUFFER_SIZE_CALC = static_max<OUTPUT_BUFFER_SIZE_MIN,
		4 * NNZ_PER_BLOCK / sizeof(VALUE_TYPE3),
		sizeof(typename CubSorter::TempStorage) / sizeof(VALUE_TYPE3),
		(98304 / 2048 * THREADS - 32) / sizeof(VALUE_TYPE3)>::value;
	const uint32_t OUTPUT_BUFFER_SIZE = !(SORTED_RESOLVE || BUFFERED_RESOLVE) ? 1 :
		OUTPUT_BUFFER_SIZE_MAX <= 0 ? OUTPUT_BUFFER_SIZE_CALC :
		static_min<OUTPUT_BUFFER_SIZE_MAX, OUTPUT_BUFFER_SIZE_CALC>::value;

	struct SMem
	{
		uint32_t bStart, bNum;
		uint32_t minCol, maxCol;
		union {
			VALUE_TYPE3 outBuffer[OUTPUT_BUFFER_SIZE];
			uint32_t row_offsets[NNZ_PER_THREAD*THREADS];
			typename CubSorter::TempStorage sortStorage;
		};
	};
	__shared__ SMem smem;

	if (threadIdx.x == 0)
	{
		if (!SORTED_RESOLVE && BUFFERED_RESOLVE)
		{
			smem.minCol = 0xFFFFFFFF;
			smem.maxCol = 0;
		}
		smem.bStart = startingIds[blockIdx.x];
		smem.bNum = startingIds[blockIdx.x + 1] - smem.bStart + 1;
	}

#pragma unroll
	for (int i = 0; i < NNZ_PER_THREAD; ++i)
	{
		smem.row_offsets[threadIdx.x + THREADS * i] = 0xFFFFFFFF;
	}
	__syncthreads();


	// special cases for single row
	// multiply with row value and atomic out... no overlap for sure
	if (smem.bNum == 1)
	{
		VALUE_TYPE3 mul = inVec[smem.bStart];
		uint32_t startOffset = blockIdx.x*(THREADS*NNZ_PER_THREAD) + threadIdx.x;
		if (SORT_DIRECT == 0)
		{
#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
			{
				uint32_t toffset = i * THREADS + startOffset;
				VALUE_TYPE3 val = matrix[toffset] * mul;
				if (SCALE) val = val * scale;
				INDEX_TYPE col_index = inIndex[toffset];
				if (val != 0)
					tempAtomicAdd(outVec + col_index, val);
			}
		}
		else
		{
			VALUE_TYPE3 values[NNZ_PER_THREAD];
			INDEX_TYPE col_indices[NNZ_PER_THREAD];
#pragma unroll
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
			{
				uint32_t toffset = i * THREADS + startOffset;
				VALUE_TYPE3 val = matrix[toffset] * mul;
				if (SCALE) val = val * scale;
				values[i] = val;
				col_indices[i] = inIndex[toffset];
			}
			if (SORT_DIRECT == 1)
				threadOddEvenMergeSort<SortAscending>(col_indices, values);
			else
				CubSorter(smem.sortStorage).SortBlockedToStriped(col_indices, values);
			for (int i = 0; i < NNZ_PER_THREAD; ++i)
				if (values[i] != 0)
					tempAtomicAdd(outVec + col_indices[i], values[i]);
		}
		return;
	}

	//load matrix
	VALUE_TYPE3 values[NNZ_PER_THREAD];
	//const int MaxVecLoadMat = 16 / sizeof(VALUE_TYPE1);
	//warp_load_vectorized<MaxVecLoadMat>(values, matrix + blockIdx.x*(THREADS*NNZ_PER_THREAD));
	//vectorized_to_blocked<MaxVecLoadMat>(values);

	ValueLoader<THREADS, NNZ_PER_THREAD, PADDEDLOAD, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, 0>::loadValues(values, matrix, num_non_zeroes);


	//mark new row starts (decode compressed)
	for (uint32_t r = threadIdx.x; r < smem.bNum; r += THREADS)
	{
		uint32_t ain = offsets[r + smem.bStart];
		//fast out for empty rows at the end of the marix...
		if (ain == num_non_zeroes)
			break;
		int bin = offsets[r + smem.bStart + 1] - blockIdx.x * THREADS*NNZ_PER_THREAD;
		int a = max(0, static_cast<int>(ain - blockIdx.x * THREADS*NNZ_PER_THREAD));
		int b = min(static_cast<int>(THREADS*NNZ_PER_THREAD), bin);

		uint32_t grow = smem.bStart + r;
		if (a < b)
			smem.row_offsets[a / NNZ_PER_THREAD + (a%NNZ_PER_THREAD)*THREADS] = grow;
		//mark all first elements for all threads that share that row
		int thread_starts = (a / static_cast<int>(NNZ_PER_THREAD) + 1)*static_cast<int>(NNZ_PER_THREAD);
		for (; thread_starts < b; thread_starts += NNZ_PER_THREAD)
			smem.row_offsets[thread_starts / NNZ_PER_THREAD + (thread_starts%NNZ_PER_THREAD)*THREADS] = grow;
	}
	__syncthreads();
	VALUE_TYPE3 mul = 0;
#pragma unroll
	for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
	{
		uint32_t trow = smem.row_offsets[threadIdx.x + i * THREADS];
		if (trow != 0xFFFFFFFF)
			mul = inVec[trow];
		values[i] = values[i] * mul;
	}

	// load offset column ids
	INDEX_TYPE col_indices[NNZ_PER_THREAD];
	//const int MaxVecLoadIndex = 16 / sizeof(INDEX_TYPE);
	//warp_load_vectorized<MaxVecLoadIndex>(col_indices, inIndex + blockIdx.x*(THREADS*NNZ_PER_THREAD));
	//vectorized_to_blocked<MaxVecLoadIndex>(col_indices);
	ValueLoader<THREADS, NNZ_PER_THREAD, PADDEDLOAD, VALUE_TYPE1, VALUE_TYPE2, VALUE_TYPE3, INDEX_TYPE, 0>::loadIndices(col_indices, inIndex, num_non_zeroes);


	//rowcount < THRESHOLD: gains from coordination probably small, so no buffered out 
	// just atomic out (SORT_DIRECT -> sort for better out mem pattern)
	if ((SORTED_RESOLVE || BUFFERED_RESOLVE) && smem.bNum <= ROWCOUNT_DIRECT_THRESHOLD)
	{
		if (SORT_DIRECT == 1)
			threadOddEvenMergeSort<SortAscending>(col_indices, values);
		else if (SORT_DIRECT > 1)
			CubSorter(smem.sortStorage).SortBlockedToStriped(col_indices, values);

#pragma unroll
		for (int i = 0; i < NNZ_PER_THREAD; ++i)
			if (values[i] != 0)
				tempAtomicAdd(outVec + col_indices[i], values[i]);
		return;
	}

	//full resolve in shared
	if (SORTED_RESOLVE)
	{
		__syncthreads();

		// sort multiplied data according to column
		CubSorter(smem.sortStorage).Sort(col_indices, values);

		if (threadIdx.x == 0)
			smem.minCol = col_indices[0];
		else if (threadIdx.x == THREADS - 1)
			smem.maxCol = col_indices[NNZ_PER_THREAD - 1];
	}
	else if (BUFFERED_RESOLVE)
	{
		//TODO: consider value: if zero we dont care (important for last)
		//determine min and max
		uint32_t mymin = col_indices[0];
		uint32_t mymax = col_indices[0];
#pragma unroll
		for (uint32_t i = 1; i < NNZ_PER_THREAD; ++i)
		{
			mymin = min(mymin, col_indices[i]);
			mymax = max(mymax, col_indices[i]);
		}
#pragma unroll
		for (uint32_t offset = 1; offset < WARP_SIZE; offset *= 2)
		{
			mymin = min(mymin, __shfl_down(mymin, offset));
			mymax = max(mymax, __shfl_down(mymax, offset));
		}
		if (laneid() == 0)
		{
			atomicMin(&smem.minCol, mymin);
			atomicMax(&smem.maxCol, mymax);
		}
	}
	else
	{
		if (SORT_DIRECT == 1)
			threadOddEvenMergeSort<SortAscending>(col_indices, values);
		else if (SORT_DIRECT > 1)
			CubSorter(smem.sortStorage).SortBlockedToStriped(col_indices, values);
		// atomic to global out
#pragma unroll
		for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
			if (values[i] != 0)
				tempAtomicAdd(outVec + col_indices[i], values[i]);
		return;
	}

	__syncthreads();

	// switch:
	// o single column:
	//   every thread adds its up, warp reduction, adomic add
	// o number column < BUFFERSIZE
	//   allocate output buffer in shared, zero buffer
	//   every thread add its up and writes its data to shared on column switch
	//   for last decide how many pivots there are and then either do a selective reduction or direct atomicadd
	// o number column larger
	//   add own up and atomic add to global

	if (smem.minCol == smem.maxCol)
	{
		//single column
		VALUE_TYPE3 out = values[0];
#pragma unroll
		for (uint32_t i = 1; i < NNZ_PER_THREAD; ++i)
			out = out + values[i];
#pragma unroll
		for (uint32_t offset = 1; offset < WARP_SIZE; offset *= 2)
			out = out + __shfl_down(out, offset);
		if (laneid() == 0)
			tempAtomicAdd(outVec + col_indices[0], out);
		return;
	}
	else if (smem.maxCol - smem.minCol < OUTPUT_BUFFER_SIZE)
	{
		//zero output buffer
		for (int i = 0; i < OUTPUT_BUFFER_SIZE / THREADS; ++i)
			smem.outBuffer[i*THREADS + threadIdx.x] = 0;

		const int REM = OUTPUT_BUFFER_SIZE - OUTPUT_BUFFER_SIZE / THREADS * THREADS;
		if (REM > 0 && threadIdx.x < REM)
			smem.outBuffer[(OUTPUT_BUFFER_SIZE - REM) + threadIdx.x] = 0;
		__syncthreads();


		if (SORTED_RESOLVE)
		{
			sortedResolve(values, col_indices, [&](INDEX_TYPE id, VALUE_TYPE3 out) {
				smem.outBuffer[id - smem.minCol] = out;
			}, [&](INDEX_TYPE id, VALUE_TYPE3 out) {
				if (out != 0)
					tempAtomicAdd(smem.outBuffer + id - smem.minCol, out);
			}, []() {
				__syncthreads();
			});
		}
		else
		{
			// atomic to outputbuffer
#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
				tempAtomicAdd(smem.outBuffer + col_indices[i] - smem.minCol, values[i]);
		}
		__syncthreads();
		//atomic to global
		for (int i = threadIdx.x; i < smem.maxCol - smem.minCol + 1; i += THREADS)
		{
			VALUE_TYPE3 out = smem.outBuffer[i];
			if (out != 0)
				tempAtomicAdd(outVec + smem.minCol + i, out);
		}
	}
	else
	{
		if (SORTED_RESOLVE)
		{
			sortedResolve(values, col_indices, [&](INDEX_TYPE id, VALUE_TYPE3 out) {
				if (out != 0)
					tempAtomicAdd(outVec + id, out);
			}, [&](INDEX_TYPE id, VALUE_TYPE3 out) {
				if (out != 0)
					tempAtomicAdd(outVec + id, out);
			}, []() {});
		}
		else
		{
			// atomic to global out
#pragma unroll
			for (uint32_t i = 0; i < NNZ_PER_THREAD; ++i)
				tempAtomicAdd(outVec + col_indices[i], values[i]);
		}

	}
}



template<int NNZ_PER_THREAD, typename T>
struct HolaLaunchBounds
{
	static const int Threads = 256;
	static const int Bounds = 0;
};

template<int NNZ_PER_THREAD>
struct HolaLaunchBounds<NNZ_PER_THREAD, float>
{
	static const int Threads = 256;
	static const int Bounds = 2048 / Threads;
};

template<>
struct HolaLaunchBounds<8, double>
{
	static const int Threads = 256;
	static const int Bounds = 1536 / Threads;
};

template<typename T>
void hola_spmv(void* tempmem, size_t& tempmemsize, dDenseVector<T>& res, const dCSR<T>& m, const dDenseVector<T>& v, HolaMode mode, bool transpose, bool padded)
{
	const bool ZeroCheck = true;
	const bool Scale = false;

	if (transpose && v.size != m.rows)
		throw std::runtime_error("SPMV dimensions mismatch");
	if (!transpose && v.size != m.cols)
		throw std::runtime_error("SPMV dimensions mismatch");

	if (mode == HolaMode::Default)
	{
		if (transpose)
			mode = m.nnz < 300000 ? HolaMode::NonSortedFour : HolaMode::NonSortedEight;
		else
			mode = m.nnz < 300000 ? HolaMode::SortedFour : HolaMode::SortedEight;
	}

	uint32_t blockSize = (mode == HolaMode::NonSortedFour || mode == HolaMode::SortedFour) ? HolaLaunchBounds<4,T>::Threads : HolaLaunchBounds<8, T>::Threads;
	uint32_t nnzperthread = (mode == HolaMode::NonSortedFour || mode == HolaMode::SortedFour) ? 4 : 8;
	uint32_t nnzperblock = blockSize * nnzperthread;
	uint32_t requiredBlocks = divup<uint32_t>(m.nnz, nnzperblock);
	
	if (tempmem == nullptr)
	{
		tempmemsize = (requiredBlocks + 2) * 4;
		return;
	}

	size_t outsize = transpose ? m.cols : m.rows;
	if (res.size < outsize)
		res.alloc(outsize + (padded ? 1024 : 0));
	res.size = outsize;

	const uint32_t blockSize4 = HolaLaunchBounds<4, T>::Threads;
	const uint32_t blockSize8 = HolaLaunchBounds<8, T>::Threads;
	
	if (transpose)
	{

		if (nnzperthread == 4)
			DetermineBlockStartsTranspose<unsigned int, 4 * HolaLaunchBounds<4, T>::Threads, T> <<< divup<uint32_t>(std::max(m.rows + 1, m.cols), HolaLaunchBounds<4, T>::Threads), HolaLaunchBounds<4, T>::Threads >> > (m.rows, m.cols, m.row_offsets, reinterpret_cast<uint32_t*>(tempmem), res.data);
		else
			DetermineBlockStartsTranspose<unsigned int, 8 * HolaLaunchBounds<8, T>::Threads, T> <<<divup<uint32_t>(std::max(m.rows + 1, m.cols), HolaLaunchBounds<8, T>::Threads), HolaLaunchBounds<8, T>::Threads >> > (m.rows, m.cols, m.row_offsets, reinterpret_cast<uint32_t*>(tempmem), res.data);


		if (padded)
		{
			switch (mode)
			{
			case HolaMode::SortedFour:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, true, Scale, false, false, true, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::SortedEight:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, true, Scale, false, false, true, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedFour:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, true, Scale, false, false, false, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedEight:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, true, Scale, false, false, false, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			}
		}
		else
		{
			switch (mode)
			{
			case HolaMode::SortedFour:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, false, Scale, false, false, true, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::SortedEight:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, false, Scale, false, false, true, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedFour:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, false, Scale, false, false, false, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedEight:
				MultiplyAlongTranspose<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, false, Scale, false, false, false, 32, 0, 0> <<<requiredBlocks, blockSize >>> (m.nnz, m.cols, m.rows, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			}
		}
	}
	else
	{
		if(nnzperthread == 4)
			DetermineBlockStarts<unsigned int, 4 * HolaLaunchBounds<4, T>::Threads, T><<<divup<uint32_t>(m.rows + 1, HolaLaunchBounds<4, T>::Threads), HolaLaunchBounds<4, T>::Threads >>>(m.rows, m.row_offsets, reinterpret_cast<uint32_t*>(tempmem), res.data);
		else
			DetermineBlockStarts<unsigned int, 8 * HolaLaunchBounds<8, T>::Threads, T><<<divup<uint32_t>(m.rows + 1, HolaLaunchBounds<8, T>::Threads), HolaLaunchBounds<8, T>::Threads >>>(m.rows, m.row_offsets, reinterpret_cast<uint32_t*>(tempmem), res.data);
		
		if (padded)
		{
			switch (mode)
			{
			case HolaMode::SortedFour:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4,T>::Bounds, true, ZeroCheck, Scale, true, 4 * blockSize4, 0> <<<requiredBlocks, blockSize >>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::SortedEight:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, true, ZeroCheck, Scale, true, 8 * blockSize8, 0> <<<requiredBlocks, blockSize >>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedFour:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, true, ZeroCheck, Scale, false, 4 * blockSize4, 0><< <requiredBlocks, blockSize >>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedEight:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, true, ZeroCheck, Scale, false, 8 * blockSize8, 0> <<<requiredBlocks, blockSize >>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			}
		}
		else
		{
			switch (mode)
			{
			case HolaMode::SortedFour:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, false, ZeroCheck, Scale, true, 4 * blockSize4, 0><<<requiredBlocks, blockSize>>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::SortedEight:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, false, ZeroCheck, Scale, true, 8 * blockSize8, 0><<<requiredBlocks, blockSize>>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedFour:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 4 * blockSize4, blockSize4, HolaLaunchBounds<4, T>::Bounds, false, ZeroCheck, Scale, false, 4 * blockSize4, 0><<<requiredBlocks, blockSize>>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			case HolaMode::NonSortedEight:
				MultiplyAlong<T, T, T, unsigned int, unsigned int, 8 * blockSize8, blockSize8, HolaLaunchBounds<8, T>::Bounds, false, ZeroCheck, Scale, false, 8 * blockSize8, 0><<<requiredBlocks, blockSize>>>(m.nnz, m.rows, m.cols, m.data, m.col_ids, m.row_offsets, v.data, res.data, 1, reinterpret_cast<uint32_t*>(tempmem));
				break;
			}
		}
	}
}

template void hola_spmv(void* tempmem, size_t& tempmemsize, dDenseVector<float>& res, const dCSR<float>& m, const dDenseVector<float>& v, HolaMode mode, bool transpose, bool padded);
template void hola_spmv(void* tempmem, size_t& tempmemsize, dDenseVector<double>& res, const dCSR<double>& m, const dDenseVector<double>& v, HolaMode mode, bool transpose, bool padded);