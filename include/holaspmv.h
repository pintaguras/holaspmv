#pragma once

#include "dCSR.h"
#include "dVector.h"

enum struct HolaMode
{
	Default,
	SortedFour,
	SortedEight,
	NonSortedFour,
	NonSortedEight
};

template<typename T>
void hola_spmv(void* tempmem, size_t& tempmemsize, dDenseVector<T>& res, const dCSR<T>& m, const dDenseVector<T>& v, HolaMode mode = HolaMode::Default, bool transpose = false, bool padded = false);