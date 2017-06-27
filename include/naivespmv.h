#pragma once

#include "dCSR.h"
#include "dVector.h"


template<typename T>
void naive_spmv(dDenseVector<T>& res, const dCSR<T>& m, const dDenseVector<T>& v, bool transpose = false);