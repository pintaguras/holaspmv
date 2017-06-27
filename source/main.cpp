#include "../include/COO.h"
#include "../include/CSR.h"
#include "../include/Vector.h"
#include "../include/dCSR.h"
#include "../include/dVector.h"
#include "../include/naivespmv.h"
#include "../include/holaspmv.h"

#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <algorithm>
#include <tuple>

#include <cuda_runtime.h>



//test settings
using DataType = double;
HolaMode holaMode = HolaMode::Default;
bool transpose = false;
unsigned int padding = 0; // 0; // 1024;




template<typename T>
std::string typeext();
template<> std::string typeext<float>();
template<> std::string typeext<double>();

template<typename T>
std::tuple<T, T, T, T, bool>  compare(const DenseVector<T>& a, const DenseVector<T>& b, T threshold = 0.05f);

int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << argv[0] << " mtx_file [device_id]\n";
		return -1;
	}

	int device = 0;
	if (argc >= 3)
		device = std::stoi(argv[2]);
	
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	CSR<DataType> csr_mat;

	//try load csr file
	std::string csr_name = std::string(argv[1]) + "_" + typeext<DataType>() + ".csr";
	try
	{
		std::cout << "trying to load csr file \"" << csr_name << "\"\n";
		csr_mat = loadCSR<DataType>(csr_name.c_str());
	}
	catch (std::exception& ex)
	{
		std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
		try
		{
			std::cout << "trying to load mtx file \"" << argv[1] << "\"\n";
			COO<DataType> coo_mat = loadMTX<DataType>(argv[1]);
			convert(csr_mat, coo_mat);
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			return -1;
		}
		try
		{
			storeCSR(csr_mat, csr_name.c_str());
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}

	try
	{ 
		//std::ofstream resfile("res.txt", std::fstream::app);
		//resfile << "\n" << argv[1] << ",";
		//resfile.flush();

		//compute ground truth
		std::cout << "creating input data\n";
		DenseVector<DataType> res, input;
		input.alloc(csr_mat.cols);

		std::mt19937 gen(123456789);
		std::uniform_real_distribution<> dis(0, 1.0);
		std::generate(&input.data[0], &input.data[input.size], [&]() {return dis(gen); });

		std::cout << "computing ground truth SPMV on the CPU\n";
		spmv(res, csr_mat, input, transpose);

		std::cout << "setting up GPU memory\n";

		dCSR<DataType> dcsr_mat;
		dDenseVector<DataType> dres, dinput;
		DenseVector<DataType> hres;
		void* dholatemp;

		convert(dcsr_mat, csr_mat, padding);
		convert(dinput, input, padding);

		std::cout << "computing naive SPMV on the GPU\n";
		naive_spmv(dres, dcsr_mat, dinput, transpose);
		if (cudaDeviceSynchronize() != cudaSuccess)
			throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
		convert(hres, dres);

		auto d = compare(res, hres);
		std::cout << "Naive max diff: " << std::get<0>(d) << " (" << std::get<2>(d) << ") avg diff: " << std::get<1>(d) << " (" << std::get<3>(d) << ")\n";
		//resfile << std::get<0>(d) << "," << std::get<1>(d) << "," << std::get<2>(d) << "," << std::get<3>(d) << ","; resfile.flush();

		size_t holatemp_req;
		hola_spmv(nullptr, holatemp_req, dres, dcsr_mat, dinput, holaMode, transpose, padding >= 512 ? true : false);
		cudaMalloc(&dholatemp, holatemp_req);
		if (cudaDeviceSynchronize() != cudaSuccess)
			throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));

		std::cout << "computing Hola SPMV on the GPU\n";
		hola_spmv(dholatemp, holatemp_req, dres, dcsr_mat, dinput, holaMode, transpose, padding >= 512 ? true : false);
		if (cudaDeviceSynchronize() != cudaSuccess)
			throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
		convert(hres, dres);

		auto d_hola = compare(res, hres);
		std::cout << "Hola max diff: " << std::get<0>(d_hola) << " (" << std::get<2>(d_hola) << ") avg diff: " << std::get<1>(d_hola) << " (" << std::get<3>(d_hola) << ")\n";
		//resfile << std::get<0>(d_hola) << "," << std::get<1>(d_hola) << "," << std::get<2>(d_hola) << "," << std::get<3>(d_hola) << "," << (std::get<4>(d_hola)?"XXXXXX,":",");resfile.flush();

		cudaFree(dholatemp);

	}
	catch (std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
		return -1;
	}

	return 0;
}

template<>
std::string typeext<float>()
{
	return "f";
}
template<>
std::string typeext<double>()
{
	return "d";
}

template<typename T>
std::tuple<T,T,T,T,bool> compare(const DenseVector<T>& a, const DenseVector<T>& b, T threshold)
{
	if (a.size != b.size)
		throw std::runtime_error("Comparison Vectors are of different size");
	T maxdiff = 0, maxampldiff = 0, avgdiff = 0, avgampldiff = 0;
	bool large = false;
	for (size_t i = 0; i < a.size; ++i)
	{
		T d = std::abs(a.data[i] - b.data[i]);
		T ampl = std::min(std::abs(a.data[i]), std::abs(b.data[i]));
		if (d > ampl*threshold)
		{
			std::cout << "large difference at " << i << ": " << a.data[i] << " vs " << b.data[i] << "\n";
			large = true;
		}
		avgdiff += d;
		avgampldiff += d / ampl;
		maxdiff = std::max(maxdiff, d);
		maxampldiff = std::max(maxampldiff, d/ampl);
	}
	avgdiff /= a.size;
	avgampldiff /= a.size;
	return std::make_tuple(maxdiff, avgdiff, maxampldiff, avgampldiff, large);
}
