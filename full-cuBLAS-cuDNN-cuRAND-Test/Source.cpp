#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

int main()
{
	std::cout << "cuDNN version: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
	std::cout << "cuBLAS version: " << CUBLAS_VER_MAJOR << "." << CUBLAS_VER_MINOR << "." << CUBLAS_VER_PATCH << std::endl;
	std::cout << "cuRAND version: " << CURAND_VERSION << std::endl;
	return 0;
}