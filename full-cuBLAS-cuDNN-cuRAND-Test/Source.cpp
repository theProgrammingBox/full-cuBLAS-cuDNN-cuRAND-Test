#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>

void PrintMatrixF16(__half* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", __half2float(arr[i * cols + j]));
		printf("\n");
	}
	printf("\n");
}

__global__ void CurandNormalizeF16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * range + min);
}

void CurandGenerateUniformF16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizeF16 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 0.0000152590218967f);
}

int main()
{
	printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
	printf("cuBLAS version: %d.%d.%d\n", CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH);
	printf("cuRAND version: %d.%d.%d\n", CURAND_VERSION / 1000, (CURAND_VERSION % 1000) / 100, CURAND_VERSION % 100);
	printf("\n");
	
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);
	
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

	const uint32_t INPUT_CHANNELS = 1;
	const uint32_t INPUT_ROWS = 16;
	const uint32_t INPUT_COLS = 16;
	
	const uint32_t OUTPUT_CHANNELS = 4;
	const uint32_t OUTPUT_ROWS = 4;
	const uint32_t OUTPUT_COLS = 4;

	const uint32_t FILTER_ROWS = 4;
	const uint32_t FILTER_COLS = 4;

	const uint32_t PADDING = 0;
	const uint32_t STRIDE = 4;
	const uint32_t DILATION = 1;

	__half* mat1GPU;
	__half* mat2GPU;
	__half* mat3GPU;
	__half* filterGPU;
	__half* mat4GPU;
	
	cudaMalloc(&mat1GPU, INPUT_ROWS << 1);
	cudaMalloc(&mat2GPU, INPUT_COLS << 1);
	cudaMalloc(&mat3GPU, INPUT_ROWS * INPUT_COLS << 1);
	cudaMalloc(&filterGPU, FILTER_ROWS * FILTER_COLS * INPUT_CHANNELS * OUTPUT_CHANNELS << 1);
	cudaMalloc(&mat4GPU, OUTPUT_ROWS * OUTPUT_COLS * OUTPUT_CHANNELS << 1);

	__half* mat1CPU = (__half*)malloc(INPUT_ROWS << 1);
	__half* mat2CPU = (__half*)malloc(INPUT_COLS << 1);
	__half* mat3CPU = (__half*)malloc(INPUT_ROWS * INPUT_COLS << 1);
	__half* filterCPU = (__half*)malloc(FILTER_ROWS * FILTER_COLS * INPUT_CHANNELS * OUTPUT_CHANNELS << 1);
	__half* mat4CPU = (__half*)malloc(OUTPUT_ROWS * OUTPUT_COLS * OUTPUT_CHANNELS << 1);

	CurandGenerateUniformF16(curandGenerator, mat1GPU, INPUT_ROWS);
	CurandGenerateUniformF16(curandGenerator, mat2GPU, INPUT_COLS);
	CurandGenerateUniformF16(curandGenerator, filterGPU, FILTER_ROWS * FILTER_COLS * INPUT_CHANNELS * OUTPUT_CHANNELS);

	cudaMemcpy(mat1CPU, mat1GPU, INPUT_ROWS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(mat2CPU, mat2GPU, INPUT_COLS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(filterCPU, filterGPU, FILTER_ROWS * FILTER_COLS * INPUT_CHANNELS * OUTPUT_CHANNELS << 1, cudaMemcpyDeviceToHost);

	PrintMatrixF16(mat1CPU, INPUT_ROWS, 1, "mat1CPU");
	PrintMatrixF16(mat2CPU, 1, INPUT_COLS, "mat2CPU");
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("filterCPU[%d]:\n", i);
		PrintMatrixF16(filterCPU + i * FILTER_ROWS * FILTER_COLS * INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS, "");
	}
	
	const __half alphaF16 = __float2half(1.0f);
	const __half betaF16 = __float2half(0.0f);
	
	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		INPUT_COLS, INPUT_ROWS, 1,
		&alphaF16,
		mat2GPU, CUDA_R_16F, INPUT_COLS, INPUT_COLS,
		mat1GPU, CUDA_R_16F, 1, INPUT_ROWS,
		&betaF16,
		mat3GPU, CUDA_R_16F, INPUT_COLS, INPUT_ROWS * INPUT_COLS,
		1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);

	cudaMemcpy(mat3CPU, mat3GPU, INPUT_ROWS * INPUT_COLS << 1, cudaMemcpyDeviceToHost);

	PrintMatrixF16(mat3CPU, INPUT_ROWS, INPUT_COLS, "mat3CPU");

	cudnnTensorDescriptor_t inputDescriptor;
	cudnnCreateTensorDescriptor(&inputDescriptor);
	cudnnSetTensor4dDescriptor
	(
		inputDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
		1, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS
	);

	cudnnTensorDescriptor_t outputDescriptor;
	cudnnCreateTensorDescriptor(&outputDescriptor);
	cudnnSetTensor4dDescriptor
	(
		outputDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
		1, OUTPUT_CHANNELS, OUTPUT_ROWS, OUTPUT_COLS
	);

	cudnnFilterDescriptor_t filterDescriptor;
	cudnnCreateFilterDescriptor(&filterDescriptor);
	cudnnSetFilter4dDescriptor
	(
		filterDescriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
		OUTPUT_CHANNELS, INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS
	);

	cudnnConvolutionDescriptor_t convolutionDescriptor;
	cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
	cudnnSetConvolution2dDescriptor
	(
		convolutionDescriptor,
		PADDING, PADDING,
		STRIDE, STRIDE,
		DILATION, DILATION,
		CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF
	);

	int maxPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
	cudnnFindConvolutionForwardAlgorithm
	(
		cudnnHandle,
		inputDescriptor, filterDescriptor, convolutionDescriptor, outputDescriptor,
		maxPropagationAlgorithms, &maxPropagationAlgorithms, forwardPropagationAlgorithms
	);
	cudnnConvolutionFwdAlgo_t forwardConvolutionAlgorithm = forwardPropagationAlgorithms[0].algo;
	delete[] forwardPropagationAlgorithms;

	size_t forwardWorkspaceSize;
	cudnnGetConvolutionForwardWorkspaceSize
	(
		cudnnHandle,
		inputDescriptor, filterDescriptor, convolutionDescriptor, outputDescriptor,
		forwardConvolutionAlgorithm, &forwardWorkspaceSize
	);
	void* forwardWorkspace;
	cudaMalloc(&forwardWorkspace, forwardWorkspaceSize);

	cudnnConvolutionForward
	(
		cudnnHandle,
		&alphaF16,
		inputDescriptor, mat3GPU,
		filterDescriptor, filterGPU,
		convolutionDescriptor, forwardConvolutionAlgorithm,
		forwardWorkspace, forwardWorkspaceSize,
		&betaF16,
		outputDescriptor, mat4GPU
	);

	cudaMemcpy(mat4CPU, mat4GPU, OUTPUT_ROWS * OUTPUT_COLS * OUTPUT_CHANNELS << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("mat4CPU[%d]:\n", i);
		PrintMatrixF16(mat4CPU + i * OUTPUT_ROWS * OUTPUT_COLS, OUTPUT_ROWS, OUTPUT_COLS, "");
	}
	
	return 0;
}