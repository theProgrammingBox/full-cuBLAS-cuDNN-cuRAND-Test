#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>

/*
IMPORTANT LESSONS
1. cudnnConvolutionForward can only work with float alpha and beta when using __half
2. cublasGemmStridedBatchedEx can only work with __half alpha and beta when using __half
*/

/*
TODO
0. Add attention
1. Optimize using << and storing bytes
2. Add more convolution before attention
*/

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

	const uint32_t BATCH_SIZE = 1;

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

	const uint32_t ATTENTION_DIMENTION = 8;
	
	const float alpha = 1.0f;
	const float beta = 0.0f;
	
	__half* gpuTensorInput;
	__half* gpuTensorFilter;
	__half* gpuTensorOutput;
	__half* gpuTensorQueryWeights;
	__half* gpuTensorQueries;
	
	cudaMalloc(&gpuTensorInput, INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS * sizeof(__half));
	cudaMalloc(&gpuTensorFilter, FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS * sizeof(__half));
	cudaMalloc(&gpuTensorOutput, OUTPUT_COLS * OUTPUT_ROWS * OUTPUT_CHANNELS * sizeof(__half));
	cudaMalloc(&gpuTensorQueryWeights, ATTENTION_DIMENTION * OUTPUT_CHANNELS * sizeof(__half));
	cudaMalloc(&gpuTensorQueries, ATTENTION_DIMENTION * OUTPUT_COLS * OUTPUT_ROWS * sizeof(__half));
	
	__half* cpuTensorInput = new __half[INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS];
	__half* cpuTensorFilter = new __half[FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS];
	__half* cpuTensorOutput = new __half[OUTPUT_COLS * OUTPUT_ROWS * OUTPUT_CHANNELS];
	__half* cpuTensorQueryWeights = new __half[ATTENTION_DIMENTION * OUTPUT_CHANNELS];
	__half* cpuTensorQueries = new __half[ATTENTION_DIMENTION * OUTPUT_COLS * OUTPUT_ROWS];

	CurandGenerateUniformF16(curandGenerator, gpuTensorInput, INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS);
	CurandGenerateUniformF16(curandGenerator, gpuTensorFilter, FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS);
	CurandGenerateUniformF16(curandGenerator, gpuTensorQueryWeights, ATTENTION_DIMENTION * OUTPUT_CHANNELS);
	
	cudaMemcpy(cpuTensorInput, gpuTensorInput, INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuTensorFilter, gpuTensorFilter, FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuTensorQueryWeights, gpuTensorQueryWeights, ATTENTION_DIMENTION * OUTPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);
	
	PrintMatrixF16(cpuTensorInput, INPUT_ROWS, INPUT_COLS, "cpuTensorInput");
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("cpuTensorFilter[%d]:\n", i);
		PrintMatrixF16(cpuTensorFilter + i * FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS, "cpuTensorFilter");
	}

	cudnnTensorDescriptor_t inputTensorDescriptor;
	cudnnTensorDescriptor_t outputTensorDescriptor;
	cudnnFilterDescriptor_t kernelTensorDescriptor;
	cudnnConvolutionDescriptor_t convolutionDescriptor;

	cudnnCreateTensorDescriptor(&inputTensorDescriptor);
	cudnnCreateTensorDescriptor(&outputTensorDescriptor);
	cudnnCreateFilterDescriptor(&kernelTensorDescriptor);
	cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

	cudnnSetTensor4dDescriptor(inputTensorDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, BATCH_SIZE, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS);
	cudnnSetTensor4dDescriptor(outputTensorDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, BATCH_SIZE, OUTPUT_CHANNELS, OUTPUT_ROWS, OUTPUT_COLS);
	cudnnSetFilter4dDescriptor(kernelTensorDescriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, OUTPUT_CHANNELS, INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS);
	cudnnSetConvolution2dDescriptor(convolutionDescriptor, PADDING, PADDING, STRIDE, STRIDE, DILATION, DILATION, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF);

	int maxPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(cudnnHandle, inputTensorDescriptor, kernelTensorDescriptor, convolutionDescriptor, outputTensorDescriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, forwardPropagationAlgorithms);
	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
	delete[] forwardPropagationAlgorithms;
	printf("Forward propagation algorithm: %d\n\n", forwardPropagationAlgorithm);

	size_t workspaceBytes = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputTensorDescriptor, kernelTensorDescriptor, convolutionDescriptor, outputTensorDescriptor, forwardPropagationAlgorithm, &workspaceBytes);
	void* workspace;
	cudaMalloc(&workspace, workspaceBytes);

	cudnnConvolutionForward(cudnnHandle, &alpha, inputTensorDescriptor, gpuTensorInput, kernelTensorDescriptor, gpuTensorFilter, convolutionDescriptor, forwardPropagationAlgorithm, workspace, workspaceBytes, &beta, outputTensorDescriptor, gpuTensorOutput);

	cudaMemcpy(cpuTensorOutput, gpuTensorOutput, OUTPUT_COLS * OUTPUT_ROWS * OUTPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("cpuTensorOutput[%d]:\n", i);
		PrintMatrixF16(cpuTensorOutput + i * OUTPUT_COLS * OUTPUT_ROWS, OUTPUT_ROWS, OUTPUT_COLS, "cpuTensorOutput");
	}
	
	PrintMatrixF16(cpuTensorOutput, OUTPUT_CHANNELS, OUTPUT_COLS * OUTPUT_ROWS, "cpuTensorOutput");
	PrintMatrixF16(cpuTensorQueryWeights, OUTPUT_CHANNELS, ATTENTION_DIMENTION, "cpuTensorQueryWeights");
	
	const __half alphaf16 = __float2half(1.0f);
	const __half betaf16 = __float2half(0.0f);
	
	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
		ATTENTION_DIMENTION, OUTPUT_COLS * OUTPUT_ROWS, OUTPUT_CHANNELS,
		&alphaf16,
		gpuTensorQueryWeights, CUDA_R_16F, ATTENTION_DIMENTION, 0,
		gpuTensorOutput, CUDA_R_16F, OUTPUT_COLS * OUTPUT_ROWS, 0,
		&betaf16,
		gpuTensorQueries, CUDA_R_16F, ATTENTION_DIMENTION, 0,
		BATCH_SIZE, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);

	cudaMemcpy(cpuTensorQueries, gpuTensorQueries, ATTENTION_DIMENTION * OUTPUT_COLS * OUTPUT_ROWS * sizeof(__half), cudaMemcpyDeviceToHost);
	
	PrintMatrixF16(cpuTensorQueries, OUTPUT_COLS * OUTPUT_ROWS, ATTENTION_DIMENTION, "cpuTensorQueries");
	
	return 0;
}