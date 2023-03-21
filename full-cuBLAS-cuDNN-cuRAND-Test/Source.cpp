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

	__half* gpuVec1;
	__half* gpuVec2;
	__half* gpuInput;
	__half* gpuFilter;
	__half* gpuOutput;
	
	cudaMalloc(&gpuVec1, INPUT_ROWS * sizeof(__half));
	cudaMalloc(&gpuVec2, INPUT_COLS * sizeof(__half));
	cudaMalloc(&gpuInput, INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS * sizeof(__half));
	cudaMalloc(&gpuFilter, FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS * sizeof(__half));
	cudaMalloc(&gpuOutput, OUTPUT_COLS * OUTPUT_ROWS * OUTPUT_CHANNELS * sizeof(__half));

	__half* cpuVec1 = new __half[INPUT_ROWS];
	__half* cpuVec2 = new __half[INPUT_COLS];
	__half* cpuInput = new __half[INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS];
	__half* cpuFilter = new __half[FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS];
	__half* cpuOutput = new __half[OUTPUT_COLS * OUTPUT_ROWS * OUTPUT_CHANNELS];

	CurandGenerateUniformF16(curandGenerator, gpuVec1, INPUT_ROWS);
	CurandGenerateUniformF16(curandGenerator, gpuVec2, INPUT_COLS);
	CurandGenerateUniformF16(curandGenerator, gpuFilter, FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS);

	cudaMemcpy(cpuVec1, gpuVec1, INPUT_ROWS * sizeof(__half), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuVec2, gpuVec2, INPUT_COLS * sizeof(__half), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuFilter, gpuFilter, FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);

	PrintMatrixF16(cpuVec1, INPUT_ROWS, 1, "cpuVec1");
	PrintMatrixF16(cpuVec2, 1, INPUT_COLS, "cpuVec2");
	
	const __half alphaF16 = __float2half(1.0f);
	const __half betaF16 = __float2half(0.0f);
	
	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		INPUT_COLS, INPUT_ROWS, 1,
		&alphaF16,
		gpuVec2, CUDA_R_16F, INPUT_COLS, INPUT_COLS,
		gpuVec1, CUDA_R_16F, 1, INPUT_ROWS,
		&betaF16,
		gpuInput, CUDA_R_16F, INPUT_COLS, INPUT_ROWS * INPUT_COLS,
		1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);

	cudaMemcpy(cpuInput, gpuInput, INPUT_COLS * INPUT_ROWS * INPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);

	PrintMatrixF16(cpuInput, INPUT_ROWS, INPUT_COLS, "cpuInput");
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("cpuFilter[%d]:\n", i);
		PrintMatrixF16(cpuFilter + i * FILTER_COLS * FILTER_ROWS * INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS, "cpuFilter");
	}

	cudnnTensorDescriptor_t input_descriptor;
	cudnnTensorDescriptor_t output_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor;

	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnCreateFilterDescriptor(&kernel_descriptor);
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);

	cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, BATCH_SIZE, INPUT_CHANNELS, INPUT_ROWS, INPUT_COLS);
	cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, BATCH_SIZE, OUTPUT_CHANNELS, OUTPUT_ROWS, OUTPUT_COLS);
	cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, OUTPUT_CHANNELS, INPUT_CHANNELS, FILTER_ROWS, FILTER_COLS);
	cudnnSetConvolution2dDescriptor(convolution_descriptor, PADDING, PADDING, STRIDE, STRIDE, DILATION, DILATION, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF);

	cudnnConvolutionFwdAlgo_t forwardPropagationAlgorithm;
	int maxPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxPropagationAlgorithms);
	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxPropagationAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, maxPropagationAlgorithms, &maxPropagationAlgorithms, forwardPropagationAlgorithms);
	forwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
	delete[] forwardPropagationAlgorithms;
	printf("Forward propagation algorithm: %d\n\n", forwardPropagationAlgorithm);

	size_t workspaceBytes = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, forwardPropagationAlgorithm, &workspaceBytes);
	void* workspace;
	cudaMalloc(&workspace, workspaceBytes);

	const float alpha = 1.0f;
	const float beta = 0.0f;
	cudnnConvolutionForward(cudnnHandle, &alpha, input_descriptor, gpuInput, kernel_descriptor, gpuFilter, convolution_descriptor, forwardPropagationAlgorithm, workspace, workspaceBytes, &beta, output_descriptor, gpuOutput);

	cudaMemcpy(cpuOutput, gpuOutput, OUTPUT_COLS * OUTPUT_ROWS * OUTPUT_CHANNELS * sizeof(__half), cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("cpuOutput[%d]:\n", i);
		PrintMatrixF16(cpuOutput + i * OUTPUT_COLS * OUTPUT_ROWS, OUTPUT_ROWS, OUTPUT_COLS, "cpuOutput");
	}
	
	return 0;
}