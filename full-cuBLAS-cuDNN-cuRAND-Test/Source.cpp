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
-1. Add positional bias
0. Add attention scores
1. add attention results
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

__global__ void GpuReluF16(__half* input, __half* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint16_t*)(input + index) >> 15)
		output[index] = 0x0000;
}

void ReluF16(__half* input, __half* output, uint32_t size)
{
	cudaMemcpy(output, input, size << 1, cudaMemcpyDeviceToDevice);
	GpuReluF16 << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
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
	
	const float alpha = 1.0f;
	const float beta = 0.0f;

	const __half alphaf16 = __float2half(1.0f);
	const __half betaf16 = __float2half(0.0f);

	const uint32_t BATCH_SIZE = 1;

	const uint32_t INPUT_CHANNELS = 1;
	const uint32_t INPUT_ROWS = 8;
	const uint32_t INPUT_COLS = 8;
	
	const uint32_t OUTPUT_CHANNELS = 4;
	const uint32_t OUTPUT_ROWS = 2;
	const uint32_t OUTPUT_COLS = 2;

	const uint32_t FILTER_ROWS = 4;
	const uint32_t FILTER_COLS = 4;

	const uint32_t PADDING = 0;
	const uint32_t STRIDE = 4;
	const uint32_t DILATION = 1;

	const uint32_t TENSOR_QUERY_DIMENTION = 8;
	const uint32_t TENSOR_VALUE_DIMENTION = 16;

	const uint32_t TENSOR_INPUT_AREA = INPUT_COLS * INPUT_ROWS;
	const uint32_t TENSOR_FILTER_AREA = FILTER_COLS * FILTER_ROWS;
	const uint32_t TENSOR_OUTPUT_AREA = OUTPUT_COLS * OUTPUT_ROWS;

	const uint32_t TENSOR_INPUT_SIZE = TENSOR_INPUT_AREA * INPUT_CHANNELS;
	const uint32_t TENSOR_FILTER_SIZE = TENSOR_FILTER_AREA * INPUT_CHANNELS * OUTPUT_CHANNELS;
	const uint32_t TENSOR_OUTPUT_SIZE = TENSOR_OUTPUT_AREA * OUTPUT_CHANNELS;
	
	const uint32_t TENSOR_QUERY_WEIGHTS_SIZE = TENSOR_QUERY_DIMENTION * OUTPUT_CHANNELS;
	const uint32_t TENSOR_VALUE_WEIGHTS_SIZE = TENSOR_VALUE_DIMENTION * OUTPUT_CHANNELS;
	
	const uint32_t TENSOR_QUERIES_SIZE = TENSOR_QUERY_DIMENTION * TENSOR_OUTPUT_AREA;
	const uint32_t TENSOR_VALUES_SIZE = TENSOR_VALUE_DIMENTION * TENSOR_OUTPUT_AREA;
	
	const uint32_t TENSOR_SCORE_SIZE = TENSOR_OUTPUT_AREA * TENSOR_OUTPUT_AREA;

	__half* gpuTensorInput;
	__half* gpuTensorFilter;
	__half* gpuTensorOutput;
	__half* gpuTensorReluOutput;
	__half* gpuTensorOutputBias;
	
	__half* gpuTensorQueryWeights;
	__half* gpuTensorKeyWeights;
	__half* gpuTensorValueWeights;
	
	__half* gpuTensorQueries;
	__half* gpuTensorKeys;
	__half* gpuTensorValues;

	__half* gpuTensorScore;
	
	cudaMalloc(&gpuTensorInput, TENSOR_INPUT_SIZE * BATCH_SIZE << 1);
	cudaMalloc(&gpuTensorFilter, TENSOR_FILTER_SIZE << 1);
	cudaMalloc(&gpuTensorOutput, TENSOR_OUTPUT_SIZE * BATCH_SIZE << 1);
	cudaMalloc(&gpuTensorReluOutput, TENSOR_OUTPUT_SIZE * BATCH_SIZE << 1);
	cudaMalloc(&gpuTensorOutputBias, TENSOR_OUTPUT_SIZE << 1);
	
	cudaMalloc(&gpuTensorQueryWeights, TENSOR_QUERY_WEIGHTS_SIZE << 1);
	cudaMalloc(&gpuTensorKeyWeights, TENSOR_QUERY_WEIGHTS_SIZE << 1);
	cudaMalloc(&gpuTensorValueWeights, TENSOR_VALUE_WEIGHTS_SIZE << 1);
	
	cudaMalloc(&gpuTensorQueries, TENSOR_QUERIES_SIZE * BATCH_SIZE << 1);
	cudaMalloc(&gpuTensorKeys, TENSOR_QUERIES_SIZE * BATCH_SIZE << 1);
	cudaMalloc(&gpuTensorValues, TENSOR_VALUES_SIZE * BATCH_SIZE << 1);

	cudaMalloc(&gpuTensorScore, TENSOR_SCORE_SIZE * BATCH_SIZE << 1);
	
	__half* cpuTensorInput = new __half[TENSOR_INPUT_SIZE * BATCH_SIZE];
	__half* cpuTensorFilter = new __half[TENSOR_FILTER_SIZE];
	__half* cpuTensorOutput = new __half[TENSOR_OUTPUT_SIZE * BATCH_SIZE];
	__half* cpuTensorReluOutput = new __half[TENSOR_OUTPUT_SIZE * BATCH_SIZE];
	__half* cpuTensorOutputBias = new __half[TENSOR_OUTPUT_SIZE];
	
	__half* cpuTensorQueryWeights = new __half[TENSOR_QUERY_WEIGHTS_SIZE];
	__half* cpuTensorKeyWeights = new __half[TENSOR_QUERY_WEIGHTS_SIZE];
	__half* cpuTensorValueWeights = new __half[TENSOR_VALUE_WEIGHTS_SIZE];
	
	__half* cpuTensorQueries = new __half[TENSOR_QUERIES_SIZE * BATCH_SIZE];
	__half* cpuTensorKeys = new __half[TENSOR_QUERIES_SIZE * BATCH_SIZE];
	__half* cpuTensorValues = new __half[TENSOR_VALUES_SIZE * BATCH_SIZE];

	__half* cpuTensorScore = new __half[TENSOR_SCORE_SIZE * BATCH_SIZE];

	CurandGenerateUniformF16(curandGenerator, gpuTensorInput, TENSOR_INPUT_SIZE * BATCH_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuTensorFilter, TENSOR_FILTER_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuTensorOutputBias, TENSOR_OUTPUT_SIZE);
	
	CurandGenerateUniformF16(curandGenerator, gpuTensorQueryWeights, TENSOR_QUERY_WEIGHTS_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuTensorKeyWeights, TENSOR_QUERY_WEIGHTS_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuTensorValueWeights, TENSOR_VALUE_WEIGHTS_SIZE);
	
	cudaMemcpy(cpuTensorInput, gpuTensorInput, TENSOR_INPUT_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuTensorFilter, gpuTensorFilter, TENSOR_FILTER_SIZE << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuTensorOutputBias, gpuTensorOutputBias, TENSOR_OUTPUT_SIZE << 1, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(cpuTensorQueryWeights, gpuTensorQueryWeights, TENSOR_QUERY_WEIGHTS_SIZE << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuTensorKeyWeights, gpuTensorKeyWeights, TENSOR_QUERY_WEIGHTS_SIZE << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuTensorValueWeights, gpuTensorValueWeights, TENSOR_VALUE_WEIGHTS_SIZE << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		printf("cpuTensorInput[%d]:\n", i);
		PrintMatrixF16(cpuTensorInput + i * TENSOR_INPUT_SIZE, INPUT_ROWS, INPUT_COLS, "cpuTensorInput");
	}
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		for (uint32_t j = 0; j < INPUT_CHANNELS; j++)
		{
			printf("cpuTensorFilter[%d][%d]:\n", i, j);
			PrintMatrixF16(cpuTensorFilter + i * TENSOR_FILTER_AREA * INPUT_CHANNELS + j * TENSOR_FILTER_AREA, FILTER_ROWS, FILTER_COLS, "cpuTensorFilter");
		}
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

	cudaMemcpy(cpuTensorOutput, gpuTensorOutput, TENSOR_OUTPUT_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		for (uint32_t j = 0; j < OUTPUT_CHANNELS; j++)
		{
			printf("cpuTensorOutput[%d][%d]:\n", i, j);
			PrintMatrixF16(cpuTensorOutput + i * TENSOR_OUTPUT_SIZE + j * TENSOR_OUTPUT_AREA, OUTPUT_ROWS, OUTPUT_COLS, "cpuTensorOutput");
		}
	}

	ReluF16(gpuTensorOutput, gpuTensorReluOutput, TENSOR_OUTPUT_SIZE * BATCH_SIZE);

	cudaMemcpy(cpuTensorReluOutput, gpuTensorReluOutput, TENSOR_OUTPUT_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		for (uint32_t j = 0; j < OUTPUT_CHANNELS; j++)
		{
			printf("cpuTensorReluOutput[%d][%d]:\n", i, j);
			PrintMatrixF16(cpuTensorReluOutput + i * TENSOR_OUTPUT_SIZE + j * TENSOR_OUTPUT_AREA, OUTPUT_ROWS, OUTPUT_COLS, "cpuTensorReluOutput");
		}
	}
	
	for (uint32_t i = 0; i < OUTPUT_CHANNELS; i++)
	{
		printf("cpuTensorOutputBias[%d]:\n", i);
		PrintMatrixF16(cpuTensorOutputBias + i * TENSOR_OUTPUT_AREA, OUTPUT_ROWS, OUTPUT_COLS, "cpuTensorOutputBias");
	}

	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		cublasAxpyEx
		(
			cublasHandle, TENSOR_OUTPUT_SIZE,
			&alpha, CUDA_R_32F,
			gpuTensorOutputBias, CUDA_R_16F, 1,
			gpuTensorReluOutput + i * TENSOR_OUTPUT_SIZE, CUDA_R_16F, 1,
			CUDA_R_32F
		);
	}

	cudaMemcpy(cpuTensorReluOutput, gpuTensorReluOutput, TENSOR_OUTPUT_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);

	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		for (uint32_t j = 0; j < OUTPUT_CHANNELS; j++)
		{
			printf("cpuTensorReluOutput[%d][%d]:\n", i, j);
			PrintMatrixF16(cpuTensorReluOutput + i * TENSOR_OUTPUT_SIZE + j * TENSOR_OUTPUT_AREA, OUTPUT_ROWS, OUTPUT_COLS, "cpuTensorReluOutput");
		}
	}
	
	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
		TENSOR_QUERY_DIMENTION, TENSOR_OUTPUT_AREA, OUTPUT_CHANNELS,
		&alphaf16,
		gpuTensorQueryWeights, CUDA_R_16F, TENSOR_QUERY_DIMENTION, TENSOR_QUERY_WEIGHTS_SIZE,
		gpuTensorReluOutput, CUDA_R_16F, TENSOR_OUTPUT_AREA, TENSOR_OUTPUT_SIZE,
		&betaf16,
		gpuTensorQueries, CUDA_R_16F, TENSOR_QUERY_DIMENTION, TENSOR_QUERIES_SIZE,
		BATCH_SIZE, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);

	cudaMemcpy(cpuTensorQueries, gpuTensorQueries, TENSOR_QUERIES_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		printf("cpuTensorQueries[%d]:\n", i);
		PrintMatrixF16(cpuTensorQueries + i * TENSOR_QUERIES_SIZE, TENSOR_OUTPUT_AREA, TENSOR_QUERY_DIMENTION, "cpuTensorQueries");
	}

	PrintMatrixF16(cpuTensorKeyWeights, OUTPUT_CHANNELS, TENSOR_QUERY_DIMENTION, "cpuTensorKeyWeights");

	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
		TENSOR_QUERY_DIMENTION, TENSOR_OUTPUT_AREA, OUTPUT_CHANNELS,
		&alphaf16,
		gpuTensorKeyWeights, CUDA_R_16F, TENSOR_QUERY_DIMENTION, TENSOR_QUERY_WEIGHTS_SIZE,
		gpuTensorReluOutput, CUDA_R_16F, TENSOR_OUTPUT_AREA, TENSOR_OUTPUT_SIZE,
		&betaf16,
		gpuTensorKeys, CUDA_R_16F, TENSOR_QUERY_DIMENTION, TENSOR_QUERIES_SIZE,
		BATCH_SIZE, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);

	cudaMemcpy(cpuTensorKeys, gpuTensorKeys, TENSOR_QUERIES_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		printf("cpuTensorKeys[%d]:\n", i);
		PrintMatrixF16(cpuTensorKeys + i * TENSOR_QUERIES_SIZE, TENSOR_OUTPUT_AREA, TENSOR_QUERY_DIMENTION, "cpuTensorKeys");
	}

	PrintMatrixF16(cpuTensorValueWeights, OUTPUT_CHANNELS, TENSOR_VALUE_DIMENTION, "cpuTensorValueWeights");
	
	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
		TENSOR_VALUE_DIMENTION, TENSOR_OUTPUT_AREA, OUTPUT_CHANNELS,
		&alphaf16,
		gpuTensorValueWeights, CUDA_R_16F, TENSOR_VALUE_DIMENTION, TENSOR_VALUE_WEIGHTS_SIZE,
		gpuTensorReluOutput, CUDA_R_16F, TENSOR_OUTPUT_AREA, TENSOR_OUTPUT_SIZE,
		&betaf16,
		gpuTensorValues, CUDA_R_16F, TENSOR_VALUE_DIMENTION, TENSOR_VALUES_SIZE,
		BATCH_SIZE, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);

	cudaMemcpy(cpuTensorValues, gpuTensorValues, TENSOR_VALUES_SIZE * BATCH_SIZE << 1, cudaMemcpyDeviceToHost);
	
	for (uint32_t i = 0; i < BATCH_SIZE; i++)
	{
		printf("cpuTensorValues[%d]:\n", i);
		PrintMatrixF16(cpuTensorValues + i * TENSOR_VALUES_SIZE, TENSOR_OUTPUT_AREA, TENSOR_VALUE_DIMENTION, "cpuTensorValues");
	}
	
	/*cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		TENSOR_VALUE_DIMENTION, TENSOR_OUTPUT_AREA, OUTPUT_CHANNELS,
		&alphaf16,
		gpuTensorValueWeights, CUDA_R_16F, TENSOR_VALUE_DIMENTION, TENSOR_VALUE_WEIGHTS_SIZE,
		gpuTensorReluOutput, CUDA_R_16F, TENSOR_OUTPUT_AREA, TENSOR_OUTPUT_SIZE,
		&betaf16,
		gpuTensorValues, CUDA_R_16F, TENSOR_VALUE_DIMENTION, TENSOR_VALUES_SIZE,
		BATCH_SIZE, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
	);*/
	
	return 0;
}