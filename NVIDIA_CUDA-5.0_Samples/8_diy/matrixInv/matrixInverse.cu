////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Inverse Matrix by CUBLAS library
8.3. cublas<t>getrfBatched(): http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched  in cuda5.0
8.4. cublas<t>getriBatched(): http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched  in cuda5.5

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes CUDA BLAS
#include <cublas_v2.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

#define BENCH_MATRIX_EXP			7 //2~10
#define BENCH_MATRIX_ROWS           (1<<BENCH_MATRIX_EXP)
#define CUBLAS_TEST_COUNT			(10) // 10~1000

#define T_ELEM	float
//#define T_ELEM	double


__inline__ __device__ __host__  float cuGet(double x)
{
    return float(x);
}

void fillupMatrixDebug(T_ELEM *A , int size )
{
    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < size; i++)
        {
            A[i + size*j ] = cuGet(i + j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float *g_idata, float *g_odata)
{
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;
	cudaError_t err1;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

	// size
	int matrixRows = BENCH_MATRIX_ROWS;
    int matrixSize = matrixRows * matrixRows;

	// data
	T_ELEM *A = NULL;
    T_ELEM **devPtrA = 0;
    T_ELEM **devPtrA_dev = NULL;

	T_ELEM **devPtrC = 0;
    T_ELEM **devPtrC_dev = NULL;

	// matrix A, input matrix
    devPtrA =(T_ELEM **)malloc( CUBLAS_TEST_COUNT * sizeof(T_ELEM));
	for (int i = 0; i < CUBLAS_TEST_COUNT ; i++)
    {
        err1 = cudaMalloc((void **)&devPtrA[i], matrixSize * sizeof(T_ELEM));
	}

    err1 = cudaMalloc((void **)&devPtrA_dev, CUBLAS_TEST_COUNT * sizeof(T_ELEM));
    err1 = cudaMemcpy(devPtrA_dev, devPtrA, CUBLAS_TEST_COUNT * sizeof(*devPtrA), cudaMemcpyHostToDevice);

    A  = (T_ELEM *)malloc(matrixSize * sizeof(T_ELEM));

	memset(A, 0xFF, matrixSize * sizeof(A[0]));
	fillupMatrixDebug( A, matrixRows );

	for (int i = 0; i < CUBLAS_TEST_COUNT ; i++)
    {
		cublasSetMatrix( matrixRows, matrixRows, sizeof(A[0]), A, matrixRows, devPtrA[i], matrixRows);
	}

	// matrix C, output inverse matrix of A
	devPtrC =(T_ELEM **)malloc( CUBLAS_TEST_COUNT * sizeof(T_ELEM));
	for (int i = 0; i < CUBLAS_TEST_COUNT ; i++)
    {
        err1 = cudaMalloc((void **)&devPtrC[i], matrixSize * sizeof(T_ELEM));
	}

    err1 = cudaMalloc((void **)&devPtrC_dev, CUBLAS_TEST_COUNT * sizeof(T_ELEM));
    err1 = cudaMemcpy(devPtrC_dev, devPtrC, CUBLAS_TEST_COUNT * sizeof(*devPtrC), cudaMemcpyHostToDevice);

	// temp data
	int *pivotArray = new int[ matrixRows*CUBLAS_TEST_COUNT ];
	int *infoArray = new int[ CUBLAS_TEST_COUNT ];

	// blas config
    cublasHandle_t handle;
	cublasCreate(&handle);

	// timer
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    cublasSetStream(handle, 0);
	
	cublasStatus_t status ;

	// LU factorization
	status = cublasSgetrfBatched(handle, 
		matrixRows, 
		devPtrA_dev, 
		matrixRows,
		pivotArray,
		infoArray,
		CUBLAS_TEST_COUNT);

	if (status != CUBLAS_STATUS_SUCCESS)
    {
          cudaError_t cuError = cudaGetLastError();
          fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status, cuError,cudaGetErrorString(cuError));
          return ;
    }

	// inversion of matrices A, output result to matrices C
	status = cublasSgetriBatched(handle, 
		matrixRows, 
		devPtrA_dev, 
		matrixRows,
		pivotArray,
		devPtrC_dev,
		matrixRows,
		infoArray,
		CUBLAS_TEST_COUNT);

	if (status != CUBLAS_STATUS_SUCCESS)
    {
          cudaError_t cuError = cudaGetLastError();
          fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status, cuError,cudaGetErrorString(cuError));
          return ;
    }


    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);



    // cleanup memory
	if (A) free (A); 

	for(int i = 0; i < CUBLAS_TEST_COUNT; ++i) {       
            if(devPtrA[i]) cudaFree(devPtrA[i]);
            if(devPtrC[i]) cudaFree(devPtrC[i]);
        }  

	if (devPtrA) free(devPtrA);           
	if (devPtrC) free(devPtrC); 

	if (devPtrA_dev) cudaFree(devPtrA_dev);
	if (devPtrC_dev) cudaFree(devPtrC_dev); 

    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
