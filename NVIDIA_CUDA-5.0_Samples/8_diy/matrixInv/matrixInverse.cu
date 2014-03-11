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
8.3. LU·Ö½â cublas<t>getrfBatched(): http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched  in cuda5.0
8.4. ÇóÄæ   cublas<t>getriBatched(): http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched  in cuda5.5

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
#define CUBLAS_TEST_COUNT			(1) // 10~1000

#define T_ELEM	float
//#define T_ELEM	double

#define SWITCH_CHAR             '-'

struct Options
{
    int sizeRow;	   // size of Row of matrix
    int sizeBatch;     // size of Batch
};

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
            A[i + size*j ] = rand()%(size*size);
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

static int processArgs(int argc, char *argv[], struct Options *opts)
{
    int error = 0;
    int oldError;
    memset(opts, 0, sizeof(*opts));

    opts->sizeRow = BENCH_MATRIX_ROWS;
    opts->sizeBatch = CUBLAS_TEST_COUNT;

    while (argc)
    {
        oldError = error;

        if (*argv[0] == SWITCH_CHAR)
        {
            switch (*(argv[0]+1))
            {
                case 'r':
                    opts->sizeRow = 1<< ( (int)atol(argv[0]+2) );
                    break;

				case 'n':
                    opts->sizeRow = (int)atol(argv[0]+2);
                    break;

                case 'b':
                    opts->sizeBatch = (int)atol(argv[0]+2);
                    break;


                default:
                    break;
            }
        }

        if (error > oldError)
        {
            fprintf(stderr, "Invalid switch '%c%s'\n",SWITCH_CHAR, argv[0]+1);
        }

        argc -= 1;
        argv++;
    }

    return error;
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

	cudaStream_t streamNo = 0;

	// size
	Options opts;
	processArgs(argc, argv, &opts);
	int matrixRows = opts.sizeRow;
    int matrixSize = matrixRows * matrixRows;
	int sizeBatch = opts.sizeBatch;

    printf("matrixRows = %d , sizeBatch = %d...\n\n", matrixRows, sizeBatch );

	// data
	T_ELEM *A = NULL;
    T_ELEM **devPtrA = 0;
    T_ELEM **devPtrA_dev = NULL;

	T_ELEM **devPtrC = 0;
    T_ELEM **devPtrC_dev = NULL;

	// matrix A, input matrix
    devPtrA =(T_ELEM **)malloc( sizeBatch * sizeof(T_ELEM));
	for (int i = 0; i < sizeBatch ; i++)
    {
        err1 = cudaMalloc((void **)&devPtrA[i], matrixSize * sizeof(T_ELEM));
	}

    err1 = cudaMalloc((void **)&devPtrA_dev, sizeBatch * sizeof(T_ELEM));
    err1 = cudaMemcpy(devPtrA_dev, devPtrA, sizeBatch * sizeof(*devPtrA), cudaMemcpyHostToDevice);

    A  = (T_ELEM *)malloc(matrixSize * sizeof(T_ELEM));

	memset(A, 0xFF, matrixSize * sizeof(A[0]));
	fillupMatrixDebug( A, matrixRows );

	for (int i = 0; i < sizeBatch ; i++)
    {
		cublasSetMatrix( matrixRows, matrixRows, sizeof(A[0]), A, matrixRows, devPtrA[i], matrixRows);
	}

	// matrix C, output inverse matrix of A
	devPtrC =(T_ELEM **)malloc( sizeBatch * sizeof(T_ELEM));
	for (int i = 0; i < sizeBatch ; i++)
    {
        err1 = cudaMalloc((void **)&devPtrC[i], matrixSize * sizeof(T_ELEM));
	}

    err1 = cudaMalloc((void **)&devPtrC_dev, sizeBatch * sizeof(T_ELEM));
    err1 = cudaMemcpy(devPtrC_dev, devPtrC, sizeBatch * sizeof(*devPtrC), cudaMemcpyHostToDevice);

	// temp data
	int *d_pivotArray = NULL;
	int *d_infoArray = NULL;
	cudaMalloc( (void**)&d_pivotArray, matrixRows*sizeBatch*sizeof(int) );
	cudaMalloc( (void**)&d_infoArray,  sizeBatch*sizeof(int) );

	int *h_infoArray = NULL;
	h_infoArray = (int*)malloc( sizeBatch*sizeof(int) );

	// blas config
    cublasHandle_t handle;
	cublasCreate(&handle);

	// timer
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    cublasSetStream(handle, streamNo );
	
	cublasStatus_t status ;

	// LU factorization
	status = cublasSgetrfBatched(handle, 
		matrixRows, 
		devPtrA_dev, 
		matrixRows,
		d_pivotArray,
		d_infoArray,
		sizeBatch);
#if 0
	cudaMemcpy( h_infoArray,  d_infoArray, sizeBatch*sizeof(int), cudaMemcpyDeviceToHost );

	for(int i=0;i<sizeBatch;i++)
	{
		if( h_infoArray[i] == 0 )
		{
			//fprintf(stderr, "%d-th matrix lu-decompose successed, !\n", i );
			continue;
		}
		else if (h_infoArray[i] > 0)
		{
			fprintf(stderr, "%d-th matrix lu-decompose failed, U(%d,%d) = 0!\n", i, h_infoArray[i], h_infoArray[i] );
			continue;
		}
		else
		{
			fprintf(stderr, "%d-th matrix lu-decompose failed, the %d-th parameter had an illegal value!\n", i, -h_infoArray[i] );
			continue;
		}
	}
#endif

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
		d_pivotArray,
		devPtrC_dev,
		matrixRows,
		d_infoArray,
		sizeBatch);

	if (status != CUBLAS_STATUS_SUCCESS)
    {
          cudaError_t cuError = cudaGetLastError();
          fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status, cuError,cudaGetErrorString(cuError));
          return ;
    }

#if 0
	cudaMemcpy( h_infoArray,  d_infoArray, sizeBatch*sizeof(int), cudaMemcpyDeviceToHost );

	for(int i=0;i<sizeBatch;i++)
	{
		if( h_infoArray[i] == 0 )
		{
			//fprintf(stderr, "%d-th matrix lu-decompose successed, !\n", i );
			continue;
		}
		else if (h_infoArray[i] > 0)
		{
			fprintf(stderr, "%d-th matrix lu-decompose failed, U(%d,%d) = 0!\n", i, h_infoArray[i], h_infoArray[i] );
			continue;
		}
	}
#endif

	 cudaError_t cudaStatus = cudaThreadSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "!!!! GPU program execution error on cudaThreadSynchronize : cudaError=%d,(%s)\n", cudaStatus,cudaGetErrorString(cudaStatus));
            return ;
        }

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);



    // cleanup memory
	if (A) free (A); 

	for(int i = 0; i < sizeBatch; ++i) {       
            if(devPtrA[i]) cudaFree(devPtrA[i]);
            if(devPtrC[i]) cudaFree(devPtrC[i]);
        }  

	if (devPtrA) free(devPtrA);           
	if (devPtrC) free(devPtrC); 

	if (h_infoArray) cudaFree(h_infoArray); 

	if (devPtrA_dev) cudaFree(devPtrA_dev);
	if (devPtrC_dev) cudaFree(devPtrC_dev); 

	if (d_pivotArray) cudaFree(d_pivotArray);
	if (d_infoArray) cudaFree(d_infoArray); 

    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
