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
8.3. LU分解 cublas<t>getrfBatched(): http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched  in cuda5.0
8.4. 求逆   cublas<t>getriBatched(): http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched  in cuda5.5

10.2. 稀疏矩阵LU分解 cusparse<t>csrilu0 : http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrilu0

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

int verifyResult( T_ELEM *A, T_ELEM *B , int n ) 
{
    T_ELEM *C  = (T_ELEM *)malloc( n * n * sizeof(T_ELEM));

	/* Host implementation of a simple version of sgemm */
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            T_ELEM prod = 0;

            for (k = 0; k < n; ++k)
            {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = prod;
        }
    }

	for (i = 0; i < n; ++i)
    {
		if( fabs( C[i * n + i] - 1.0f ) > 1.0e-3 )
		{
			free(C);
			return i;
		}
	}

	free(C);
	return 0;
}

int verifyResultBLAS( T_ELEM *A, T_ELEM *B , int n ) 
{
    cublasStatus_t status;
	
	// blas config
    cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;

	T_ELEM *d_C;
	cudaMalloc((void **)&d_C, n * n * sizeof(T_ELEM));
    T_ELEM *C  = (T_ELEM *)malloc( n * n * sizeof(T_ELEM));

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, d_C, n);

	cudaMemcpy( C, d_C, n * n * sizeof(T_ELEM) , cudaMemcpyDeviceToHost );


	cudaFree( d_C );
	cublasDestroy(handle);

	for (int i = 0; i < n; ++i)
    {
		if( fabs( C[i * n + i] - 1.0f ) > 1.0e-3 )
		{
			free(C);
			return i;
		}
	}

	free(C);
	return 0;

}

// 批量矩阵求逆，调用blas库， C[i] = A[i] ^ -1
// inverse batch of matrices
int inverseMatrixBLAS( T_ELEM **A , T_ELEM **C , int matrixRows , int sizeBatch ,int bDebug = false )
{
	int  matrixSize = matrixRows * matrixRows;
	cudaError_t err1;

	// temp data
    T_ELEM **devPtrA = 0;
    T_ELEM **devPtrA_dev = NULL;

	T_ELEM **devPtrC = 0;
    T_ELEM **devPtrC_dev = NULL;

	// temp data for matrix A, input matrix
    devPtrA =(T_ELEM **)malloc( sizeBatch * sizeof(T_ELEM));
	for (int i = 0; i < sizeBatch ; i++)
    {
        cudaMalloc((void **)&devPtrA[i], matrixSize * sizeof(T_ELEM));
		cublasSetMatrix( matrixRows, matrixRows, sizeof(T_ELEM), A[i], matrixRows, devPtrA[i], matrixRows);
	}
	
    cudaMalloc((void **)&devPtrA_dev, sizeBatch * sizeof(T_ELEM));
    cudaMemcpy(devPtrA_dev, devPtrA, sizeBatch * sizeof(*devPtrA), cudaMemcpyHostToDevice);


	// temp data for matrix C, output inverse matrix of A
	devPtrC =(T_ELEM **)malloc( sizeBatch * sizeof(T_ELEM));
	for (int i = 0; i < sizeBatch ; i++)
    {
        cudaMalloc((void **)&devPtrC[i], matrixSize * sizeof(T_ELEM));
	}

    cudaMalloc((void **)&devPtrC_dev, sizeBatch * sizeof(T_ELEM));
    cudaMemcpy(devPtrC_dev, devPtrC, sizeBatch * sizeof(*devPtrC), cudaMemcpyHostToDevice);


	// temp data middle
	int *d_pivotArray = NULL;
	int *d_infoArray = NULL;
	cudaMalloc( (void**)&d_pivotArray, matrixRows*sizeBatch*sizeof(int) );
	cudaMalloc( (void**)&d_infoArray,  sizeBatch*sizeof(int) );
	
	int *h_infoArray = NULL;

	// blas config
    cublasHandle_t handle;
	cublasCreate(&handle);
    cublasSetStream(handle, 0 );

	// timer begin
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	// LU factorization ， 矩阵LU三角分解
	cublasStatus_t status = cublasSgetrfBatched(handle, 
		matrixRows, 
		devPtrA_dev, 
		matrixRows,
		d_pivotArray,
		d_infoArray,
		sizeBatch);

	
	if (status != CUBLAS_STATUS_SUCCESS)
    {
          cudaError_t cuError = cudaGetLastError();
          fprintf(stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status, cuError,cudaGetErrorString(cuError));
          return -1;
    }

	// 检测LU分解是否顺利执行
	if( bDebug )
	{
		h_infoArray = (int*)malloc( sizeBatch*sizeof(int) );

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
	}

#if 1// if 0 单独测上一步LU分解的时间， 1536*1536矩阵时间分布：700-1800ms GPU480
	// inversion of matrices A, output result to matrices C ， 三角矩阵求逆
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
          return -1 ;
    }
#endif
	// 检测三角矩阵求逆是否顺利执行
	if( bDebug )
	{
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
	}

	// timer end
	cudaError_t cudaStatus = cudaThreadSynchronize();

	if (cudaStatus != cudaSuccess)
    {
		fprintf(stderr, "!!!! GPU program execution error on cudaThreadSynchronize : cudaError=%d,(%s)\n", cudaStatus,cudaGetErrorString(cudaStatus));
		return -1;
	}

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

	// 逆矩阵结果从显存返回内存，gpu -> cpu
	for(int i=0; i< sizeBatch; i++)
	{
		cudaMemcpy( C[i], devPtrC[i], matrixSize * sizeof(T_ELEM) , cudaMemcpyDeviceToHost );
	}

	// 验证逆矩阵计算结果是否正确
	if( bDebug )
	{
		int bStatus = 0;
		for(int i=0; i< sizeBatch; i++)
		{
	#if 0
			bStatus = verifyResultBLAS( devPtrA[i], devPtrC[i], matrixRows );
	#else
			bStatus = verifyResult( A[i], C[i], matrixRows );	
	#endif
			if( bStatus )
			{
				printf( "Matrix Inverse Wrong! A*A^(-1) [%d,%d] !=1 \n" ,bStatus ,bStatus );
				break;
			}
		}
	}


    // 释放辅助计算用到的临时内存
	for(int i = 0; i < sizeBatch; ++i) 
	{       
            if(devPtrA[i]) cudaFree(devPtrA[i]);
            if(devPtrC[i]) cudaFree(devPtrC[i]);
	}  

	if (devPtrA) free(devPtrA);           
	if (devPtrC) free(devPtrC); 

	if (devPtrA_dev)	cudaFree(devPtrA_dev);
	if (devPtrC_dev)	cudaFree(devPtrC_dev); 

	if (d_pivotArray)	cudaFree(d_pivotArray);
	if (d_infoArray)	cudaFree(d_infoArray); 
	if (h_infoArray)	free(h_infoArray); 

	cublasDestroy(handle);

	return 0;
}

// 单个矩阵求逆，调用blas库， C = A ^ -1
// inverse a matrix
int inverseMatrixBLAS( T_ELEM *A , T_ELEM *C, int matrixRows, int bDebug = false)
{
	// 初始化matrix A, input matrix
	T_ELEM **ABatch  = (T_ELEM **)malloc( 1 * sizeof(T_ELEM*));
	*ABatch  = A;

	// matrix C, output inverse matrix of A
	T_ELEM **CBatch  = (T_ELEM **)malloc( 1 * sizeof(T_ELEM*));
	*CBatch  = C;

	inverseMatrixBLAS( ABatch, CBatch, matrixRows, 1, bDebug ) ;

	if (ABatch) free (ABatch);
    if (CBatch) free (CBatch);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

	// size
	Options opts;
	processArgs(argc, argv, &opts);
	int matrixRows = opts.sizeRow;
    int matrixSize = matrixRows * matrixRows;
	int sizeBatch = opts.sizeBatch;

    printf("matrixRows = %d , sizeBatch = %d...\n\n", matrixRows, sizeBatch );

	// data
	T_ELEM **A = NULL;
	T_ELEM **C = NULL;

	// 初始化matrix A, input matrix
	A  = (T_ELEM **)malloc(sizeBatch * sizeof(T_ELEM*));
	for (int i = 0; i < sizeBatch ; i++)
    {
		A[i]  = (T_ELEM *)malloc(matrixSize * sizeof(T_ELEM));

		// 矩阵用随机数模拟
		memset(A[i], 0xFF, matrixSize * sizeof(A[0]));
		fillupMatrixDebug( A[i], matrixRows );
	}

	// matrix C, output inverse matrix of A
	C  = (T_ELEM **)malloc(sizeBatch * sizeof(T_ELEM*));
	for (int i = 0; i < sizeBatch ; i++)
    {
		C[i]  = (T_ELEM *)malloc(matrixSize * sizeof(T_ELEM));
	}

	// 单个矩阵求逆
	int bTestResult = inverseMatrixBLAS( A[0], C[0], matrixRows );
	if( bTestResult==0 )
		printf("\nSingle Matrix Inverse Successfully!\n\n\n");

	// 批量矩阵求逆， 默认10个
	bTestResult = inverseMatrixBLAS( A, C, matrixRows, sizeBatch );
	if( bTestResult==0 )
		printf("\nBatch of Matrix Inverse Successfully!\n");

    // cleanup memory
	for(int i = 0; i < sizeBatch; ++i) 
	{  
		if (A[i]) free (A[i]); 
		if (C[i]) free (C[i]);
    }  
    if (A) free (A);
    if (C) free (C);

    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
