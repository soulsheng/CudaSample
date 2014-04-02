

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes CUDA BLAS
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "matrixInverse.cuh"

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

void fillupMatrixDebug(T_ELEM *A , int size )
{
	for (int j = 0; j < size; j++)
	{
		for (int i = 0; i < size; i++)
		{
			if( i==j || i==j-1 || i==j+1 )
				A[i + size*j ] = rand()%(size*size);
		}
	}
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
	printf("%s Starting...\n\n", argv[0]);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int devID = 0;//findCudaDevice(argc, (const char **)argv);

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
		memset(A[i], 0, matrixSize * sizeof(A[0]));
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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
	main(int argc, char **argv)
{
	runTest(argc, argv);
}
