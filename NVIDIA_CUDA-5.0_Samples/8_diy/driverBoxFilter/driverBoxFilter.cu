
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
//#include "timerCPP.h"

using namespace std;

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "boxFilter.cuh"

// 最优配置
// 对于gtx480	BLOCK_SIZE_2D = (4,128)(3.7ms pointer), (8,32)(5.2ms texture)
// 对于gt640	BLOCK_SIZE_2D = (4,32)(40ms pointer), (4,128)(14.7ms texture)

#define		USE_TEXTURE_ADDRESS	1
#define		ENABLE_TIMER	1

#if	ENABLE_TIMER
#include <helper_timer.h>
#endif

int getMaxThreadsPerBlock()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("\nDevice: \"%s\"\n", deviceProp.name);
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

	return deviceProp.maxThreadsPerBlock;
}

class driverBoxFilter{
public:
	driverBoxFilter()	{ }
	~driverBoxFilter()	{ release(); }
	void initialize( );

	void run();
	void readFile(float **ppBuffer, std::string filename, int& width, int& height);
	void writeFile(float *buffer, std::string filename, int width, int height=1);
	void readSize(int **ppBuffer, std::string filename, int n);
	void writeSize(int *buffer, std::string filename, int n);
	bool verifyResult(float *p, float *p_ref, int n);

private:
	float *imCum_C, *C_ones, *N;
	float *pN, *pN_ref;
	int *pSize;
	int nBufferSizeFloat;
	int height, width, r;

	void release();
};

void driverBoxFilter::readFile(float **ppBuffer, std::string filename, int& width, int& height)
{
	FILE* fp=fopen( filename.c_str(),"rb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fread( &width, sizeof(int), 1, fp);
	fread( &height, sizeof(int), 1, fp);

	nBufferSizeFloat = width * height * sizeof(float);
	*ppBuffer = (float*)malloc( nBufferSizeFloat );

	fread( *ppBuffer, width*height*sizeof(float), 1, fp);

	fclose(fp);
}

void driverBoxFilter::writeFile(float *buffer, std::string filename, int width, int height/*=1*/)
{
	FILE* fp=fopen( filename.c_str(),"wb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fwrite( &width, sizeof(int), 1, fp);
	fwrite( &height, sizeof(int), 1, fp);
	fwrite( buffer, width*height*sizeof(float), 1, fp);
	fclose(fp);
}

void driverBoxFilter::readSize(int **ppBuffer, std::string filename, int n)
{
	FILE* fp=fopen( filename.c_str(),"rb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fread( &n, sizeof(int), 1, fp);
	*ppBuffer = (int*)malloc( n * sizeof(int) );

	fread( *ppBuffer, n * sizeof(int), 1, fp);

	fclose(fp);
}

void driverBoxFilter::writeSize(int *buffer, std::string filename, int n)
{
	FILE* fp=fopen( filename.c_str(),"wb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fwrite( &n, sizeof(int), 1, fp);
	fwrite( buffer, n*sizeof(int), n, fp);
	fclose(fp);
}

bool driverBoxFilter::verifyResult(float *p, float *p_ref, int n)
{
	for (int i=0; i<n; i++ ) 
	{
		if ( fabs( p[i] - p_ref[i] ) > 1e-5 )
			return false;
	}

	return true ;
}

void driverBoxFilter::initialize( )
{
	//read Parameter
	readSize( &pSize, "size.out", 1);
	r = pSize[0];
	readFile( &pN_ref, "N.out", width, height );


	pN = (float*)malloc( nBufferSizeFloat );

	cudaMalloc((void**)&imCum_C,nBufferSizeFloat);
    cudaMalloc((void**)&C_ones,nBufferSizeFloat);
    cudaMalloc((void**)&N,nBufferSizeFloat);

	thrust::fill( thrust::device_ptr<float>(C_ones), thrust::device_ptr<float>(C_ones + width*height), 1.0f);

	initTexture(width, height);

}

void driverBoxFilter::release( )
{
	free( pN );			free( pN_ref );
	cudaFree(imCum_C);  cudaFree(N);		cudaFree(C_ones);
}

void driverBoxFilter::run()
{
	boxfilter(C_ones,imCum_C,N,r,height,width);

	cudaMemcpy( pN, N, nBufferSizeFloat, cudaMemcpyDeviceToHost );

	bool bSuccess = verifyResult( pN, pN_ref, width*height );

	if( bSuccess )
		printf("right result \n");
	else
		printf("wrong result \n");
}

int main(int argc, char* argv[])
{

	driverBoxFilter	test;
	
	test.initialize();
	
	test.run();

	return 0;
}