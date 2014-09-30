
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
//#include "timerCPP.h"

using namespace std;

#include <cuda_runtime.h>


// 最优配置
// 对于gtx480	BLOCK_SIZE_2D = (4,128)(3.7ms pointer), (8,32)(5.2ms texture)
// 对于gt640	BLOCK_SIZE_2D = (4,32)(40ms pointer), (4,128)(14.7ms texture)

#define		USE_TEXTURE_ADDRESS	1
#define		ENABLE_TIMER	1

#if	ENABLE_TIMER
#include <helper_timer.h>
#endif

#if USE_TEXTURE_ADDRESS
texture<float, 2, cudaReadModeElementType> tex;
#endif

__global__ void warmCUDA( float* C_min_RGB, float *C_win_dark,int height,int width )
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=threadIdx.y+blockIdx.y*blockDim.y;
	
	if(i<height&&j<width)
	{
		C_win_dark[i*width +j] = C_min_RGB[i*width +j] ;
	}
}

__global__ void minwCUDA( float* C_min_RGB, float *C_win_dark,int height,int width,int winsize)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=threadIdx.y+blockIdx.y*blockDim.y;

	int indexis=0,indexie=0;
	int indexjs=0,indexje=0;

	float temp=999.0;
	
	if(i<height&&j<width)
	{
		if(i<=2*winsize)
		{
			indexis=winsize;
			indexie=i+7;
			if(j<=2*winsize)
			{
				indexjs=winsize;
				indexje=j+7;
			}
			else if(j>2*winsize&&j<width-2*winsize)
			{
				indexjs=j-7;
				indexje=j+7;
			}
			else if(j>=width-2*winsize&&j<width)
			{
				indexjs=j-7;
				indexje=width-winsize-1;
			}
		}
		else if(i>2*winsize&&i<height-2*winsize)
		{
			indexis=i-7;
			indexie=i+7;
			if(j<=2*winsize)
			{
				indexjs=winsize;
				indexje=j+7;
			}
			else if(j>2*winsize&&j<width-2*winsize)
			{
				indexjs=j-7;
				indexje=j+7;
			}
			else if(j>=width-2*winsize&&j<width)
			{
				indexjs=j-7;
				indexje=width-winsize-1;
			}
		}
		else if(i>=height-2*winsize&&i<height)
		{
			indexis=i-7;
			indexie=height-winsize-1;
			if(j<=2*winsize)
			{
				indexjs=winsize;
				indexje=j+7;
			}
			else if(j>2*winsize&&j<width-2*winsize)
			{
				indexjs=j-7;
				indexje=j+7;
			}
			else if(j>=width-2*winsize&&j<width)
			{
				indexjs=j-7;
				indexje=width-winsize-1;
			}
		}
		for(int m=indexis;m<=indexie;m++)
			for(int n=indexjs;n<=indexje;n++)
#if USE_TEXTURE_ADDRESS
			{
				float p = tex2D(tex, n, m);
				temp = temp< p ?temp:p ;
			}
#else
				temp=temp<C_min_RGB[m*width+n]?temp:C_min_RGB[m*width+n];
#endif

		C_win_dark[i*width+j]=C_win_dark[i*width+j]<temp?C_win_dark[i*width+j]:temp;
	}

}

bool compareResult( float* win_dark, float* win_dark_ref, int n ) 
{
	for (int i=0; i<n; i++ ) 
	{
		if ( fabs( win_dark[i] - win_dark_ref[i] ) > 1e-5 )
			return false;
	}

	return true ;
}

int getMaxThreadsPerBlock()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("\nDevice: \"%s\"\n", deviceProp.name);
	printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);

	return deviceProp.maxThreadsPerBlock;
}
extern "C"
void cudaGenerateDrakKernel( )
{
	cudaError_t cudaStatus,err;
	FILE *fp=NULL;
	int height = 720;
	int width = 1280;
	int C_size1 = height * width * sizeof(float) ;
	int win_size = 7;
	//timerC GPUtime;
#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
#endif

	////////////////////////////////////////////////
	float *win_dark = (float *)calloc(height*width, sizeof(float));
	fp=fopen( "win_dark.in","rb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fread( win_dark, height*width*sizeof(float), 1, fp);
	fclose(fp);

	float *C_win_dark;
	cudaStatus = cudaMalloc((void**)&C_win_dark,C_size1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
	/*cudaStatus = cudaMemcpy(C_win_dark,win_dark,C_size1, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }*/

	////////////////////////////////////////////////
	float *min_RGB = (float *)calloc(height*width,sizeof(float));
	fp=fopen( "min_RGB.in","rb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fread( min_RGB, height*width*sizeof(float), 1, fp);
	fclose(fp);

	float *C_min_RGB;
	cudaStatus = cudaMalloc((void**)&C_min_RGB,C_size1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	////////////////////////////////////////////////
	float *win_dark_ref = (float *)calloc(height*width,sizeof(float));
	fp=fopen( "win_dark.out","rb");
	if(fp == NULL)
	{
		printf("Cann't open the file!\n");
		exit(0);
	}
	fread( win_dark_ref, height*width*sizeof(float), 1, fp);
	fclose(fp);

#if USE_TEXTURE_ADDRESS
	// cuda texture ------------------------------------------------------------------------------------------
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* arr_min_RGB;
    err = cudaMallocArray(&arr_min_RGB, &channelDesc, width, height);
    err = cudaMemcpyToArray(arr_min_RGB, 0, 0, min_RGB, C_size1, cudaMemcpyHostToDevice);

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;

    // Bind the array to the texture
    err = cudaBindTextureToArray(tex, arr_min_RGB, channelDesc);
#endif

	// warm up cuda
	//dim3 grid(16,16);
	//dim3 block(16,16);
	//warmCUDA<<<grid,block>>>( C_min_RGB, C_win_dark,height,width );

	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
		printf("cudaGetLastError");
	}

	float	minTime = 1000.0f;
	int		minBlockSizeScale	= 1024;
	int		minBlockSizeBase	= 1024;
	int nMaxThreadsPerBlock = getMaxThreadsPerBlock();
	int blockSizeScale	= 4;
	int blockSizeBase	= 4;
	for( blockSizeBase	= 4;blockSizeBase * blockSizeScale <= nMaxThreadsPerBlock; blockSizeBase*=2)
	{
		for( blockSizeScale	= 4;blockSizeBase * blockSizeScale <= nMaxThreadsPerBlock; blockSizeScale*=2)
		{
#if USE_TEXTURE_ADDRESS
			cudaMemcpyToArray(arr_min_RGB, 0, 0, min_RGB, C_size1, cudaMemcpyHostToDevice);
			cudaBindTextureToArray(tex, arr_min_RGB);
#else
			cudaStatus = cudaMemcpy(C_min_RGB,min_RGB,C_size1, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
#endif
			cudaStatus = cudaMemcpy(C_win_dark,win_dark,C_size1, cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}


			dim3 grid((height+blockSizeScale-1)/blockSizeScale,(width+blockSizeBase-1)/blockSizeBase);
			dim3 block(blockSizeScale,blockSizeBase);

#if ENABLE_TIMER
			cudaDeviceSynchronize();
			sdkResetTimer(&timer);
			sdkStartTimer(&timer);
#endif

			minwCUDA<<<grid,block>>>( C_min_RGB, C_win_dark,height,width,win_size);
			err = cudaGetLastError();
			if( err != cudaSuccess )
				printf("Error minwCUDA!\n");
			
			cudaDeviceSynchronize();

#if ENABLE_TIMER
			sdkStopTimer(&timer);
							
			float gTime = sdkGetTimerValue(&timer);
			if ( gTime< minTime)
			{
				minTime = gTime;
				minBlockSizeScale	= blockSizeScale;
				minBlockSizeBase	= blockSizeBase;
			}

			cout << "GPUtime: " << gTime  << endl;
			cout << "blockSize: " << blockSizeScale*blockSizeBase << " = " << blockSizeScale << " * " << blockSizeBase << "\n" << endl;
#endif
	
			cudaStatus = cudaMemcpy(win_dark,C_win_dark,C_size1,cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}

		

			bool result_C=compareResult( win_dark, win_dark_ref, height*width ) ;

			if(result_C==false)
				printf("Error compareResult!\n");

		}
		blockSizeScale	= 4;
	}

	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
		printf("cudaGetLastError");
	}

#if ENABLE_TIMER
	cout << "Minimum GPUtime: " << minTime  << endl;
	cout << "blockSize: " << minBlockSizeScale*minBlockSizeBase << " = " << minBlockSizeScale << " * " << minBlockSizeBase << "\n" << endl;

	sdkDeleteTimer(&timer);
#endif

	free( win_dark );
	free( min_RGB );
	cudaFree( C_win_dark ) ;
	cudaFree( C_min_RGB );
#if USE_TEXTURE_ADDRESS
	cudaUnbindTexture( tex );
	cudaFreeArray( arr_min_RGB );
#endif
}

int main(int argc, char* argv[])
{
	cudaGenerateDrakKernel();

	return 0;
}