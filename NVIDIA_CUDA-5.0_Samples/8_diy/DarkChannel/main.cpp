

#include "DarkChannel.h"
#include "DarkChannel.cuh"
//#include "warmupCUDA.cuh"

//#include "timerCU.h"
#include "bmpHandler.h"
//#include "arrayUtility.h"
//#include "bmpResizer.h"
//#include "bmpResizerGPU.h"

#include <cuda_runtime.h>
#include <helper_timer.h>

#include <stdlib.h>
#include <iostream>
using namespace std;

#define USE_GPU	1
#define	IMAGE_FILE_TEST		"DarkChannel.bmp"
#define	ENABLE_RESIZE		0
#define	ENABLE_TIMER	1

int main()
{
	int width, height;
	BMPHandler::getImageSize( IMAGE_FILE_TEST, height, width );
	
	unsigned int *RGBA_In=(unsigned int *)malloc(height*width*sizeof(unsigned int));
	unsigned int *RGBA_Out=(unsigned int *)malloc(height*width*sizeof(unsigned int));

#if ENABLE_RESIZE
	float scale = 2.0f;
	int	  width_resized = int( width / scale );
	int	  height_resized = int( height / scale );

	byte *B_In_small,*G_In_small,*R_In_small;//存储原始图像的垂直翻转，第一行跟最后一行交换像素值
	mallocArray( &B_In_small, height_resized, width_resized );
	mallocArray( &G_In_small, height_resized, width_resized );
	mallocArray( &R_In_small, height_resized, width_resized );


	BMPResizer::readImageDataAndResize( IMAGE_FILE_TEST, 
		B_In, G_In, R_In, 
		B_In_small, G_In_small, R_In_small, 
		height_resized, width_resized );
#else
	BMPHandler::readImageData( IMAGE_FILE_TEST, RGBA_In );
#endif

#if USE_GPU
	// GPU buffer original size
	int nBufferSizeOriginal = height*width*sizeof(unsigned int);

	unsigned int *d_RGBA_In;
	cudaMalloc((void**)&d_RGBA_In, nBufferSizeOriginal );
	cudaMemcpy( d_RGBA_In, RGBA_In, nBufferSizeOriginal, cudaMemcpyHostToDevice );

	
#if ENABLE_RESIZE
	// GPU buffer resize begin
	int  nBufferSizeResized = width_resized * height_resized * sizeof(byte);
	byte *d_R_In_resized,*d_G_In_resized,*d_B_In_resized;
	cudaMalloc((void**)&d_R_In_resized, nBufferSizeResized );
	cudaMalloc((void**)&d_G_In_resized, nBufferSizeResized );
	cudaMalloc((void**)&d_B_In_resized, nBufferSizeResized );

	BMPResizerGPU::readImageDataAndResize( IMAGE_FILE_TEST, 
		d_B_In, d_G_In, d_R_In, 
		d_B_In_resized, d_G_In_resized, d_R_In_resized, 
		height_resized, width_resized );
#endif



	unsigned int *d_RGBA_Out;
	cudaMalloc((void**)&d_RGBA_Out, nBufferSizeOriginal );

	// warmupCUDA( RGBA_In, width, height );
	DarkChannelGPU	m_DarkChannel( width, height );


#endif

#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
#endif

#if USE_GPU
	// 图像去雾处理
	m_DarkChannel.Enhance( d_RGBA_In, d_RGBA_Out );
#else
	DarkChannel( RGBA_In, RGBA_Out, width, height );
#endif

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time ALL: %f (ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);
#endif

#if USE_GPU
	// copy gpu buffer back to cpu
	cudaMemcpy(RGBA_Out,d_RGBA_Out,nBufferSizeOriginal,cudaMemcpyDeviceToHost);
#endif

	// 保存结果图像
	BMPHandler::saveImage("out.bmp", RGBA_Out, height, width );


	//Memory release


	free( RGBA_In );
	free( RGBA_Out );

#if USE_GPU
	cudaFree( d_RGBA_In );
	cudaFree( d_RGBA_Out );
#endif
	
	cudaDeviceReset();

	return 0;
}
