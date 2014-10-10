

//#include "DarkChannel.h"
#include "DarkChannel.cuh"
//#include "warmupCUDA.cuh"

//#include "timerCU.h"
#include "bmpHandler.h"
//#include "arrayUtility.h"
//#include "bmpResizer.h"
//#include "bmpResizerGPU.h"
#include "resizeNN.cuh"

#include <cuda_runtime.h>
#include "helper_timer.h"

#include <stdlib.h>
#include <iostream>
using namespace std;

#define USE_GPU	1
#define	IMAGE_FILE_TEST		"DarkChannel.bmp"
#define	ENABLE_RESIZE		1
#define	ENABLE_TIMER	1

int main()
{
	int width, height;
	BMPHandler::getImageSize( IMAGE_FILE_TEST, height, width );
	
	unsigned int *RGBA_In=(unsigned int *)malloc(height*width*sizeof(unsigned int));
	unsigned int *RGBA_Out=(unsigned int *)malloc(height*width*sizeof(unsigned int));
	
	BMPHandler::readImageData( IMAGE_FILE_TEST, RGBA_In );

	float scale = 2.0f;
	int	  width_resized = int( width / scale );
	int	  height_resized = int( height / scale );

#if USE_GPU
	// GPU buffer original size
	int nBufferSizeOriginal = height*width*sizeof(unsigned int);

	unsigned int *d_RGBA_In;
	cudaMalloc((void**)&d_RGBA_In, nBufferSizeOriginal );
	cudaMemcpy( d_RGBA_In, RGBA_In, nBufferSizeOriginal, cudaMemcpyHostToDevice );

	unsigned int *d_RGBA_In_resized = NULL;
#if ENABLE_RESIZE
	// GPU buffer resize begin
	int  nBufferSizeResized = width_resized * height_resized * sizeof(unsigned int);
	cudaMalloc((void**)&d_RGBA_In_resized, nBufferSizeResized );

	CUResizeNN::process( d_RGBA_In, d_RGBA_In_resized, width, height, width_resized, height_resized );
#endif



	unsigned int *d_RGBA_Out;
	cudaMalloc((void**)&d_RGBA_Out, nBufferSizeOriginal );

	// warmupCUDA( RGBA_In, width, height );
#if ENABLE_RESIZE
	DarkChannelGPU	m_DarkChannel( width_resized, height_resized, width, height );
#else
	DarkChannelGPU	m_DarkChannel( width, height );
#endif

#endif

#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
#endif

#if USE_GPU
	// 图像去雾处理
	m_DarkChannel.Enhance( d_RGBA_In, d_RGBA_Out, d_RGBA_In_resized );
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
