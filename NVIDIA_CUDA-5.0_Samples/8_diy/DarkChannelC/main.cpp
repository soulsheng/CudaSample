

#include "DarkChannel.h"
//#include "DarkChannel.cuh"
//#include "warmupCUDA.cuh"

//#include "timerCU.h"
#include "bmpHandler.h"
//#include "arrayUtility.h"
//#include "bmpResizer.h"
//#include "bmpResizerGPU.h"

//#include <cuda_runtime.h>
#include "helper_timer.h"

#include <stdlib.h>
#include <iostream>
using namespace std;


#define	IMAGE_FILE_TEST		"DarkChannel.bmp"

#define	ENABLE_TIMER	1

int main()
{
	int width, height;
	BMPHandler::getImageSize( IMAGE_FILE_TEST, height, width );
	
	unsigned int *RGBA_In=(unsigned int *)malloc(height*width*sizeof(unsigned int));
	unsigned int *RGBA_Out=(unsigned int *)malloc(height*width*sizeof(unsigned int));

	BMPHandler::readImageData( IMAGE_FILE_TEST, RGBA_In );


#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
#endif


	DarkChannel( RGBA_In, RGBA_Out, width, height );

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time ALL: %f (ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);
#endif


	// ±£´æ½á¹ûÍ¼Ïñ
	BMPHandler::saveImage("out.bmp", RGBA_Out, height, width );


	//Memory release
	free( RGBA_In );
	free( RGBA_Out );

	return 0;
}
