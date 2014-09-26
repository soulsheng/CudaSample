
#include "boxFilter.cuh"
#include <cuda_runtime.h>

#define		BLOCK_SIZE_1D	64	// BLOCK_SIZE = BLOCK_SIZE_1D

__global__ void scan_kernel( float *imDst_C,float * imSrc_C, int nElement, int nBlockStride, int nThreadStride )
{
	__shared__ float sdata[BLOCK_SIZE_1D*2];//blockDim.x*2
	int offset = 0;
	for(  ; offset < nElement ; offset += blockDim.x )
	{
		int length = blockDim.x;
		if( offset + blockDim.x > nElement )
		{
			length = nElement % blockDim.x;
		}

		if( threadIdx.x+offset >= nElement )
			return;

		sdata[threadIdx.x] = imSrc_C[(threadIdx.x+offset)*nThreadStride + blockIdx.x*nBlockStride ];

		if( offset!=0 && threadIdx.x == 0 )
			sdata[0] += imDst_C[(threadIdx.x+offset-1)*nThreadStride + blockIdx.x*nBlockStride ];

		__syncthreads();

		int first = 0;

		for ( int d=1;d<length; d+=d, first=blockDim.x-first )
		{
			if( threadIdx.x < d )
				sdata[threadIdx.x+blockDim.x-first] = sdata[threadIdx.x+first];
			else
				sdata[threadIdx.x+blockDim.x-first] = sdata[threadIdx.x+first] + sdata[threadIdx.x-d+first];
			__syncthreads();
		}

		imDst_C[(threadIdx.x+offset)*nThreadStride + blockIdx.x*nBlockStride ] = sdata[threadIdx.x+first];
		__syncthreads();

	}
}

__global__ void delta( float *imDst_C,float *imCum_C,int r, int nElement, int nBlockStride, int nThreadStride)
{
	if( threadIdx.x > nElement )
		return;

	for( int nCurrentIndex = threadIdx.x;nCurrentIndex< nElement; nCurrentIndex+=blockDim.x )
	{
		int nCurrentElement = nCurrentIndex*nThreadStride + blockIdx.x*nBlockStride;

		if( nCurrentIndex < r + 1 )
			imDst_C[ nCurrentElement ] =	imCum_C[ nCurrentElement + r*nThreadStride ];
		else if ( nCurrentIndex >= r + 1 && nCurrentIndex < nElement-r )
			imDst_C[ nCurrentElement ] =	imCum_C[ nCurrentElement + r*nThreadStride ] -  
															imCum_C[ nCurrentElement - (r + 1)*nThreadStride ] ;
		else //if ( nCurrentIndex >= height-r && nCurrentIndex < height )
			imDst_C[ nCurrentElement ] =	imCum_C[ (nElement-1)*nThreadStride + blockIdx.x*nBlockStride ] -  
															imCum_C[ nCurrentElement - (r + 1)*nThreadStride ] ;
	}

}

//boxfilter
/*%   BOXFILTER   O(1) time box filtering using cumulative sum
%
%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
%   - Running time independent of r; 
%   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
%   - But much faster.*/
void boxfilter(float *imSrc,float *imCum_C,float *imDst,int r,int height,int width)
{
	
	int nBlockSize;

	// 垂直Y方向累加，半径r，即2*r+1个数累加
	nBlockSize = height>BLOCK_SIZE_1D? BLOCK_SIZE_1D:  height;

	scan_kernel<<<width, nBlockSize >>>( imCum_C,imSrc, height, 1, width);// 垂直Y方向累加

	delta<<<width,nBlockSize>>>( imDst,imCum_C,r, height, 1, width ); // 垂直Y方向等距离相减

	// 水平X方向累加，半径r，即2*r+1个数累加
	nBlockSize = width>BLOCK_SIZE_1D? BLOCK_SIZE_1D: width ;

	scan_kernel<<<height, nBlockSize >>>( imCum_C,imDst, width, width, 1);// 水平X方向累加

	delta<<<height,nBlockSize>>>( imDst,imCum_C,r, width, width, 1 ); // 水平X方向等距离相减
}