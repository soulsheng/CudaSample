
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
void boxfilter0(float *imSrc,float *imCum_C,float *imDst,int r,int height,int width)
{
	
	int nBlockSize;

	// ��ֱY�����ۼӣ��뾶r����2*r+1�����ۼ�
	nBlockSize = height>BLOCK_SIZE_1D? BLOCK_SIZE_1D:  height;

	scan_kernel<<<width, nBlockSize >>>( imCum_C,imSrc, height, 1, width);// ��ֱY�����ۼ�

	delta<<<width,nBlockSize>>>( imDst,imCum_C,r, height, 1, width ); // ��ֱY����Ⱦ������

	// ˮƽX�����ۼӣ��뾶r����2*r+1�����ۼ�
	nBlockSize = width>BLOCK_SIZE_1D? BLOCK_SIZE_1D: width ;

	scan_kernel<<<height, nBlockSize >>>( imCum_C,imDst, width, width, 1);// ˮƽX�����ۼ�

	delta<<<height,nBlockSize>>>( imDst,imCum_C,r, width, width, 1 ); // ˮƽX����Ⱦ������
}



// process row
__device__ void
d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++)
    {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++)
    {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    // main loop
    for (int x = (r + 1); x < w - r; x++)
    {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++)
    {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void
d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++)
    {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += id[(h-1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void
d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void
d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}

#if 0
// texture version
// texture fetches automatically clamp to edge of image
__global__ void
d_boxfilter_x_tex(float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int x =- r; x <= r; x++)
    {
        t += tex2D(tex, x, y);
    }

    od[y * w] = t * scale;

    for (int x = 1; x < w; x++)
    {
        t += tex2D(tex, x + r, y);
        t -= tex2D(tex, x - r - 1, y);
        od[y * w + x] = t * scale;
    }
}

__global__ void
d_boxfilter_y_tex(float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int y = -r; y <= r; y++)
    {
        t += tex2D(tex, x, y);
    }

    od[x] = t * scale;

    for (int y = 1; y < h; y++)
    {
        t += tex2D(tex, x, y + r);
        t -= tex2D(tex, x, y - r - 1);
        od[y * w + x] = t * scale;
    }
}
#endif

void boxfilter(float *imSrc,float *imCum_C,float *imDst,int r,int height,int width)
{
	d_boxfilter_x_global<<< height / BLOCK_SIZE_1D, BLOCK_SIZE_1D, 0 >>>(imSrc, imCum_C, width, height, r);
	d_boxfilter_y_global<<< width / BLOCK_SIZE_1D, BLOCK_SIZE_1D, 0 >>>(imCum_C, imDst, width, height, r);

}
