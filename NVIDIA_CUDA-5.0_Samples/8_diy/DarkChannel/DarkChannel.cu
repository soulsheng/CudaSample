/*************************************************
  Copyright (C), 2014.
  File name:Drack.cu     
  Author:Li Zhe       Version:1.0        Date:2014.03.24 
  update:SoulSheng		Date: 2014-8-22
  Description:
  Function List:  
*************************************************/
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "DarkChannel.cuh"
//#include "timerCU.h"
//#include "bmpResizerGPU.h"
//#include "warmupCUDA.cuh"

#include <iostream>
using namespace std;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

// 最优配置
// 对于gt640  BLOCK_SIZE_1D = 64(128,256同), BLOCK_SIZE_2D = (4,32)(45ms pointer), (4,128)(21ms texture)
// 对于gt480  BLOCK_SIZE_1D = 64(128,256同), BLOCK_SIZE_2D = (4,128)(6.7ms pointer), (8,32)(9.2ms texture)
// 
#define		BLOCK_SIZE_1D	64	// BLOCK_SIZE = BLOCK_SIZE_1D
#define		BLOCK_SIZE_2D_X	4	// BLOCK_SIZE = BLOCK_SIZE_2D_X * BLOCK_SIZE_2D_Y
#define		BLOCK_SIZE_2D_Y	128	// BLOCK_SIZE = BLOCK_SIZE_2D_X * BLOCK_SIZE_2D_Y


#define		ENABLE_TIMER	0


//矩阵元素除以255 
__global__ void division(float *d_R_In,float *d_G_In,float *d_B_In,
	byte *d_P_In_byte, byte *d_G_In_byte, byte *d_B_In_byte, 
	int height, int width)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    
    if(i<width*height)
    {
        d_R_In[i] = d_P_In_byte[i] / 255.0f;
		d_G_In[i] = d_G_In_byte[i] / 255.0f;  
		d_B_In[i] = d_B_In_byte[i] / 255.0f;
    }
}

__global__ void winTT(float *C_win_dark,float *C_win_t,int height, int width)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
     
    if(i<height*width)
    {
        C_win_t[i] = 1-0.95 * C_win_dark[i];
    }
}

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

void DarkChannelGPU::sort_by_key( float* pKey, unsigned int* pValue, int size )
{
	thrust::sequence( 
		thrust::device_ptr<unsigned int>( pValue ), 
		thrust::device_ptr<unsigned int>( pValue + size ) );

	thrust::sort_by_key(	thrust::device_ptr<float>( pKey ), 
							thrust::device_ptr<float>( pKey + size ), 
							thrust::device_ptr<unsigned int>( pValue ),
							thrust::greater<float>() ); // 从大到小

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

//矩阵点乘
__global__ void dotProduct(float *C,float *A,float *B,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
		C[i]=A[i]*B[i];
    }
}
__global__ void cov_Ip(float *cov_Ip_x,float *mean_Ip,float *N,float *mean_I,float *mean_p,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
		cov_Ip_x[i]=mean_Ip[i]/N[i]-(mean_I[i]*mean_p[i])/(N[i]*N[i]);
    }
}

__global__ void var_I(float *var_I_x,float *N,float *mean_I1,float *mean_I2,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
		var_I_x[i]=var_I_x[i]/N[i]-(mean_I1[i]*mean_I2[i])/(N[i]*N[i]);
    }
}

__global__ void ax(float *ar,float *ag,float *ab,float *var_I_rr,float *var_I_rg,float *var_I_rb,float *var_I_gg,float *var_I_gb,float *var_I_bb,
	                float *cov_Ip_r,float *cov_Ip_g,float *cov_Ip_b,float eps,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	float z=0.0;
	float S0,S1,S2,S3,S4,S5,S6,S7,S8;
	float I0,I1,I2,I3,I4,I5,I6,I7,I8;

	if(i<height*width)
	{
		S0=var_I_rr[i]+eps;
		S1=var_I_rg[i];
		S2=var_I_rb[i];
		S3=var_I_rg[i];
		S4=var_I_gg[i]+eps;
		S5=var_I_gb[i];
		S6=var_I_rb[i];
		S7=var_I_gb[i];
		S8=var_I_bb[i]+eps;

		z=S0*S4*S8+S1*S5*S6+S2*S3*S7
		 -S2*S4*S6-S0*S5*S7-S1*S3*S8;
        
		I0=(1/z)*(S4*S8-S5*S7);      
		I1=-(1/z)*(S3*S8-S5*S6);
		I2=(1/z)*(S3*S7-S4*S6);
		I3=-(1/z)*(S1*S8-S2*S7);
		I4=(1/z)*(S0*S8-S2*S6);
		I5=-(1/z)*(S0*S7-S1*S6);
		I6=(1/z)*(S1*S5-S2*S4);
		I7=-(1/z)*(S0*S5-S2*S3);
		I8=(1/z)*(S0*S4-S1*S3);

		ar[i]=cov_Ip_r[i]*I0+cov_Ip_g[i]*I3+cov_Ip_b[i]*I6;
		ag[i]=cov_Ip_r[i]*I1+cov_Ip_g[i]*I4+cov_Ip_b[i]*I7;
		ab[i]=cov_Ip_r[i]*I2+cov_Ip_g[i]*I5+cov_Ip_b[i]*I8;
		
	}
}
__global__ void meanb(float *mean_b,float *ar,float *ag,float *ab,float *mean_p,float *mean_I_r,float *mean_I_g,float *mean_I_b,float *N,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
		mean_b[i]=(mean_p[i]-ar[i]*mean_I_r[i]-ag[i]*mean_I_g[i]-ab[i]*mean_I_b[i])/N[i];
	}
}
/*q[i*width+j]=(box_aI_r[i*width+j]*R_P[i*width+j]
			            + box_aI_g[i*width+j]*G_P[i*width+j]
						+ box_aI_b[i*width+j]*B_P[i*width+j]
						+ box_b[i*width+j])/N[i*width+j];*/
__global__ void qbox(float *q,float *box_aI_r,float *box_aI_g,float *box_aI_b,float *d_R_In,float *d_G_In,float *d_B_In,float *box_b,float *N,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
		q[i]=(box_aI_r[i]*d_R_In[i]+box_aI_g[i]*d_G_In[i]+box_aI_b[i]*d_B_In[i]+box_b[i])/N[i];
	}
}

//rangemax
__global__ void rangeMax(float *windark,int *rangemaxId,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	float tempf=0.0;
	int   tempi=0;
	for(int stride=blockDim.x*gridDim.x;stride>0;stride/=2)
	{
		__syncthreads();
		if(i<stride)
		{
			if(windark[i]>windark[i+stride])
			{
				windark[i]=windark[i];
				rangemaxId[i]=rangemaxId[i];
			}
			else
			{
				tempf=windark[i];
				windark[i]=windark[i+stride];
				windark[i+stride]=tempf;

				tempi=rangemaxId[i];
				rangemaxId[i]=rangemaxId[i+stride];
				rangemaxId[i+stride]=tempi;
			}
		}
	}
	
}

/*************************************************
Function: min
  Description: Take the minimum of the two input values
  Input: int num1          the input num 1
		 int num2          the input num 2
  Return: The minimum of the two input values
*************************************************/

__global__ void minRGB(float *d_R_In,float *d_G_In,float *d_B_In,float *C_min_RGB,int height, int width,int winsize)
{
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int j=threadIdx.y+blockIdx.y*blockDim.y;

	float temp;

	if(i<height&&j<width)
	{
		temp=d_R_In[i*width+j]<d_G_In[i*width+j]?d_R_In[i*width+j]:d_G_In[i*width+j];
		temp=temp<d_B_In[i*width+j]?temp:d_B_In[i*width+j];
		C_min_RGB[i*width+j]=temp;
	}
}
__global__ void minwCUDA(float *C_min_RGB,float *C_win_dark,int height,int width,int winsize)
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
				temp=temp<C_min_RGB[m*width+n]?temp:C_min_RGB[m*width+n];


		C_win_dark[i*width+j]=C_win_dark[i*width+j]<temp?C_win_dark[i*width+j]:temp;
	}

}
/*************************************************
Function: min C 
  Description: Take the minimum of the two input values
  Input: int num1          the input num 1
		 int num2          the input num 2
  Return: The minimum of the two input values
*************************************************/
void minwC(float *d_R_In,float *d_G_In,float *d_B_In,float *C_win_dark,int height, int width,int winsize)
{
	int indexi;
	int indexj;

	float temp;
	
	for(int i=winsize;i<height-winsize;i++)
		for(int j=winsize;j<width-winsize;j++)
		{
			temp=0.0;
			temp=d_R_In[i*width+j]<d_G_In[i*width+j]?d_R_In[i*width+j]:d_G_In[i*width+j];
			temp=temp<d_B_In[i*width+j]?temp:d_B_In[i*width+j];
			for(int indexi=0;indexi<15;indexi++)
			    for(int indexj=0;indexj<15;indexj++)
			    {
				     C_win_dark[(i-winsize+indexi)*width+(j-winsize+indexj)]=temp<C_win_dark[(i-winsize+indexi)*width+(j-winsize+indexj)]?temp:C_win_dark[(i-winsize+indexi)*width+(j-winsize+indexj)];
			    }
			
		}
}


__global__ void radi(float *C_radi_pro,unsigned int *C_ID,float *d_R_In,float *d_G_In,float *d_B_In,int range)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<range)
	{
        C_radi_pro[i] = d_R_In[C_ID[i]]+d_G_In[C_ID[i]]+d_B_In[C_ID[i]];
	}
}

__global__	void val1(float *C_val1,float *d_R_In,float *d_G_In,float *d_B_In,float atom,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
		C_val1[i]=(70.0/255.0)/abs(((d_R_In[i]+d_G_In[i]+d_B_In[i]) / 3.0) - atom);
	}
}

__global__ void alpha1(float *C_alpha,float *C_val1,float *q,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	float val2,val3,val4;
	
	if(i<height*width)
	{	
        val2 = C_val1[i]>1.0 ? C_val1[i]:1.0;
        val3 = q[i]>0.1 ? q[i] : 0.1;
        val4 = val2 * val3;
        C_alpha[i] = val4<1.0 ? val4 : 1.0;
	}
}

__global__ void Out(byte *d_R_Out,byte *d_G_Out,byte *d_B_Out,float *C_alpha,float *d_R_In,float *d_G_In,float *d_B_In,float atmo,int height,int width)
{
	int i=threadIdx.x+blockIdx.x*blockDim.x;

	if(i<height*width)
	{
        d_R_Out[i] = ( (d_R_In[i]-atmo)/ C_alpha[i] + atmo )*255;
        d_G_Out[i] = ( (d_G_In[i]-atmo)/ C_alpha[i] + atmo )*255;
        d_B_Out[i] = ( (d_B_In[i]-atmo)/ C_alpha[i] + atmo )*255;
	}
}


__global__
void user2Align_kernel( byte *B,byte *G,byte *R,unsigned int *RGBA, int width, int height, bool b2Int )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if ( x < width && y < height )
	{
		int index = y*width + x;

		if ( b2Int )
		{
			RGBA[index] = (B[index] << 16)
						| (G[index] << 8) | R[index] | 0xff000000;
		}
		else
		{
			unsigned int val = RGBA[index];
			R[index] = ( val ) & 0x000000ff;
			G[index] = ( val>> 8 ) & 0x000000ff;
			B[index] = ( val>> 16 ) & 0x000000ff;
		}
	}
}

void DarkChannelGPU::user2Align( byte *B,byte *G,byte *R,unsigned int *RGBA, int width, int height, bool b2Int )
{
	//int blocks=((height*width+511)/512);
    //seperate_buffer_kernel<<<blocks,512>>>(B, G, R, BGR, width, height);
	const dim3 block(32, 8);
    const dim3 grid( (width+block.x-1)/block.x, (height+block.y-1)/block.y );

	user2Align_kernel<<<grid,block>>>(B, G, R, RGBA, width, height, b2Int );
}

void DarkChannelGPU::Enhance(	byte *d_B_In_byte,	byte *d_G_In_byte,	byte *d_R_In_byte,
								byte *d_B_In_resized, byte *d_G_In_resized, byte *d_R_In_resized,			
								byte *d_B_Out,	byte *d_G_Out,	byte *d_R_Out,
								float scale )
{
	
	cudaError_t cudaStatus,err;
	//************division
#if ENABLE_TIMER
	timerCU	timerCPU;
	timerCPU.start();
#endif
  
	int sizeBlock1D = BLOCK_SIZE_1D;
	int blocks=((height_original*width_original+sizeBlock1D-1)/sizeBlock1D);
    division<<<blocks,sizeBlock1D>>>(d_R_In_original,d_G_In_original,d_B_In_original,
		d_R_In_byte,d_G_In_byte,d_B_In_byte,height_original,width_original);
	
	blocks=((height*width+sizeBlock1D-1)/sizeBlock1D);
    division<<<blocks,sizeBlock1D>>>(d_R_In,d_G_In,d_B_In,
		d_R_In_resized,d_G_In_resized,d_B_In_resized,height,width);
	
#if ENABLE_TIMER
	timerCPU.stop();
	cout << "division时间" << timerCPU.getTime() << "\n" << endl;
	timerCPU.start();
#endif
	////**********************DeviceToHost********************

	dim3 grid((height+BLOCK_SIZE_2D_X-1)/BLOCK_SIZE_2D_X,(width+BLOCK_SIZE_2D_Y-1)/BLOCK_SIZE_2D_Y);
	dim3 block(BLOCK_SIZE_2D_X,BLOCK_SIZE_2D_Y);

	minRGB<<<grid,block>>>(d_R_In,d_G_In,d_B_In,C_min_RGB,height,width,win_size);

	thrust::fill( 
		thrust::device_ptr<float>( C_win_dark ) ,
		thrust::device_ptr<float>( C_win_dark + width*height ) ,
		1.0f
		);

	minwCUDA<<<grid,block>>>(C_min_RGB,C_win_dark,height,width,win_size);

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "minw时间" << timerCPU.getTime() << "\n" << endl;
	
    //***********************winTT**************
	timerCPU.start();
#endif
  

    //选定精确dark value坐标
    winTT<<<blocks,sizeBlock1D>>>(C_win_dark,C_win_t,height,width);
	
#if ENABLE_TIMER
	timerCPU.stop();
	cout << "winTT时间" << timerCPU.getTime() << "\n" << endl;
	
	// *****************start********************
    //*************************N-boxfilter**********
	timerCPU.start();
#endif
	  

	thrust::fill( thrust::device_ptr<float>(C_ones), thrust::device_ptr<float>(C_ones + width*height), 1.0f);

	boxfilter(C_ones,imCum_C,N,r,height,width);

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "N-boxfilter时间" << timerCPU.getTime() << "\n" << endl;

	//************************mean_I_r -boxfilter****************
	timerCPU.start();   
#endif

	boxfilter(d_R_In,imCum_C,mean_I_r,r,height,width);
	boxfilter(d_G_In,imCum_C,mean_I_g,r,height,width);
	boxfilter(d_B_In,imCum_C,mean_I_b,r,height,width);
	
#if ENABLE_TIMER
	timerCPU.stop();
	cout << "mean_I_r_g_b-boxfilter时间" << timerCPU.getTime() << "\n" << endl;

	//**********************mean_p=boxfilter*************
	timerCPU.start();
#endif
    
	boxfilter(C_win_t,imCum_C,mean_p,r,height,width);

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "mean_p-boxfilter时间" << timerCPU.getTime() << "\n" << endl;

	//**************************mean_Ip_r=boxfilter**************
	timerCPU.start();  
#endif

	dotProduct<<<blocks,sizeBlock1D>>>(R_Pp,d_R_In,C_win_t,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(G_Pp,d_G_In,C_win_t,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(B_Pp,d_B_In,C_win_t,height,width);

	boxfilter(R_Pp,imCum_C,mean_Ip_r,r,height,width);
	boxfilter(G_Pp,imCum_C,mean_Ip_g,r,height,width);
	boxfilter(B_Pp,imCum_C,mean_Ip_b,r,height,width);

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "mean_Ip_r_g_b-boxfilter时间" << timerCPU.getTime() << "\n" << endl;

	//*************************cov_Ip_r*************************
	timerCPU.start();    
#endif

	cov_Ip<<<blocks,sizeBlock1D>>>(cov_Ip_r,mean_Ip_r,N,mean_I_r,mean_p,height,width);
	cov_Ip<<<blocks,sizeBlock1D>>>(cov_Ip_g,mean_Ip_g,N,mean_I_g,mean_p,height,width);
	cov_Ip<<<blocks,sizeBlock1D>>>(cov_Ip_b,mean_Ip_b,N,mean_I_b,mean_p,height,width);

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "cov_Ip_r_g_b-boxfilter时间" << timerCPU.getTime() << "\n" << endl;
	
	//***********************************var_I**************
	timerCPU.start();   
#endif

	dotProduct<<<blocks,sizeBlock1D>>>(RR_P,d_R_In,d_R_In,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(RG_P,d_R_In,d_G_In,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(RB_P,d_R_In,d_B_In,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(GG_P,d_G_In,d_G_In,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(GB_P,d_G_In,d_B_In,height,width);
	dotProduct<<<blocks,sizeBlock1D>>>(BB_P,d_B_In,d_B_In,height,width);


	boxfilter(RR_P,imCum_C,var_I_rr,r,height,width);
	boxfilter(RG_P,imCum_C,var_I_rg,r,height,width);
	boxfilter(RB_P,imCum_C,var_I_rb,r,height,width);
	boxfilter(GG_P,imCum_C,var_I_gg,r,height,width);
	boxfilter(GB_P,imCum_C,var_I_gb,r,height,width);
	boxfilter(BB_P,imCum_C,var_I_bb,r,height,width);

	var_I<<<blocks,sizeBlock1D>>>(var_I_rr,N,mean_I_r,mean_I_r,height,width);
	var_I<<<blocks,sizeBlock1D>>>(var_I_rg,N,mean_I_r,mean_I_g,height,width);
	var_I<<<blocks,sizeBlock1D>>>(var_I_rb,N,mean_I_r,mean_I_b,height,width);
	var_I<<<blocks,sizeBlock1D>>>(var_I_gg,N,mean_I_g,mean_I_g,height,width);
	var_I<<<blocks,sizeBlock1D>>>(var_I_gb,N,mean_I_g,mean_I_b,height,width);
	var_I<<<blocks,sizeBlock1D>>>(var_I_bb,N,mean_I_b,mean_I_b,height,width);
	
#if ENABLE_TIMER
	timerCPU.stop();
	cout << "var_I-boxfilter时间" << timerCPU.getTime() << "\n" << endl;
	
	//*********************************ar_g_b*********************
	timerCPU.start();
#endif

	ax<<<blocks,sizeBlock1D>>>(ar,ag,ab,var_I_rr,var_I_rg,var_I_rb,var_I_gg,var_I_gb,var_I_bb,cov_Ip_r,cov_Ip_g,cov_Ip_b,eps,height,width);
	
#if ENABLE_TIMER
	timerCPU.stop();
	cout << "ar_g_b时间" << timerCPU.getTime() << "\n" << endl;
	//**********************************q*********************************
	timerCPU.start();	
#endif

	meanb<<<blocks,sizeBlock1D>>>(mean_b,ar,ag,ab,mean_p,mean_I_r,mean_I_g,mean_I_b,N,height,width);

	boxfilter(ar,imCum_C,box_aI_r,r,height,width);
	boxfilter(ag,imCum_C,box_aI_g,r,height,width);
	boxfilter(ab,imCum_C,box_aI_b,r,height,width);
	boxfilter(mean_b,imCum_C,box_b,r,height,width);

   

	qbox<<<blocks,sizeBlock1D>>>(q,box_aI_r,box_aI_g,box_aI_b,d_R_In,d_G_In,d_B_In,box_b,N,height,width);
	

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "q时间" << timerCPU.getTime() << "\n" << endl;
	//***************************sort****************
	timerCPU.start();
#endif

	sort_by_key( C_win_dark, C_ID, height * width );

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "sort时间" << timerCPU.getTime() << "\n" << endl;

	//******************************去雾**********************
	timerCPU.start();
#endif

	int block1=(range+255)/256;
 

	radi<<<block1,256>>>(C_radi_pro,C_ID,d_R_In,d_G_In,d_B_In,range);
 

	thrust::sort(	thrust::device_ptr<float>(C_radi_pro), 
					thrust::device_ptr<float>(C_radi_pro + range), 
					thrust::greater<float>() ); // 从大到小
	float radi_pro;


	cudaMemcpy(&radi_pro,C_radi_pro,1*sizeof(float),cudaMemcpyDeviceToHost);
  

	float atmo;
    atmo = radi_pro/ 3.0 ; 
   

	val1<<<blocks,sizeBlock1D>>>(C_val1,d_R_In,d_G_In,d_B_In,atmo,height,width);


	alpha1<<<blocks,sizeBlock1D>>>(C_alpha,C_val1,q,height,width);

	// resize
	if( scale != 1 )
		;//BMPResizerGPU::resize( C_alpha, C_alpha_original, height, width, height_original, width_original );
	else
		cudaMemcpy( C_alpha_original, C_alpha, height * width * sizeof(float), cudaMemcpyDeviceToDevice );

	blocks=(( height_original * width_original + sizeBlock1D-1)/sizeBlock1D);

	Out<<<blocks,sizeBlock1D>>>(d_R_Out,d_G_Out,d_B_Out,
		C_alpha_original,
		d_R_In_original,d_G_In_original,d_B_In_original,
		atmo,height_original,width_original);
 

#if ENABLE_TIMER
	timerCPU.stop();
	cout << "去雾时间" << timerCPU.getTime() << "\n" << endl;
#endif
}


void DarkChannelGPU::Enhance(	unsigned int *d_BGR_In_byte, unsigned int *d_BGR_Out_byte )
{
	
	user2Align( d_B_In_byte,	d_G_In_byte,	d_R_In_byte, 
				d_BGR_In_byte, 
				width, height, 
				false );

	Enhance( d_B_In_byte, d_G_In_byte, d_R_In_byte,
			 d_B_In_byte, d_G_In_byte, d_R_In_byte,
			 d_B_Out_byte,d_G_Out_byte,d_R_Out_byte,
			 1.0f );

	user2Align( d_B_Out_byte,	d_G_Out_byte,	d_R_Out_byte, 
				d_BGR_Out_byte, 
				width, height, 
				true );
}

DarkChannelGPU::DarkChannelGPU( int width, int height, int width_original, int height_original )
{
	this->width = width;
	this->height = height;
	this->width_original = width_original;
	this->height_original = height_original;
	nBufferSizeFloat = width * height * sizeof(float);
	nBufferSizeFloatOriginal = width_original * height_original * sizeof(float);
	nBufferSizeByte = width * height * sizeof(byte);

	initialize();
}

DarkChannelGPU::DarkChannelGPU( int width, int height )
{
	this->width = width;
	this->height = height;
	this->width_original = width;
	this->height_original = height;
	nBufferSizeFloat = width * height * sizeof(float);
	nBufferSizeFloatOriginal = width_original * height_original * sizeof(float);
	nBufferSizeByte = width * height * sizeof(byte);

	initialize();
}

DarkChannelGPU::~DarkChannelGPU()
{
	release();
}

//void DarkChannelGPU::Enhance(float *B_P,float *G_P,float *R_P,
//					float *B_Out,float *G_Out,float *R_Out)
//{

//}


void DarkChannelGPU::initialize()
{
	win_size = 7;//窗口大小
	r=20;
	eps=0.001;

	img_size=height*width;
	range = ceil(img_size * 0.001);
 

	cudaMalloc((void**)&d_R_In,nBufferSizeFloat);
    cudaMalloc((void**)&d_G_In,nBufferSizeFloat);
    cudaMalloc((void**)&d_B_In,nBufferSizeFloat);
 
	cudaMalloc((void**)&d_R_In_original, nBufferSizeFloatOriginal);
    cudaMalloc((void**)&d_G_In_original, nBufferSizeFloatOriginal);
    cudaMalloc((void**)&d_B_In_original, nBufferSizeFloatOriginal);

	cudaMalloc((void**)&C_min_RGB,nBufferSizeFloat);
	cudaMalloc((void**)&C_win_dark,nBufferSizeFloat);
	cudaMalloc((void**)&C_win_t,nBufferSizeFloat);

	cudaMalloc((void**)&imCum_C,nBufferSizeFloat);
    cudaMalloc((void**)&C_ones,nBufferSizeFloat);
    cudaMalloc((void**)&N,nBufferSizeFloat);

	cudaMalloc((void**)&mean_I_r,nBufferSizeFloat);
    cudaMalloc((void**)&mean_I_g,nBufferSizeFloat);
    cudaMalloc((void**)&mean_I_b,nBufferSizeFloat);

	cudaMalloc((void**)&mean_p,nBufferSizeFloat);

	cudaMalloc((void**)&mean_Ip_r,nBufferSizeFloat);
    cudaMalloc((void**)&mean_Ip_g,nBufferSizeFloat);
    cudaMalloc((void**)&mean_Ip_b,nBufferSizeFloat);
    

	cudaMalloc((void**)&R_Pp,nBufferSizeFloat);
    cudaMalloc((void**)&G_Pp,nBufferSizeFloat);
    cudaMalloc((void**)&B_Pp,nBufferSizeFloat);
	cudaMalloc((void**)&cov_Ip_r,nBufferSizeFloat);
    cudaMalloc((void**)&cov_Ip_g,nBufferSizeFloat);
    cudaMalloc((void**)&cov_Ip_b,nBufferSizeFloat);

	cudaMalloc((void**)&RR_P,nBufferSizeFloat);
    cudaMalloc((void**)&RG_P,nBufferSizeFloat);
    cudaMalloc((void**)&RB_P,nBufferSizeFloat);
    cudaMalloc((void**)&GG_P,nBufferSizeFloat);
    cudaMalloc((void**)&GB_P,nBufferSizeFloat);
    cudaMalloc((void**)&BB_P,nBufferSizeFloat);

	cudaMalloc((void**)&var_I_rr,nBufferSizeFloat);
    cudaMalloc((void**)&var_I_rg,nBufferSizeFloat);
    cudaMalloc((void**)&var_I_rb,nBufferSizeFloat);
    cudaMalloc((void**)&var_I_gg,nBufferSizeFloat);
    cudaMalloc((void**)&var_I_gb,nBufferSizeFloat);
    cudaMalloc((void**)&var_I_bb,nBufferSizeFloat);

	cudaMalloc((void**)&ar,nBufferSizeFloat);
	cudaMalloc((void**)&ag,nBufferSizeFloat);
    cudaMalloc((void**)&ab,nBufferSizeFloat);  

	cudaMalloc((void**)&mean_b,nBufferSizeFloat);

	cudaMalloc((void**)&box_aI_r,nBufferSizeFloat);
    cudaMalloc((void**)&box_aI_g,nBufferSizeFloat);
    cudaMalloc((void**)&box_aI_b,nBufferSizeFloat);

    cudaMalloc((void**)&box_b,nBufferSizeFloat);
	cudaMalloc((void**)&q,nBufferSizeFloat);
 	
	cudaMalloc((void**)&C_radi_pro,range*sizeof(float));
    cudaMalloc((void**)&C_ID,img_size*sizeof(unsigned int));
	cudaMalloc((void**)&C_val1,nBufferSizeFloat);
	cudaMalloc((void**)&C_alpha,nBufferSizeFloat);  
	cudaMalloc((void**)&C_alpha_original, nBufferSizeFloatOriginal );

	cudaMalloc((void**)&d_R_In_byte, nBufferSizeByte );
	cudaMalloc((void**)&d_G_In_byte, nBufferSizeByte );
	cudaMalloc((void**)&d_B_In_byte, nBufferSizeByte );

	cudaMalloc((void**)&d_R_Out_byte, nBufferSizeByte );
	cudaMalloc((void**)&d_G_Out_byte, nBufferSizeByte );
	cudaMalloc((void**)&d_B_Out_byte, nBufferSizeByte );

}

void DarkChannelGPU::release()
{


	cudaFree(d_R_In);	cudaFree(d_G_In);	cudaFree(d_B_In);
	cudaFree(d_R_In_original);	cudaFree(d_G_In_original);	cudaFree(d_B_In_original);
	cudaFree(C_min_RGB);cudaFree(C_win_dark);cudaFree(C_win_t);
	cudaFree(imCum_C);  cudaFree(N);		cudaFree(C_ones);
	cudaFree(mean_I_r);cudaFree(mean_I_g);cudaFree(mean_I_b);
	cudaFree(mean_p);
	cudaFree(mean_Ip_r);cudaFree(mean_Ip_g);cudaFree(mean_Ip_b);
	cudaFree(R_Pp);cudaFree(G_Pp);cudaFree(B_Pp);
	cudaFree(cov_Ip_r);cudaFree(cov_Ip_g);cudaFree(cov_Ip_b);
	cudaFree(RR_P);cudaFree(RG_P);cudaFree(RB_P);
	cudaFree(GG_P);cudaFree(GB_P);cudaFree(BB_P);
	cudaFree(var_I_rr);cudaFree(var_I_rg);cudaFree(var_I_rb);cudaFree(var_I_gg);cudaFree(var_I_gb);cudaFree(var_I_bb);
	cudaFree(ar);cudaFree(ag);cudaFree(ab);
	cudaFree(mean_b);
	cudaFree(box_aI_r);cudaFree(box_aI_g);cudaFree(box_aI_b);
	cudaFree(box_b);
	cudaFree(q);
	cudaFree(C_radi_pro);cudaFree(C_ID);cudaFree(C_val1);cudaFree(C_alpha);
	cudaFree(C_alpha_original);

	cudaFree(d_R_In_byte);	cudaFree(d_G_In_byte);	cudaFree(d_B_In_byte);
	cudaFree(d_R_Out_byte);	cudaFree(d_G_Out_byte);	cudaFree(d_B_Out_byte);
}
