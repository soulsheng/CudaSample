/*************************************************
  Copyright (C), 2014.
  File name:Drack.cu     
  Author:Li Zhe       Version:1.0        Date:2014.03.24 
  Description:
  Function List:  
*************************************************/

#include "DarkChannel.h"
//#include "timerCU.h"
//#include "bmpResizer.h"
#include "bmpHandler.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>

using namespace std;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

#define		ENABLE_RESIZE		0
#define		ENABLE_TIMER		0

#if ENABLE_TIMER
#include <helper_timer.h>
#endif

/*************************************************
  Function: rgbGray
  Description: Converting an RGB image into a grayscale image
  Input: float **rgb_gray  grayscale image
         float **R_P       R component of the original image
		 float **G_P       G component of the original image
		 float **B_P       B component of the original image
		 int width          width of image
		 int height         height of image
  Return: grayscale image of the original image
*************************************************/

void init2Array(float *w, int height, int width, float initValue)
{
    int i;
     
    for(i=0;i<height*width;i++)
    {
        w[i] = initValue;
    }
    
}



/*************************************************
Function: min
  Description: Take the minimum of the two input values
  Input: int num1          the input num 1
		 int num2          the input num 2
  Return: The minimum of the two input values
*************************************************/

float min2(float num1,float num2)
{
    float temp1 = num1;
    float temp2 = num2;
    
    if(temp1 < temp2)
        return temp1;
    else return temp2;
}


//求M行一列的数组 （reshape） 
void oneColumn(float *one_column,float *three_dimision,int height3,int width3,int column)
{
    int i,j;
    int k = 0;
    
    for(i=0;i<height3;i++)
    {
        for(j=0;j<width3;j++)
        {
            one_column[k++] = three_dimision[i*width3+j];
        }
    }    

}
//按列赋值
void oneColumn2(float *one_column,float *three_dimision,int height3,int width3,int column)
{
    int i,j;
    int k = 0;
    
    for(i=0;i<height3;i++)
    {
        for(j=0;j<width3;j++)
        {
            one_column[k++] = three_dimision[j*width3+i];
        }
    }  
}
//求3*3矩阵 
void threeDimision(float *threeDim,int threeDimWidth,float *rgb,int rgbWidth,int beginO,int endO,int beginT,int endT)
{
    int i,j;
    int num1=beginO;
    int num2=beginT;
              
    for(i=beginO;i<=endO;i++)
    {
        for(j=beginT;j<=endT;j++)
        {
            threeDim[(i-num1)*threeDimWidth+j-num2] = rgb[i*rgbWidth+j];
        }
    }
       
}

/***************************************************************/
//函数名称：InvMatrix
//函数功能：求矩阵A的逆矩阵
//输入参数：矩阵A的首地址，矩阵A的行数、列数，逆矩阵的首地址
//返回数值：成功求出逆矩阵返回0，否则返回-1
/***************************************************************/
void InvMatrix(float *InvMA,float *MA,int Row,int Col)
{
	float ratio=0;
	int i,j,k;
	
	float *tTempMA=(float *)malloc(Row*Col*sizeof(float));
	
	if(Row!=Col)	//只有方阵才可以求逆变换
	{	
		printf("不是方阵，不能求逆！");
		exit(0);
	}
	else
	{
		for(i=0;i<Row;i++)	
		{
			for(j=0;j<Col;j++)
			{
				tTempMA[i*Col+j]=MA[i*Col+j];
				InvMA[i*Col+j] = 0;
			}
			InvMA[i*Col+i] = 1;
		}

		//进行初等行变换
		for(k=0;k<Row;k++)	
		{
			for(i=0;i<Col;i++)
			{
				if(i!=k)
				{
					ratio=tTempMA[i*Col+k]/tTempMA[k*Col+k];
					for(j=0;j<Col;j++)
					{
						tTempMA[i*Col+j]-=tTempMA[k*Col+j]*ratio;
						InvMA[i*Col+j]-= InvMA[k*Col+j]*ratio;
					}
				}
			}
		}
		for(i=0;i<Row;i++)	//最终得出逆矩阵
		{
			for(j=0;j<Col;j++)
			{
				InvMA[i*Col+j]/=tTempMA[i*Col+i];
			}
		}
		
		
        free(tTempMA);
	}
}

//求矩阵转置 
void reverse(float *reverse,float *source,int height,int width) //子函数，把数组行与列调换     转置后的长宽
{
    int i,j;
    
    for(i=0;i<width;i++) //数组行与列调换
    {
        for(j=0;j<height;j++)
        {
            reverse[j*width+i]=source[i*height+j];
        }
    }
    
} 

//求矩阵相乘            
void multiply(float *array1,float *source1,float *source2,int height,int width,int dimision)
{
    int i,j,k;

    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
		{
			array1[i*width+j]=0.0;
		}
	}

    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            for(k=0;k<dimision;k++)
            {
                array1[i*width+j] += source1[i*dimision+k]*source2[k*width+j];
                
            }
        }
    }
    
}

//单位矩阵 
void unitMatrix(float *unit_matrix,int height,int width)
{
    int i,j;
    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            if(i!=j)
                unit_matrix[i*width+j] = 0; 
            else 
                unit_matrix[i*width+j] = 1;
        }     
    }   
    
}

void dividedByConst(float *array1, int height, int width, int num)
{
    int i,j;
    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            array1[i*width+j] = array1[i*width+j] / num;
        }
    }
    
}
        
//
void multipliedByConst(float *array1, int height, int width, float num1,int num2)
{
    int i,j;    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            array1[i*width+j] = (num1/num2) * array1[i*width+j];
        }
    }
}

//复制矩阵 (将1*width矩阵复制成height*width矩阵，每一列的元素都一样)
void repmat(float *arraySrc,float *arrayDest,int height,int width)
{
     int i,j;
     for(i=0;i<width;i++)
     {
         for(j=0;j<height;j++)
         {
             arrayDest[j*width+i] = arraySrc[i];
         }
     }
           
}

//复制矩阵 (将1*width矩阵复制成height*width矩阵，每一列的元素都一样)
void repmat2(float *arraySrc,float *arrayDest,int height,int width)
{
     int i,j;
     
     for(i=0;i<height;i++)
     {
         for(j=0;j<width;j++)
         {
             arrayDest[i*width+j] = arraySrc[i];
         }
     }
       
}
//boxfilter
/*%   BOXFILTER   O(1) time box filtering using cumulative sum
%
%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
%   - Running time independent of r; 
%   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
%   - But much faster.*/
void boxfilter(float *imSrc,float *imDst,int r,int height,int width)
{
	
	float *imCum=(float *)calloc(width*height,sizeof(float));

	//cumulative sum over Y axis
	//求矩阵imSrc各列的累加和
	for(int j=0;j<width;j++)
	{
		imCum[j]=imSrc[j];
	}
	for(int i=1;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			imCum[i*width+j]=imSrc[i*width+j]+imCum[(i-1)*width+j];
		}
	}
	//difference over Y axis
	for(int i=0;i<r+1;i++)
	{
		for(int j=0;j<width;j++)
		{
			imDst[i*width+j]=imCum[(i+r)*width+j];
		}
	}
	for(int i=r+1;i<height-r;i++)
	{
		for(int j=0;j<width;j++)
		{
			imDst[i*width+j]=imCum[(i+r)*width+j]-imCum[(i-r-1)*width+j];
		}
	}
	for(int i=height-r;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			imDst[i*width+j]=imCum[(height-1)*width+j]-imCum[(i-r-1)*width+j];
		}
	}
	//cumulative sum over X axis
	//求矩阵imDst各行的累加和
	for(int i=0;i<height;i++)
	{
		imCum[i*width]=imDst[i*width];
	}
	for(int j=1;j<width;j++)
	{
		for(int i=0;i<height;i++)
		{
			imCum[i*width+j]=imDst[i*width+j]+imCum[i*width+j-1];
		}
	}
	//difference over Y axis
	for(int j=0;j<r;j++)
	{
		for(int i=0;i<height;i++)
		{
			imDst[i*width+j]=imCum[i*width+j+r];
		}
	}
	for(int j=r+1;j<width-r;j++)
	{
		for(int i=0;i<height;i++)
		{
			imDst[i*width+j]=imCum[i*width+j+r]-imCum[i*width+j-r-1];
		}
	}
	for(int j=width-r;j<width;j++)
	{
		for(int i=0;i<height;i++)
		{
			imDst[i*width+j]=imCum[i*width+width-1]-imCum[i*width+j-r-1];
		}
	}

	free(imCum);

}

//矩阵点除
void dianchu(float *C,float *A,float *B,int height,int width)
{
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			C[i*width+j]=A[i*width+j]/B[i*width+j];
		}

}
//矩阵点乘
void diancheng(float *C,float *A,float *B,int height,int width)
{
	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			C[i*width+j]=A[i*width+j]*B[i*width+j];
		}

}
//矩阵元素除以255 
void division(float *R_P_Div, byte *byteIn, int height, int width)
{
    int i,j;
    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            R_P_Div[i*width+j] = byteIn[i*width+j] / 255.0f;
        }
    }
   
}

void DarkChannel(byte* B_In, byte* G_In, byte* R_In,
	byte *B_In_resized, byte *G_In_resized, byte *R_In_resized,
	byte* B_Out, byte* G_Out, byte* R_Out, 
	int width_original, int height_original, float scale)
{
	int i,j;
	int win_size = 7;//窗口大小
 
#if !ENABLE_RESIZE
	scale = 1;
#endif

	int width = (int)( width_original / scale );
	int height = (int)( height_original / scale );
    int height1 = 1; 
    int width1 = 1;
    
	float *R_P=(float *)malloc(height*width*sizeof(float));
	float *G_P=(float *)malloc(height*width*sizeof(float));
	float *B_P=(float *)malloc(height*width*sizeof(float));

#if !ENABLE_RESIZE
	division(R_P,R_In,height,width);
	division(G_P,G_In,height,width);
	division(B_P,B_In,height,width);
#else
	//************division
	timerCPU.start();
	division(R_P,R_In_resized,height,width);
    division(G_P,G_In_resized,height,width);
    division(B_P,B_In_resized,height,width);

	timerCPU.stop();
	cout << "division resized" << timerCPU.getTime() << "\n" << endl;

	float *R_P_original=(float *)malloc(height_original*width_original*sizeof(float));
	float *G_P_original=(float *)malloc(height_original*width_original*sizeof(float));
	float *B_P_original=(float *)malloc(height_original*width_original*sizeof(float));

	//************division
	timerCPU.start();
	division(R_P_original,R_In,height_original,width_original );
    division(G_P_original,G_In,height_original,width_original );
    division(B_P_original,B_In,height_original,width_original );

	timerCPU.stop();
	cout << "division original" << timerCPU.getTime() << "\n" << endl;
#endif
	//******************minwC****************************

#if ENABLE_TIMER	
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);

	sdkStartTimer(&timer);
#endif

    float *win_dark = (float *)malloc(height*width*sizeof(float *));
	
			
    init2Array(win_dark,height,width,1);//定义一个矩阵，有h行w列，里面数值全部为1
    
    int m,n,k;
    
    //计算分块darkchannel
    float m_pos_min;
    for(j=win_size;j<width-win_size;j++)
    {
        for(i=win_size;i<height-win_size;i++)
        {
            m_pos_min = min2(min2(R_P[i*width+j],G_P[i*width+j]),B_P[i*width+j]);//每个（i,j）像元三通道的最小值
            
            for(n=j-win_size;n<j+win_size+1;n++)//该像元的15邻域
            {
                for(m=i-win_size;m<i+win_size+1;m++)
                {
                    if(win_dark[m*width+n] > m_pos_min)
                    {
                        win_dark[m*width+n] = m_pos_min;
                    }
                }
            }
        }
    }

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time minw: %f (ms)\n", sdkGetTimerValue(&timer));

	 //***********************winTT**************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

    float *win_t = (float *)malloc(height*width*sizeof(float *));
	
    
    //选定精确dark value坐标
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            win_t[i*width+j] = 1-0.95 * win_dark[i*width+j];//得到的win_t还是160行，243列。
        }
    }

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time winTT: %f (ms)\n", sdkGetTimerValue(&timer));

	// *****************start********************
	 //*************************N-boxfilter**********
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

    int   r=20;
	float eps=0.001;
	float *imDst=(float *)calloc(width*height,sizeof(float));
	float *ones=(float *)calloc(width*height,sizeof(float));
	float *N=(float *)calloc(width*height,sizeof(float));

	for (int i = 0; i < width; i++)
	{
	    for (int j = 0; j < height; j++)
        {
             ones[j*width+i]=1.0;
        }
	} 
	boxfilter(ones,N,r,height,width);

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time N-boxfilter: %f (ms)\n", sdkGetTimerValue(&timer));

	//************************mean_I_r -boxfilter****************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	float *mean_I_r=(float *)calloc(width*height,sizeof(float));
	float *mean_I_g=(float *)calloc(width*height,sizeof(float));
	float *mean_I_b=(float *)calloc(width*height,sizeof(float));

	boxfilter(R_P,mean_I_r,r,height,width);
	boxfilter(G_P,mean_I_g,r,height,width);
	boxfilter(B_P,mean_I_b,r,height,width);
	
#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time mean_I_r_g_b-boxfilter: %f (ms)\n", sdkGetTimerValue(&timer));
	
	//**********************mean_p=boxfilter*************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	float *mean_p=(float *)calloc(width*height,sizeof(float));

	boxfilter(win_t,mean_p,r,height,width);

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time mean_p--boxfilter: %f (ms)\n", sdkGetTimerValue(&timer));
	
	//**************************mean_Ip_r=boxfilter**************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	float *mean_Ip_r=(float *)calloc(width*height,sizeof(float));
	float *mean_Ip_g=(float *)calloc(width*height,sizeof(float));
	float *mean_Ip_b=(float *)calloc(width*height,sizeof(float));
	float *R_Pp=(float *)calloc(width*height,sizeof(float));
	float *G_Pp=(float *)calloc(width*height,sizeof(float));
	float *B_Pp=(float *)calloc(width*height,sizeof(float));
	
	diancheng(R_Pp,R_P,win_t,height,width);
	diancheng(G_Pp,G_P,win_t,height,width);
	diancheng(B_Pp,B_P,win_t,height,width);

	boxfilter(R_Pp,mean_Ip_r,r,height,width);
	boxfilter(G_Pp,mean_Ip_g,r,height,width);
	boxfilter(B_Pp,mean_Ip_b,r,height,width);

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time mean_Ip_r_g_b-boxfilter: %f (ms)\n", sdkGetTimerValue(&timer));
	
	//*************************cov_Ip_r*************************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	float *cov_Ip_r=(float *)calloc(width*height,sizeof(float));
	float *cov_Ip_g=(float *)calloc(width*height,sizeof(float));
	float *cov_Ip_b=(float *)calloc(width*height,sizeof(float));
	float *mean_Imp_r=(float *)calloc(width*height,sizeof(float));
	float *mean_Imp_g=(float *)calloc(width*height,sizeof(float));
	float *mean_Imp_b=(float *)calloc(width*height,sizeof(float));

	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			cov_Ip_r[i*width+j]=mean_Ip_r[i*width+j]/N[i*width+j]-(mean_I_r[i*width+j]*mean_p[i*width+j])/(N[i*width+j]*N[i*width+j]);
			cov_Ip_g[i*width+j]=mean_Ip_g[i*width+j]/N[i*width+j]-(mean_I_g[i*width+j]*mean_p[i*width+j])/(N[i*width+j]*N[i*width+j]);
			cov_Ip_b[i*width+j]=mean_Ip_b[i*width+j]/N[i*width+j]-(mean_I_b[i*width+j]*mean_p[i*width+j])/(N[i*width+j]*N[i*width+j]);
		}

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time cov_Ip_r_g_b-boxfilter: %f (ms)\n", sdkGetTimerValue(&timer));
		
	//***********************************var_I**************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	float *var_I_rr=(float *)calloc(width*height,sizeof(float));
	float *var_I_rg=(float *)calloc(width*height,sizeof(float));
	float *var_I_rb=(float *)calloc(width*height,sizeof(float));
	float *var_I_gg=(float *)calloc(width*height,sizeof(float));
	float *var_I_gb=(float *)calloc(width*height,sizeof(float));
	float *var_I_bb=(float *)calloc(width*height,sizeof(float));
	float *RR_P=(float *)calloc(width*height,sizeof(float));
	float *RG_P=(float *)calloc(width*height,sizeof(float));
	float *RB_P=(float *)calloc(width*height,sizeof(float));
	float *GG_P=(float *)calloc(width*height,sizeof(float));
	float *GB_P=(float *)calloc(width*height,sizeof(float));
	float *BB_P=(float *)calloc(width*height,sizeof(float));

	diancheng(RR_P,R_P,R_P,height,width);
	diancheng(RG_P,R_P,G_P,height,width);
	diancheng(RB_P,R_P,B_P,height,width);
	diancheng(GG_P,G_P,G_P,height,width);
	diancheng(GB_P,G_P,B_P,height,width);
	diancheng(BB_P,B_P,B_P,height,width);

	boxfilter(RR_P,var_I_rr,r,height,width);
	boxfilter(RG_P,var_I_rg,r,height,width);
	boxfilter(RB_P,var_I_rb,r,height,width);
	boxfilter(GG_P,var_I_gg,r,height,width);
	boxfilter(GB_P,var_I_gb,r,height,width);
	boxfilter(BB_P,var_I_bb,r,height,width);

	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			var_I_rr[i*width+j]=var_I_rr[i*width+j]/N[i*width+j]-(mean_I_r[i*width+j]*mean_I_r[i*width+j])/(N[i*width+j]*N[i*width+j]);
			var_I_rg[i*width+j]=var_I_rg[i*width+j]/N[i*width+j]-(mean_I_r[i*width+j]*mean_I_g[i*width+j])/(N[i*width+j]*N[i*width+j]);
			var_I_rb[i*width+j]=var_I_rb[i*width+j]/N[i*width+j]-(mean_I_r[i*width+j]*mean_I_b[i*width+j])/(N[i*width+j]*N[i*width+j]);
			var_I_gg[i*width+j]=var_I_gg[i*width+j]/N[i*width+j]-(mean_I_g[i*width+j]*mean_I_g[i*width+j])/(N[i*width+j]*N[i*width+j]);
			var_I_gb[i*width+j]=var_I_gb[i*width+j]/N[i*width+j]-(mean_I_g[i*width+j]*mean_I_b[i*width+j])/(N[i*width+j]*N[i*width+j]);
			var_I_bb[i*width+j]=var_I_bb[i*width+j]/N[i*width+j]-(mean_I_b[i*width+j]*mean_I_b[i*width+j])/(N[i*width+j]*N[i*width+j]);
		}
	
	
#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time var_I-boxfilter: %f (ms)\n", sdkGetTimerValue(&timer));
		
	//*********************************ar_g_b*********************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	float *ar=(float *)calloc(width*height,sizeof(float));
	float *ag=(float *)calloc(width*height,sizeof(float));
	float *ab=(float *)calloc(width*height,sizeof(float));

	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			float *Sigma=(float *)calloc(3*3,sizeof(float));
			Sigma[0]=var_I_rr[i*width+j]+eps;
			Sigma[1]=var_I_rg[i*width+j];
			Sigma[2]=var_I_rb[i*width+j];
			Sigma[3]=var_I_rg[i*width+j];
			Sigma[4]=var_I_gg[i*width+j]+eps;
			Sigma[5]=var_I_gb[i*width+j];
			Sigma[6]=var_I_rb[i*width+j];
			Sigma[7]=var_I_gb[i*width+j];
			Sigma[8]=var_I_bb[i*width+j]+eps;
			
			float *InvSigma=(float *)calloc(3*3,sizeof(float));
			InvMatrix(InvSigma,Sigma,3,3);
			ar[i*width+j]=cov_Ip_r[i*width+j]*InvSigma[0]+cov_Ip_g[i*width+j]*InvSigma[3]+cov_Ip_b[i*width+j]*InvSigma[6];
			ag[i*width+j]=cov_Ip_r[i*width+j]*InvSigma[1]+cov_Ip_g[i*width+j]*InvSigma[4]+cov_Ip_b[i*width+j]*InvSigma[7];
			ab[i*width+j]=cov_Ip_r[i*width+j]*InvSigma[2]+cov_Ip_g[i*width+j]*InvSigma[5]+cov_Ip_b[i*width+j]*InvSigma[8];
		}
	
	
#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time ar_g_b: %f (ms)\n", sdkGetTimerValue(&timer));
	
	//**********************************q*********************************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

    float *mean_Ia_r=(float *)calloc(width*height,sizeof(float));
	float *mean_Ia_g=(float *)calloc(width*height,sizeof(float));
	float *mean_Ia_b=(float *)calloc(width*height,sizeof(float));
	float *mean_b=(float *)calloc(width*height,sizeof(float));

	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			mean_b[i*width+j]=(mean_p[i*width+j]-ar[i*width+j]*mean_I_r[i*width+j]
			                               -ag[i*width+j]*mean_I_g[i*width+j]
										   -ab[i*width+j]*mean_I_b[i*width+j])/N[i*width+j];
		}
	
	float *q=(float *)calloc(width*height,sizeof(float));
	float *aI_r=(float *)calloc(width*height,sizeof(float));
	float *aI_g=(float *)calloc(width*height,sizeof(float));
	float *aI_b=(float *)calloc(width*height,sizeof(float));
	float *box_aI_r=(float *)calloc(width*height,sizeof(float));
	float *box_aI_g=(float *)calloc(width*height,sizeof(float));
	float *box_aI_b=(float *)calloc(width*height,sizeof(float));
	float *box_b=(float *)calloc(width*height,sizeof(float));

	boxfilter(ar,box_aI_r,r,height,width);
	boxfilter(ag,box_aI_g,r,height,width);
	boxfilter(ab,box_aI_b,r,height,width);
	boxfilter(mean_b,box_b,r,height,width);

	for(int i=0;i<height;i++)
		for(int j=0;j<width;j++)
		{
			q[i*width+j]=(box_aI_r[i*width+j]*R_P[i*width+j]
			            + box_aI_g[i*width+j]*G_P[i*width+j]
						+ box_aI_b[i*width+j]*B_P[i*width+j]
						+ box_b[i*width+j])/N[i*width+j];
		}
	

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time	q: %f (ms)\n", sdkGetTimerValue(&timer));

	//***************************sort****************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif
   
    
    int range = ceil(width * height * 0.001);
    float *radi_pro = (float*)calloc(range,sizeof(float)); //申请二维数组动态储存空间 
    //float **init2Array(float **w, int height, int width, int initValue)    
    float *a = (float*)calloc(width,sizeof(float)); //申请二维数组动态储存空间 height*width    
    float *b = (float*)calloc(width,sizeof(float)); //申请二维数组动态储存空间 height*width;
    
    float c = 0.0;
    int d = 0;
    int bb = 0;
    
    float *marray = (float*)malloc(sizeof(float)*height*width); //申请二维数组动态储存空间 height*width
    int s;
    for(s=0;s<range;s++)
    {
        for(j=0;j<width;j++)
        {
            for(i=0;i<height;i++)
            {
                if(win_dark[i*width+j]>a[j]) 
                {   // a存放最大值，b存放最大值所在行
                    a[j] = win_dark[i*width+j];
                    b[j] = i;
                }
            }
        }
        
        for(j=0;j<width;j++)
        {
            if(a[j]>c) 
            {
                //c存放最大值，d存放最大值所在行
                c = a[j];
                d = j;
            } 
        }
        
        bb = (int)b[d];
        
        init2Array(marray,height,width,0);
        marray[bb*width+d] = 1;
        
		for(i=0;i<height;i++)
		{
			for(j=0;j<width;j++)
			{
				win_dark[i*width+j] = win_dark[i*width+j] - c * marray[i*width+j];
			}
		}
        
        radi_pro[s] = R_P[bb*width+d]+G_P[bb*width+d]+B_P[bb*width+d];
    }
    
#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time sort: %f (ms)\n", sdkGetTimerValue(&timer));

	//******************************去雾**********************
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

    float atmo = 0.0;//Atmospheric optical
    float radi_pro_val = radi_pro[0];
    
    for(i=0;i<s;i++)
    {
        if(radi_pro[i]>radi_pro_val) radi_pro_val = radi_pro[i];
    }
    atmo = radi_pro_val / 3 ;   
    float *inten = (float*)calloc(height*width,sizeof(float)); //申请二维数组动态储存空间 height*width
		 
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            inten[i*width+j] = (R_P[i*width+j]+G_P[i*width+j]+B_P[i*width+j]) / 3;
        }
    }
		 
    float kk = 70.0;
        
	float *karray = (float*)malloc(sizeof(float)*height*width); //申请二维数组动态储存空间 height*width
            
    init2Array(karray,height,width,0);  
          
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            karray[i*width+j] = karray[i*width+j] + kk/255;
        }
    } 
    float *cha = (float*)malloc(sizeof(float)*height*width); //申请二维数组动态储存空间 height*width
				
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            cha[i*width+j] = fabs(inten[i*width+j] - atmo);
        }
    }
    float *alpha = (float*)calloc(height*width,sizeof(float));    		
    float val1,val2,val3,val4;
			
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            val1 = karray[i*width+j] / cha[i*width+j];
            val2 = val1>1 ? val1 : 1 ;
            val3 = (q[i*width+j]>0.1) ? q[i*width+j] : 0.1;
            val4 = val2 * val3;
            alpha[i*width+j] = val4<1 ? val4 : 1;
        }
    }
#if ENABLE_RESIZE   
	float *alpha_original = (float*)calloc(height_original*width_original,sizeof(float));
	BMPResizer::resize( alpha, alpha_original, height, width, height_original, width_original );
	height = height_original;
	width = width_original;
    //float *dehaze_R = (float*)malloc(sizeof(float)*height*width); //申请二维数组动态储存空间 height*width
    //float *dehaze_G = (float*)malloc(sizeof(float)*height*width); //申请二维数组动态储存空间 height*width    
    //float *dehaze_B = (float*)malloc(sizeof(float)*height*width); //申请二维数组动态储存空间 height*width

    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            R_Out[i*width+j] = ( (R_P_original[i*width+j]-atmo)/ alpha_original[i*width+j] + atmo )*255;
        }
    }
    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            G_Out[i*width+j] = ( (G_P_original[i*width+j]-atmo)/ alpha_original[i*width+j] + atmo )*255;
        }
    }
    
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
        {
            B_Out[i*width+j] = ( (B_P_original[i*width+j]-atmo)/ alpha_original[i*width+j] + atmo )*255;
        }
    }
#else
	for(i=0;i<height;i++)
	{
		for(j=0;j<width;j++)
		{
			R_Out[i*width+j] = ( (R_P[i*width+j]-atmo)/ alpha[i*width+j] + atmo )*255;
		}
	}

	for(i=0;i<height;i++)
	{
		for(j=0;j<width;j++)
		{
			G_Out[i*width+j] = ( (G_P[i*width+j]-atmo)/ alpha[i*width+j] + atmo )*255;
		}
	}

	for(i=0;i<height;i++)
	{
		for(j=0;j<width;j++)
		{
			B_Out[i*width+j] = ( (B_P[i*width+j]-atmo)/ alpha[i*width+j] + atmo )*255;
		}
	}
#endif

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time alpha: %f (ms)\n", sdkGetTimerValue(&timer));

	sdkDeleteTimer(&timer);
#endif

	///***************************Binary image**********************************/

	//fp=fopen("E:\\vido\\zengqiang\\zengqiang\\3_B.txt", "w+");
	//	
 //   if (!fp)
 //   {
 //       perror("cannot open file");
 //   }
 // 
 //   for (i = 0; i < height; i++)
	//{
	//    for (j = 0; j <width; j++)
 //       {
 //            fprintf(fp,"%f",B_P[i*width+j]);
	//		 fputc(' ',fp);
 //       }
 //       fputc('\n',fp);
	//} 
 //   fclose(fp);
 //   printf("写入完毕\n");
 //   
 //   fp=fopen("E:\\vido\\zengqiang\\zengqiang\\3_G.txt", "w+");
	//	
 //   if (!fp)
 //   {
 //       perror("cannot open file");
 //   }
 // 
 //   for (i = 0; i < height; i++)
	//{
	//    for (j = 0; j <width; j++)
 //       {
 //            fprintf(fp,"%f",G_P[i*width+j]);
	//		 fputc(' ',fp);
 //       }
 //       fputc('\n',fp);
	//} 
 //   fclose(fp);
 //   printf("写入完毕\n");
    
    /*fp=fopen("E:\\vido\\zengqiang\\zengqiang\\3_R.txt", "w+");
		
    if (!fp)
    {
        perror("cannot open file");
    }
  
    for (i = 0; i < height; i++)
	{
	    for (j = 0; j <width; j++)
        {
             fprintf(fp,"%f",R_P[i*width+j]);
			 fputc(' ',fp);
        }
        fputc('\n',fp);
	} 
    fclose(fp);
    printf("写入完毕\n");
    
    fp=fopen("E:\\vido\\zengqiang\\zengqiang\\win_drak.txt", "w+");
		
    if (!fp)
    {
        perror("cannot open file");
    }
  
    for (i = 0; i < height; i++)
	{
	    for (j = 0; j <width; j++)
        {
             fprintf(fp,"%f",win_dark[i*width+j]);
			 fputc(' ',fp);
        }
        fputc('\n',fp);
	} 
    fclose(fp);
    printf("写入完毕\n");*/
#if 0
    fp=fopen("E:\\vido\\zengqiang\\zengqiang\\dehaze_B.txt", "w+");
		
    if (!fp)
    {
        perror("cannot open file");
    }
  
    for (i = 0; i < height; i++)
	{
	    for (j = 0; j <width; j++)
        {
             fprintf(fp,"%f",dehaze_B[i*width+j]);
			 fputc(' ',fp);
        }
        fputc('\n',fp);
	} 
    fclose(fp);
    printf("写入完毕\n");
    
    fp=fopen("E:\\vido\\zengqiang\\zengqiang\\dehaze_G.txt", "w+");
		
    if (!fp)
    {
        perror("cannot open file");
    }
  
    for (i = 0; i < height; i++)
	{
	    for (j = 0; j <width; j++)
        {
             fprintf(fp,"%f",dehaze_G[i*width+j]);
			 fputc(' ',fp);
        }
        fputc('\n',fp);
	} 
    fclose(fp);
    printf("写入完毕\n");
    
    fp=fopen("E:\\vido\\zengqiang\\zengqiang\\dehaze_R.txt", "w+");
		
    if (!fp)
    {
        perror("cannot open file");
    }
  
    for (i = 0; i < height; i++)
	{
	    for (j = 0; j <width; j++)
        {
             fprintf(fp,"%f",dehaze_R[i*width+j]);
			 fputc(' ',fp);
        }
        fputc('\n',fp);
	} 
    fclose(fp);
    printf("写入完毕\n");
#endif
    // 释放临时内存空间
	free(win_dark);
	free(win_t);

	free( B_P );
	free( G_P );
	free( R_P );
#if ENABLE_RESIZE
	free( B_P_original );
	free( G_P_original );
	free( R_P_original );
#endif
}

void DarkChannel( unsigned int* RGBA_In, unsigned int* RGBA_Out, int width, int height )
{
	byte *B_In,*G_In,*R_In;//存储原始图像的垂直翻转，第一行跟最后一行交换像素值
	byte *B_Out, *G_Out, *R_Out;	// 存储结果图像像素值

	B_In=(byte *)malloc(height*width*sizeof(byte));
	G_In=(byte *)malloc(height*width*sizeof(byte));
	R_In=(byte *)malloc(height*width*sizeof(byte));
	B_Out=(byte *)malloc(height*width*sizeof(byte));
	G_Out=(byte *)malloc(height*width*sizeof(byte));
	R_Out=(byte *)malloc(height*width*sizeof(byte));

	BMPHandler::bgr2Int( RGBA_In, B_In, G_In, R_In, width, height, false );

	DarkChannel( B_In, G_In, R_In, NULL, NULL, NULL, B_Out, G_Out, R_Out, width, height, 1.0 );

	BMPHandler::bgr2Int( RGBA_Out, B_Out, G_Out, R_Out, width, height, true );

	free(R_In);		free(G_In);		free(B_In);
	free(B_Out);	free(G_Out);	free(R_Out);
}

