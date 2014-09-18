
	/**
	* @file DarkChannel.cuh          
	* @brief 暗通道去雾GPU版本               
	*                    
	* @author SoulSheng 2014-6-23     
	*/

#ifndef		DARK_CHANNEL_CUH_INCLUDE
#define		DARK_CHANNEL_CUH_INCLUDE


#include "commonDefinition.h"

/**       
*	DarkChannelGPU类，通过GPU实现去雾功能   
*/ 
class DarkChannelGPU
{
public:
	DarkChannelGPU( int width, int height, int width_original, int height_original );
	DarkChannelGPU( int width, int height );
	~DarkChannelGPU();

	/**       
	*	去雾处理函数 主入口函数
	*       
	* @param d_B_In, d_G_In, d_R_In		输入图像，RGB三个通道的显存地址      
	* @param d_B_Out,d_G_Out,d_R_Out	输出图像，RGB三个通道的显存地址     
	*       
	* @return  空
	*/ 
	void Enhance(	byte *d_B_In_byte,	byte *d_G_In_byte,	byte *d_R_In_byte,			
					byte *d_B_In_resized, byte *d_G_In_resized, byte *d_R_In_resized,			
					byte *d_B_Out,	byte *d_G_Out,	byte *d_R_Out,
					float scale = 1.0f );

	void Enhance(	unsigned int *d_BGR_In_byte, unsigned int *d_BGR_Out_byte );

protected:
	void initialize();
	void release();
	void cudaGenerateDrakKernel( );
	void sort_by_key( float* pKey, unsigned int* pValue, int size );
	void user2Align( byte *B,byte *G,byte *R,unsigned int *RGBA, int width, int height, bool b2Int );

private:
	float *B_P, *G_P, *R_P;
	float *B_Out, *G_Out, *R_Out;
	int width, height;
	int width_original, height_original;
	int nBufferSizeByte;
	
private:
	int nBufferSizeFloat, nBufferSizeFloatOriginal;
	int win_size;//窗口大小
	int r;
	float eps;
	int range;
	int img_size;

private:
	float *d_R_In,*d_G_In,*d_B_In;
	float *d_R_In_original,*d_G_In_original,*d_B_In_original;
	float *C_min_RGB;
	float *C_win_dark;
    float *C_win_t;
  	float *imCum_C, *C_ones, *N;
  	float *mean_I_r,*mean_I_g,*mean_I_b;

 	float *mean_p;
	float *mean_Ip_r,*mean_Ip_g,*mean_Ip_b;
	float *R_Pp,*G_Pp,*B_Pp;
  	float *cov_Ip_r,*cov_Ip_g,*cov_Ip_b;
	float *RR_P,*RG_P,*RB_P,*GG_P,*GB_P,*BB_P;
 	float *var_I_rr,*var_I_rg,*var_I_rb,*var_I_gg,*var_I_gb,*var_I_bb;
  	float *ar,*ag,*ab;
	float *mean_b;
	float *box_aI_r,*box_aI_g,*box_aI_b,*box_b;
    float *q;
	float *C_radi_pro;
	unsigned int   *C_ID;
 	float *C_val1;
 	float *C_alpha;
 	float *C_alpha_original;

	byte *d_R_In_byte,*d_G_In_byte,*d_B_In_byte;
	byte *d_R_Out_byte,*d_G_Out_byte,*d_B_Out_byte;

};

#endif // DARK_CHANNEL_CUH_INCLUDE