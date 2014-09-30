

#ifndef		DARK_CHANNEL_H_INCLUDE
#define		DARK_CHANNEL_H_INCLUDE

#include "commonDefinition.h"

// 图像去雾处理
void DarkChannel(byte* B_In, byte* G_In, byte* R_In,
	byte *B_In_resized, byte *G_In_resized, byte *R_In_resized,
	byte* B_Out, byte* G_Out, byte* R_Out, 
	int width_original, int height_original, float scale = 1.0f );

// 图像去雾处理
void DarkChannel(unsigned int* RGBA_In, unsigned int* RGBA_Out, int width, int height );

#endif //	DARK_CHANNEL_H_INCLUDE