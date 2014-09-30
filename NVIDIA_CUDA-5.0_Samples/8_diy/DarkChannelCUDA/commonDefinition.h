
#ifndef COMMONDEFINITION
#define COMMONDEFINITION

#include <stdio.h>
#include <math.h>

typedef struct target_position
{
	float *x;
	float *y;
}TARGETPOSITION;


typedef unsigned char byte;

#define PI 3.1415926

#define E 2.7182818

#define target_number 10


#define		ENABLE_VERIFY_VALUE	0	// 输出中间变量
#define		ENABLE_REFRESH_IMAGE_FILTER		1	//	粒子滤波是否更新图片背景
#endif