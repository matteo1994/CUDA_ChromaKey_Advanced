#pragma once

#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class ColorYCbCr;

class ColorRGB {
public:
	ColorRGB(unsigned char red, unsigned char green, unsigned char blue) : r{ red }, g{ green }, b{ blue } {}
	ColorRGB() : r{ 0 }, g{ 0 }, b{ 0 } {}

	float RGBtoYf(ColorYCbCr* );
	ColorYCbCr& RGBtoY();

	unsigned char r;
	unsigned char g;
	unsigned char b;

};

class ColorYCbCr {
public:
	ColorYCbCr(float y, char cb, char cr) : Y{ y }, Cb{ cb }, Cr{ cr } {}
	ColorYCbCr() :Y{ 0 }, Cb{ 0 }, Cr{ 0 } {}

	ColorRGB& YtoRGB();

	float Y;
	char Cb;
	char Cr;

};

class KeyColor {
public:
	KeyColor(ColorRGB color, float, float);

	ColorRGB cRGB;
	ColorYCbCr cYCbCr;

	char kg;
	float accept_angle_cos;
	float accept_angle_sin;
	unsigned char accept_angle_tg;
	unsigned char accept_angle_ctg;
	unsigned char one_over_kc;
	unsigned char kfgy_scale;
	float angle;
	float noise_level;
};

class ImageBMP {
public:
	ImageBMP(char * filename);

	ImageBMP& readFile();
	ImageBMP& writePixelMap(ColorRGB* map);
	ImageBMP& copyHeaderFrom(ImageBMP*);

	char* filename;
	int valid;
	int filesize;
	int datapos;
	int width;
	int height;
	FILE* file;
};

int get32bitFromFile(FILE* );

int closeCauseError();

