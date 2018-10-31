
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>
#include "Utility.h"
#include <time.h>
#include <algorithm>

#define K_Y 0
#define K_NOISE 1
#define K_CB 0
#define K_CR 1
#define K_KG 2
#define K_TG 3
#define K_CTG 4
#define K_OV 5
#define K_SC 6


const int N_FLOAT_INFO = 2;
const int N_CHAR_INFO = 7;

__constant__ float keyInfoFloat[N_FLOAT_INFO];
__constant__ char keyInfoChar[N_CHAR_INFO];



__device__ short clamp(short n) {
	n &= -(n >= 0);
	return n | ((255 - n) >> 15);
}

cudaError_t applyChromaKey(short*, short*, short*, ColorRGB*, KeyColor*, int, int, int, int);

__global__ void CK(short* fr, short* fg, short* fb, unsigned char* or, unsigned char* og, unsigned char* ob, int size_X, int size_Y) {

	
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	//unsigned int iy = blockIdx.y;

	unsigned int i = iy * size_X + ix;


	if (i < size_X * size_Y) {
		
		//Conversion from RGB color space to YCbCr color space

		//Fetch foreground
		unsigned char * ptr1 = (unsigned char*)&fr[i];
		unsigned char * ptr2 = (unsigned char*)&fg[i];
		unsigned char * ptr3 = (unsigned char*)&fb[i];

		float fy = 0.257*(*ptr1) + 0.504*(*ptr2) + 0.098*(*ptr3);
		char fCb = -0.148*(*ptr1) - 0.291*(*ptr2) + 0.439*(*ptr3);
		char fCr = 0.439*(*ptr1) - 0.368*(*ptr2) - 0.071*(*ptr3);

		//Fetch background
		ptr1++;
		ptr2++;
		ptr3++;

		float by = 0.257*(*ptr1) + 0.504*(*ptr2) + 0.098*(*ptr3);
		char bCb = -0.148*(*ptr1) - 0.291*(*ptr2) + 0.439*(*ptr3);
		char bCr = 0.439*(*ptr1) - 0.368*(*ptr2) - 0.071*(*ptr3);

		//Declaration of pixel output value in YCbCr
		float resY;
		char resCb;
		char resCr;

		
		int  kfg, kbg;
		int  alfa1, x1, y1;
		short tmp, tmp1;
		char x, z;
		
		//Conversion of Color space to XZ (custom and user defined)

		//X axis
		tmp = ((short)(fCb)* keyInfoChar[K_CB] +
			(short)(fCr)* keyInfoChar[K_CR]) >> 7;   // /128; 
		
		/*
		if (tmp > 127) tmp = 127;
		if (tmp < -128) tmp = -128;
		*/
		tmp = tmp > 127 ? 127 : tmp;
		tmp = tmp < -128 ? -128 : tmp;
		x = tmp;

		//Z axis
		tmp = ((short)(fCr)* keyInfoChar[K_CB] -
			(short)(fCb)* keyInfoChar[K_CR]) >> 7;   // /128; 

		/*
		if (tmp > 127) tmp = 127;
		if (tmp < -128) tmp = -128;
		*/
		tmp = tmp > 127 ? 127 : tmp;
		tmp = tmp < -128 ? -128 : tmp;
		z = tmp;

		//Project the pixel value vertically onto the threshold Z = Z'X = X * tg(alfa)
		
		tmp = ((short)(x)* (unsigned char)keyInfoChar[K_TG]) >> 4; // /0x10 hex -> /16; 
		//if (tmp > 127) tmp = 127;
		tmp = tmp > 127 ? 127 : tmp;

		//If the absolute value of Z is more than the projection the pixel is outside the threshold so it's pure foreground

		if (abs(z) > tmp) {
			// keep foreground Kfg = 0
			resY = fy;
			resCb = fCb;
			resCr = fCr;
			
			//Conversion of output pixel value back to RGB color space

			float tmp3 = 1.164*resY + 1.596*resCr;
			tmp3 = (tmp3 > 0) ? tmp3 : 0;
			or[i] = (tmp3 < 255) ? tmp3 : 255;

			//out_r[i] = (unsigned char) by;

			tmp3 = 1.164*resY - 0.813*resCr - 0.392*resCb;
			tmp3 = (tmp3 > 0) ? tmp3 : 0;
			og[i] = (tmp3 < 255) ? tmp3 : 255;

			//out_g[i] = (unsigned char)bCb;

			tmp3 = 1.164*resY + 2.017*resCb;
			tmp3 = (tmp3 > 0) ? tmp3 : 0;
			ob[i] = (tmp3 < 255) ? tmp3 : 255;

			//out_b[i] = (unsigned char) bCr;
		}
		
		else {

			// Compute Kfg (implicitly) and Kbg, suppress foreground in XZ coord according to Kfg 

			tmp = ((short)(z)* (unsigned char)keyInfoChar[K_CTG]) >> 4; // /0x10;
			tmp = tmp > 127 ? 127 : tmp;
			tmp = tmp < -128 ? -128 : tmp;
			x1 = abs(tmp);
			y1 = z;

			//Calculating Kfg as Kfg = X - abs(Z)/tg(alfa)
			tmp1 = x - x1;
			if (tmp1 < 0) tmp1 = 0;

			//Calculating Kbg as Kbg = X'/ (KCx) where X' = X - abs(Z)/tg(alfa) and KCx = 2 * sqrt(Cb^2 + Cr^2)
			kbg = (((unsigned char)(tmp1)*(unsigned short)(unsigned char)keyInfoChar[K_OV]) / 2);
			if (kbg < 0) kbg = 0;
			if (kbg > 255) kbg = 255;

			//Calculate luminance component opportunly scaled -> Y' = Y - (y * Kfg) where y = Y / (KCx / 2)
			tmp = ((unsigned short)(tmp1)*(unsigned char)keyInfoChar[K_SC]) >> 4;  // /0x10;
			if (tmp > 0xFF) tmp = 0xFF;
			tmp = fy - tmp;
			if (tmp < 0) tmp = 0;
			resY = tmp; //It's already converted back to Y

			// Convert suppressed foreground back to CbCr

			tmp = ((char)(x1)*(short)(keyInfoChar[K_CB]) -
				(char)(y1)*(short)(keyInfoChar[K_CR])) >> 7; // /128;
			if (tmp < -128) resCb = -128;
			else if (tmp > 127) resCb = 127;
			else resCb = tmp;

			tmp = ((char)(x1)*(short)(keyInfoChar[K_CR]) +
				(char)(y1)*(short)(keyInfoChar[K_CB])) >> 7; // /128;
			if (tmp < -128) resCr = -128;
			else if (tmp > 127) resCr = 127;
			else resCr = tmp;

			

			// Deal with noise. For now, a circle around the key color with
			//radius of noise_level treated as exact key color. Introduces
			//sharp transitions.
			//

			/*
			tmp = z * (short)(z)+(x - keyInfoChar[K_KG])*(short)(x - keyInfoChar[K_KG]);
			if (tmp > 0xFFFF) tmp = 0xFFFF;
			if (tmp < keyInfoFloat[K_NOISE] * keyInfoFloat[K_NOISE]) {
				// Uncomment this if you want total suppression within the noise circle
				//resY=resCb=resCr=0;
				kbg = 255;
			}*/

			
			
			// Add Kbg*background
			
			tmp = resY + ((unsigned short)(kbg) * by) / 256;
			resY = (tmp < 255) ? tmp : 255;
			tmp = resCb + ((unsigned short)(kbg) * bCb) / 256;
			if (tmp < -128) resCb = -128;
			else if (tmp > 127) resCb = 127;
			else resCb = tmp;
			tmp = resCr + ((unsigned short)(kbg) * bCr) / 256;
			if (tmp < -128) resCr = -128;
			else if (tmp > 127) resCr = 127;
			else resCr = tmp;
			


			//Convert output's components back to RGB color space

			float tmp3 = 1.164*resY + 1.596*resCr;
			tmp3 = (tmp3 > 0) ? tmp3 : 0;
			or[i] = (tmp3 < 255) ? tmp3 : 255;

			//out_r[i] = (unsigned char) by;

			tmp3 = 1.164*resY - 0.813*resCr - 0.392*resCb;
			tmp3 = (tmp3 > 0) ? tmp3 : 0;
			og[i] = (tmp3 < 255) ? tmp3 : 255;

			//out_g[i] = (unsigned char)bCb;

			tmp3 = 1.164*resY + 2.017*resCb;
			tmp3 = (tmp3 > 0) ? tmp3 : 0;
			ob[i] = (tmp3 < 255) ? tmp3 : 255;
	
			//out_b[i] = (unsigned char) bCr;

		}
		
	}
	
}

int main(int argc, char **argv)
{

	/*must be 11 command line arguments
	filenames fg, bg, out, r, g, b, angle, noise blocksizeX blocksizeY*/
	if (argc != 11)
	{
		printf("must be 10 command line arguments\n");
		printf(" fg bg out r g b angle noise blocksizeX blocksizeY");
		return(1);
	}

	//Load foreground and background images from files (BMP images in this situation)

	ImageBMP* fg = new ImageBMP(argv[1]);
	fg->readFile();

	fg->valid = (fg->valid == 1) ? 1 : closeCauseError();

	ImageBMP* bg = new ImageBMP(argv[2]);
	bg->readFile();

	bg->valid = (bg->valid == 1) ? 1 : closeCauseError();

	std::cout << "Opening " << fg->filename << " in lettura as foreground... " << fg->filesize << " " << fg->width << "x" << fg->height << " start: " << fg->datapos << std::endl;
	std::cout << "Opening " << bg->filename << " in lettura as background... " << bg->filesize << " " << bg->width << "x" << bg->height << " start: " << bg->datapos << std::endl;

	//Start to build a placeholder output file (a BMP image)

	ImageBMP* out = new ImageBMP(argv[3]);
	out->copyHeaderFrom(fg);

	out->valid = (out->valid == 1) ? 1 : closeCauseError();

	std::cout << out->filename << " setted as output file " << out->filesize << " " << out->width << "x" << out->height << " start: " << out->datapos << std::endl;

	
	//Start a timer to check how long does it take to boot the GPU-CPU connection
	const clock_t begin_time6 = clock();

	cudaFree(0);

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	//Print out the fist timer
	std::cout << "Initializing GPU: " << float(clock() - begin_time6) / CLOCKS_PER_SEC << " sec" << std::endl;
	

	//Fetch the Key color from command line

	ColorRGB kcRGB = ColorRGB((unsigned char)atoi(argv[4]), (unsigned char)atoi(argv[5]), (unsigned char)atoi(argv[6]));

	
	//Encapsulate all the info about the chroma key in a KeyColor Object
	KeyColor* kc = new KeyColor(kcRGB, (float)atof(argv[7]), (float)atof(argv[8]));

	/*
	ColorRGB* fgRGB = new ColorRGB[fg->width * fg->height]; //Foreground matrix
	ColorRGB* bgRGB = new ColorRGB[bg->width * bg->height]; //Background matrix
	*/
	ColorRGB* outRGB = new ColorRGB[out->width * out->height]; //Setting up Memory for output matrix

	int size = fg->width * fg->height;

	short* fgbgR = new short[size];
	short* fgbgG = new short[size];
	short* fgbgB = new short[size];
	

	int pos = fg->datapos;
	int mx = 3 * fg->width;
	switch (mx % 4)
	{
	case 1:
		mx = mx + 3;
		break;
	case 2:
		mx = mx + 2;
		break;
	case 3:
		mx = mx + 1;
		break;
	}



	// Fill matrix for Foreground and Background


	int i, j;
	for (i = 0; i < fg->height; i++) {
		fseek(fg->file, pos, SEEK_SET);
		fseek(bg->file, pos, SEEK_SET);

		for (j = pos; j < (fg->width + pos); j++) {

			char b = getc(fg->file);
			char g = getc(fg->file);
			char r = getc(fg->file);
			char b_bg = getc(bg->file);
			char g_bg = getc(bg->file);
			char r_bg = getc(bg->file);

			char * tmp = (char*)&fgbgR[i * fg->width + (j - pos)];
			*tmp = r;
			tmp++;
			*tmp = r_bg;

			tmp = (char*)&fgbgG[i * fg->width + (j - pos)];
			*tmp = g;
			tmp++;
			*tmp = g_bg;

			tmp = (char*)&fgbgB[i * fg->width + (j - pos)];
			*tmp = b;
			tmp++;
			*tmp = b_bg;

			/*
			fgRGB[i * fg->width + (j - pos)] = ColorRGB(r, g, b);
			bgRGB[i * fg->width + (j - pos)] = ColorRGB(r_bg, g_bg, b_bg);
			*/
			//printf("Read pixel %d,%d: (%d,%d,%d)\n", (j - pos), i, r,g,b);
		}
		pos = mx + pos;

	}

    // Add vectors in parallel.
	//cudaError_t cudaStatus = applyChromaKey(fgRGB, bgRGB, outRGB, kc, fg->width, fg->height);



	
	
	/*
	unsigned char* r = new unsigned char[size];
	unsigned char* g = new unsigned char[size];
	unsigned char* b = new unsigned char[size];
	
	
	float* y = new float[size];
	char* cB = new char[size];
	char* cR = new char[size];
	
	for (int i = 0; i < fg->width * fg->height; i++) {
		
		r[i] = fgRGB[i].r;
		g[i] = fgRGB[i].g;
		b[i] = fgRGB[i].b;
		

		y[i] = fgRGB[i].RGBtoY().Y;
		cB[i] = fgRGB[i].RGBtoY().Cb;
		cR[i] = fgRGB[i].RGBtoY().Cr;


	}

	
	unsigned char* bgr = new unsigned char[size];
	unsigned char* bgg = new unsigned char[size];
	unsigned char* bgb = new unsigned char[size];
	

	float* by = new float[size];
	char* bcB = new char[size];
	char* bcR = new char[size];

	for (int i = 0; i < size; i++) {
	
		bgr[i] = bgRGB[i].r;
		bgg[i] = bgRGB[i].g;
		bgb[i] = bgRGB[i].b;
		

		by[i] = bgRGB[i].RGBtoY().Y;
		bcB[i] = bgRGB[i].RGBtoY().Cb;
		bcR[i] = bgRGB[i].RGBtoY().Cr;
	}

	*/

	//cudaError_t cudaStatus = applyChar(y,cB,cR, by, bcB, bcR, kc, outRGB, fg->width, fg->height);

	const clock_t begin_time7 = clock();

	cudaStatus = applyChromaKey(fgbgR, fgbgG, fgbgB, outRGB, kc, fg->width, fg->height, atoi(argv[9]), atoi(argv[10]));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	std::cout << "Total CK time: " << float(clock() - begin_time7) / CLOCKS_PER_SEC << " sec" << std::endl;
	
	out->writePixelMap(outRGB);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	/**/


	//system("PAUSE");
    return 0;
}

cudaError_t applyChromaKey(short* r, short* g, short* b, ColorRGB* out, KeyColor* key, int X, int Y, int dimX, int dimY) {

	int size = X * Y;

	cudaError_t cudaStatus;

	short* dev_r;
	short* dev_g;
	short* dev_b;

	/*

	unsigned char* dev_r;
	unsigned char* dev_g;
	unsigned char* dev_b;

	unsigned char* dev_br;
	unsigned char* dev_bg;
	unsigned char* dev_bb;
	*/
	
	
	unsigned char* dev_out_r;
	unsigned char* dev_out_g;
	unsigned char* dev_out_b;
	
	/*

	const clock_t begin_time6 = clock();

	cudaFree(0);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	std::cout << "Initializing GPU: " << float(clock() - begin_time6) / CLOCKS_PER_SEC << " sec" << std::endl;

	*/

	const clock_t begin_time = clock();

	cudaStatus = cudaMalloc((void**)&dev_r, size * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for output on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_g, size * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for foreground on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for background on GPU...\n");
		return cudaStatus;
	}


	/*
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_r, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for output on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_g, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for foreground on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for background on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_br, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for output on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_bg, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for foreground on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_bb, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for background on GPU...\n");
		return cudaStatus;
	}

	*/
	cudaStatus = cudaMalloc((void**)&dev_out_r, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for background on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_out_g, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for background on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_out_b, size * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Creating memory for background on GPU...\n");
		return cudaStatus;
	}
	

	//std::cout << "Creating memory on GPU: " << float(clock() - begin_time) / CLOCKS_PER_SEC << " sec" << std::endl;

	/*
	const clock_t begin_time4 = clock();
	
	short* r = new short[size];
	short* g = new short[size];
	short* b = new short[size];

	unsigned char* tmp;

	for (int i = 0; i < size; i++) {

		tmp = (unsigned char*)&r[i];
		*tmp = fg[i].r;
		tmp++;
		*tmp = bg[i].r;
		tmp = (unsigned char*)&g[i];
		*tmp = fg[i].g;
		tmp++;
		*tmp = bg[i].g;
		tmp = (unsigned char*)&b[i];
		*tmp = fg[i].b;
		tmp++;
		*tmp = bg[i].b;
	}

	unsigned char* r = new unsigned char[size];
	unsigned char* g = new unsigned char[size];
	unsigned char* b = new unsigned char[size];

	for (int i = 0; i < size; i++) {
		
		r[i] = fg[i].r;
		g[i] = fg[i].g;
		b[i] = fg[i].b;
	
	}
	
	unsigned char* bgr = new unsigned char[size];
	unsigned char* bgg = new unsigned char[size];
	unsigned char* bgb = new unsigned char[size];
	

	for (int i = 0; i < size; i++) {
		
		bgr[i] = bg[i].r;
		bgg[i] = bg[i].g;
		bgb[i] = bg[i].b;

	}
	

	std::cout << "Fetching RGB values: " << float(clock() - begin_time4) / CLOCKS_PER_SEC << " sec" << std::endl;
	*/

	const clock_t begin_time5 = clock();

	float* kIFloat = new float[N_FLOAT_INFO];
	char* kIChar = new char[N_CHAR_INFO];

	kIFloat[K_Y] = key->cYCbCr.Y;
	kIFloat[K_NOISE] = key->noise_level;
	kIChar[K_CB] = key->cYCbCr.Cb;
	kIChar[K_CR] = key->cYCbCr.Cr;
	kIChar[K_TG] = (char)key->accept_angle_tg;
	kIChar[K_CTG] = (char)key->accept_angle_ctg;
	kIChar[K_KG] = key->kg;
	kIChar[K_OV] = (char)key->one_over_kc;
	kIChar[K_SC] = (char)key->kfgy_scale;

	cudaStatus = cudaMemcpy(dev_r, r, size * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_g, g, size * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpyToSymbol(keyInfoFloat, kIFloat, N_FLOAT_INFO * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpyToSymbol(keyInfoChar, kIChar, N_CHAR_INFO * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}


	/*
	cudaStatus = cudaMemcpy(dev_br, bgr, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_bg, bgg, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_bb, bgb, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! Coping foreground matrix on GPU...\n");
		return cudaStatus;
	}
	*/

	//std::cout << "Copy data in GPU memory: " << float(clock() - begin_time5) / CLOCKS_PER_SEC << " sec" << std::endl;

	const clock_t begin_time2 = clock();

	/*
	dim3 block(blockSize);
	dim3 grid((size + block.x - 1) / block.x);
	*/

	int dimx = dimX;
	int dimy = dimY;
	
	dim3 block(dimx, dimy);
	dim3 grid((X + block.x - 1) / block.x, (Y + block.y - 1) / block.y);
	/*
	dim3 block(dimx);
	dim3 grid((X + block.x - 1) / block.x, Y);
	*/

	printf("Execution configuration <<<(%d, %d), (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);
	
	CK << <grid, block >> >(dev_r, dev_g, dev_b, dev_out_r, dev_out_g, dev_out_b, X, Y);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CK per Pixel launch failed: %s\n", cudaGetErrorString(cudaStatus));

		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CK per Pixel!\n", cudaStatus);

		return cudaStatus;
	}

	//std::cout << "Kernel's execution: " << float(clock() - begin_time2) / CLOCKS_PER_SEC << " sec" << std::endl;

	const clock_t begin_time3 = clock();

	/*
	unsigned char* out_r = new unsigned char[size];
	unsigned char* out_g = new unsigned char[size];
	unsigned char* out_b = new unsigned char[size];

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(out_r, dev_br, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(out_g, dev_bg, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(out_b, dev_bb, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

		return cudaStatus;
	}

	*/

	unsigned char* out_r = new unsigned char[size];
	unsigned char* out_g = new unsigned char[size];
	unsigned char* out_b = new unsigned char[size];

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(out_r, dev_out_r, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(out_g, dev_out_g, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(out_b, dev_out_b, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

		return cudaStatus;
	}

	for (int i = 0; i < size; i++) {

		out[i] = ColorRGB((unsigned char)out_r[i], (unsigned char)out_g[i], (unsigned char)out_b[i]);
		//outRGB[i] = ColorRGB(out_r[i], out_g[i], out_b[i]);
	}

	//std::cout << "Copy back results on main memory: " << float(clock() - begin_time3) / CLOCKS_PER_SEC << " sec" << std::endl;

	cudaFree(dev_r);
	cudaFree(dev_b);
	cudaFree(dev_g);
	/*
	cudaFree(dev_br);
	cudaFree(dev_bb);
	cudaFree(dev_bg);
	*/
	cudaFree(dev_out_r);
	cudaFree(dev_out_b);
	cudaFree(dev_out_g);

	return cudaStatus;
}