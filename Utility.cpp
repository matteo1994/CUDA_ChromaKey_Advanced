#include <stdio.h>
#include <math.h>
#include "Utility.h"
#include <iostream>

const double pi() { return acos(-1); }


float ColorRGB::RGBtoYf(ColorYCbCr* cY) {
	float tmp, tmp1, tmp2;
	
	cY->Y = 0.257*this->r + 0.504*this->g + 0.098*this->b;
	tmp1 = -0.148*this->r - 0.291*this->g + 0.439*this->b;
	tmp2 = 0.439*this->r - 0.368*this->g - 0.071*this->b;
	tmp = sqrt(tmp1*tmp1 + tmp2 * tmp2);
	cY->Cb = 127 * (tmp1 / tmp);
	cY->Cr = 127 * (tmp2 / tmp);
	
	return tmp;
}

ColorYCbCr& ColorRGB::RGBtoY() {
	ColorYCbCr* cY = new ColorYCbCr();

	cY->Y = 0.257*this->r + 0.504*this->g + 0.098*this->b;
	cY->Cb = -0.148*this->r - 0.291*this->g + 0.439*this->b;
	cY->Cr = 0.439*this->r - 0.368*this->g - 0.071*this->b;

	return *cY;
}

ColorRGB& ColorYCbCr::YtoRGB() {

	ColorRGB* cRGB = new ColorRGB();

	float tmp = 1.164*this->Y + 1.596*this->Cr;
	tmp = (tmp > 0) ? tmp : 0;
	cRGB->r = (tmp < 255) ? tmp : 255;

	tmp = 1.164*this->Y - 0.813*this->Cr - 0.392*this->Cb;
	tmp = (tmp > 0) ? tmp : 0;
	cRGB->g = (tmp < 255) ? tmp : 255;

	tmp = 1.164*this->Y + 2.017*this->Cb;
	tmp = (tmp > 0) ? tmp : 0;
	cRGB->b = (tmp < 255) ? tmp : 255;

	return *cRGB;
}

KeyColor::KeyColor(ColorRGB color, float angle, float noise_level) {

	cRGB = ColorRGB(color.r, color.g, color.b);
	cYCbCr = ColorYCbCr();
	this->angle = angle;
	this->noise_level = noise_level;

	float kgl = cRGB.RGBtoYf(&cYCbCr);

	accept_angle_cos = cos(pi() * angle / 180);
	accept_angle_sin = sin(pi() * angle / 180);

	float tmp = 0xF * tan(pi() * angle / 180);
	if (tmp > 0xFF) tmp = 0xFF;

	accept_angle_tg = tmp;

	tmp = 0xF / tan(pi() * angle / 180);
	if (tmp > 0xFF) tmp = 0xFF;

	accept_angle_ctg = tmp;

	tmp = 1 / (kgl);
	this->one_over_kc = 0xFF * 2 * tmp - 0xFF;

	tmp = 0xF * (float)(cYCbCr.Y) / kgl;
	if (tmp > 0xFF) tmp = 0xFF;
	kfgy_scale = tmp;
	if (kgl > 127) kgl = 127;
	kg = kgl;


}

ImageBMP::ImageBMP(char * filename) {
	this->filename = filename;
	this->height = 0;
	this->width = 0;
	this->valid = 1;
	this->datapos = -1;
	this->file = nullptr;
}

int get32bitFromFile(FILE * f)
{
	/*reads a 32 bit int from file*/
	int a, b, c, d;
	a = getc(f);
	b = getc(f);
	c = getc(f);
	d = getc(f);
	/*printf("get32 %i %i %i %i
	", a, b, c, d);*/
	return (a + 256 * b + 65536 * c + 16777216 * d);
}

int closeCauseError() {
	exit(0); 
	return 0;
}

ImageBMP& ImageBMP::readFile() {
	errno_t err = fopen_s(&this->file, this->filename, "rb");

	if (err != 0) {
		perror("Error opening the file: ");
		//std::cout << "Errore";

		this->valid = 0;
		return *this;
	}
	
	//Read file's header

	/*check magic number*/
	char b = (char)getc(this->file);
	char m = (char)getc(this->file);

	if (b != 'B' || m != 'M')
	{
		this->valid = 0;
		return *this;
	}

	this->valid = 1;
	this->filesize = get32bitFromFile(this->file);

	/*skip past reserved section*/
	getc(this->file);
	getc(this->file);
	getc(this->file);
	getc(this->file);
	this->datapos = get32bitFromFile(this->file);

	/*get width and height from fixed positions*/
	fseek(this->file, 18, SEEK_SET);
	this->width = get32bitFromFile(this->file);
	this->height = get32bitFromFile(this->file);
	return *this;
}

ImageBMP& ImageBMP::copyHeaderFrom(ImageBMP* o) {
	errno_t err = fopen_s(&this->file, this->filename, "wb+");
	
	if (err != 0)
	{
		perror("Error opening the file: ");

		this->valid = 0;
		printf("\nExit\n");
		return *this;
	}

	err = fopen_s(&o->file, o->filename, "rb");

	if (err != 0)
	{
		perror("Error opening the file: ");

		o->valid = 0;
		printf("\nExit\n");
		return *this;
	}

	fseek(o->file, 0, SEEK_SET);
	
	for (int i = 0; i < o->datapos; i++) {
		fputc(getc(o->file), this->file);
	}
	for (int i = 0; i < o->filesize; i++) {
		fputc(0, this->file);
	}

	this->datapos = o->datapos;
	this->filesize = o->filesize;
	this->height = o->height;
	this->width = o->width;
	this->valid = 1;


	return *this;
}

ImageBMP& ImageBMP::writePixelMap(ColorRGB* map) {
	
	int pos = this->datapos;
	int mx = 3 * this->width;
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

	/* Fill matrix for Foreground and Background */
	int i, j;
	for (i = 0; i < this->height; i++) {
		
		fseek(this->file, pos, SEEK_SET);

		for (j = pos; j < (this->width + pos); j++) {

			fputc(map[i * this->width + (j - pos)].b, this->file);
			fputc(map[i * this->width + (j - pos)].g, this->file);
			fputc(map[i * this->width + (j - pos)].r, this->file);

		}
		pos = mx + pos;
	}

	
	return *this;
}


