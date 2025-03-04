#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

// Kernel for grayscale conversion
void grayscale(
__global unsigned char* input_image, 
__global unsigned char* output_image, 
const int width, 
const int height) 

{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            unsigned char r = input_image[idx];
            unsigned char g = input_image[idx + 1];
            unsigned char b = input_image[idx + 2];
            unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            output_image[y * width + x] = gray;
        }
    }
}

// Kernel for Laplacian filtering
__kernel void laplacian(
__global const uchar* grayscale_output, 
__global uchar* padded, 
__global uchar* filtered_output, 
int width, 
int height
) 
{
    int padded_width = width + 2;
    int padded_height = height + 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            padded[(y + 1) * padded_width + (x + 1)] = grayscale_output[idx];
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                padded[y * padded_width + x] = 0;
            }
        }
    }

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            int laplacian[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};
            int filtered_value = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = padded[(y + ky + 1) * padded_width + (x + kx + 1)];
                    filtered_value += pixel * laplacian[ky + 1][kx + 1];
                }
            }
            filtered_output[idx] = filtered_value < 0 ? 0 : (filtered_value > 255 ? 255 : filtered_value);
        }
    }
}

// Kernel for sharpening
__kernel void sharpen(
__global uchar* original_image, 
__global uchar* filtered_output, 
__global uchar* sharpened_output, 
int width, 
int height
) 
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int sharpened_value = original_image[idx] + filtered_output[idx];
            sharpened_output[idx] = sharpened_value < 0 ? 0 : (sharpened_value > 255 ? 255 : sharpened_value);
        }
    }
}

