#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

// Kernel for grayscale conversion
void grayscale(ap_uint<8>* input_image, ap_uint<8>* output_image, const int width, const int height) 
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            ap_uint<8> r = input_image[idx];
            ap_uint<8> g = input_image[idx + 1];
            ap_uint<8> b = input_image[idx + 2];
            ap_uint<8> gray = static_cast<ap_uint<8>>(0.299f * r + 0.587f * g + 0.114f * b);
            output_image[y * width + x] = gray;
        }
    }
}

// Kernel for Laplacian filtering
void laplacian(ap_uint<8>* grayscale_output, ap_uint<8>* padded, ap_uint<8>* filtered_output, int width, int height) {
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
void sharpen(ap_uint<8>* original_image, ap_uint<8>* filtered_output, ap_uint<8>* sharpened_output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int sharpened_value = original_image[idx] + filtered_output[idx];
            sharpened_output[idx] = sharpened_value < 0 ? 0 : (sharpened_value > 255 ? 255 : sharpened_value);
        }
    }
}

