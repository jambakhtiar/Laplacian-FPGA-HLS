#include <iostream>
#include <vector>
#include <cstring>
#include "ap_int.h"

//#include "bmpfunction.cpp"

// Include kernel function declarations
void grayscale(ap_uint<8>* input_image, ap_uint<8>* output_image, const int width, const int height);
void laplacian(ap_uint<8>* grayscale_output, ap_uint<8>* padded, ap_uint<8>* filtered_output, int width, int height);
void sharpen(ap_uint<8>* original_image, ap_uint<8>* filtered_output, ap_uint<8>* sharpened_output, int width, int height);

// BMP handling functions
void readBMP(const char* filename, std::vector<ap_uint<8>>& data, int& width, int& height);
void writeBMP(const char* filename, const std::vector<ap_uint<8>>& data, int width, int height);
void writeBMPGray(const char* filename, const std::vector<ap_uint<8>>& data, int width, int height);

int main() {
    int width, height;
    std::vector<ap_uint<8>> input_image;
    readBMP("/home/jam/Downloads/Laplacian/src/rocks.bmp", input_image, width, height);

    std::vector<ap_uint<8>> output_image(width * height);
    std::vector<ap_uint<8>> padded((width + 2) * (height + 2), 0);
    std::vector<ap_uint<8>> filtered_output(width * height);
    std::vector<ap_uint<8>> sharpened_output(width * height);

    // Perform grayscale conversion
    grayscale(input_image.data(), output_image.data(), width, height);
    writeBMPGray("/home/jam/Downloads/Laplacian/src/grey.bmp", output_image, width, height);

    // Perform Laplacian filtering
    laplacian(output_image.data(), padded.data(), filtered_output.data(), width, height);
    writeBMPGray("/home/jam/Downloads/Laplacian/src/laplacian.bmp", filtered_output, width, height);

    // Perform sharpening
    sharpen(output_image.data(), filtered_output.data(), sharpened_output.data(), width, height);
    writeBMPGray("/home/jam/Downloads/Laplacian/src/sharp.bmp", sharpened_output, width, height);

    return 0;
}
