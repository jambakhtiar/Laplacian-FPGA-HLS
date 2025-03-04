#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include "ap_int.h"

using namespace std;

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};          // File type always BM which is 0x4D42
    uint32_t fileSize{0};               // Size of the file (in bytes)
    uint16_t reserved1{0};              // Reserved, always 0
    uint16_t reserved2{0};              // Reserved, always 0
    uint32_t offsetData{0};             // Start position of pixel data (bytes from the beginning of the file)
};

struct BMPInfoHeader {
    uint32_t size{0};                      // Size of this header (in bytes)
    int32_t width{0};                      // width of bitmap in pixels
    int32_t height{0};                     // width of bitmap in pixels
                                           //       (if positive, bottom-up, with origin in lower left corner)
                                           //       (if negative, top-down, with origin in upper left corner)
    uint16_t planes{1};                    // No. of planes for the target device, this is always 1
    uint16_t bitCount{0};                  // No. of bits per pixel
    uint32_t compression{0};               // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
    uint32_t sizeImage{0};                 // 0 - for uncompressed images
    int32_t xPixelsPerMeter{0};
    int32_t yPixelsPerMeter{0};
    uint32_t colorsUsed{0};                // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
    uint32_t colorsImportant{0};           // No. of colors used for displaying the bitmap. If 0 all colors are required
};
#pragma pack(pop)

void readBMP(const char* filename, vector<ap_uint<8>>& data, int& width, int& height) {
    ifstream inp{filename, ios_base::binary};
    if (inp) {
        BMPHeader header;
        BMPInfoHeader infoHeader;

        inp.read((char*)&header, sizeof(header));
        inp.read((char*)&infoHeader, sizeof(infoHeader));

        width = infoHeader.width;
        height = infoHeader.height;

        size_t rowStride = (infoHeader.width * 3 + 3) & ~3;
        vector<ap_uint<8>> tmpData(rowStride * infoHeader.height);

        inp.seekg(header.offsetData, inp.beg);
        inp.read((char*)tmpData.data(), tmpData.size());

        data.resize(infoHeader.width * infoHeader.height * 3);
        for (int y = 0; y < infoHeader.height; ++y) {
            for (int x = 0; x < infoHeader.width * 3; ++x) {
                data[(y * infoHeader.width * 3) + x] = tmpData[(y * rowStride) + x];
            }
        }
    } else {
        cerr << "Error opening BMP file." << endl;
    }
}

void writeBMP(const char* filename, const vector<ap_uint<8>>& data, int width, int height) {
    ofstream out{filename, ios_base::binary};
    if (out) {
        BMPHeader header;
        BMPInfoHeader infoHeader;

        size_t rowStride = (width * 3 + 3) & ~3;

        header.fileSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + rowStride * height;
        header.offsetData = sizeof(BMPHeader) + sizeof(BMPInfoHeader);

        infoHeader.size = sizeof(BMPInfoHeader);
        infoHeader.width = width;
        infoHeader.height = height;
        infoHeader.bitCount = 24;
        infoHeader.sizeImage = rowStride * height;

        out.write((const char*)&header, sizeof(header));
        out.write((const char*)&infoHeader, sizeof(infoHeader));

        vector<ap_uint<8>> tmpData(rowStride * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width * 3; ++x) {
                tmpData[(y * rowStride) + x] = data[(y * width * 3) + x];
            }
        }

        out.write((const char*)tmpData.data(), tmpData.size());
    } else {
        cerr << "Error writing BMP file." << endl;
    }
}

void writeBMPGray(const char* filename, const vector<ap_uint<8>>& data, int width, int height) {
    ofstream out{filename, ios_base::binary};
    if (out) {
        BMPHeader header;
        BMPInfoHeader infoHeader;

        size_t rowStride = (width * 3 + 3) & ~3;

        header.fileSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + rowStride * height;
        header.offsetData = sizeof(BMPHeader) + sizeof(BMPInfoHeader);

        infoHeader.size = sizeof(BMPInfoHeader);
        infoHeader.width = width;
        infoHeader.height = height;
        infoHeader.bitCount = 24;
        infoHeader.sizeImage = rowStride * height;

        out.write((const char*)&header, sizeof(header));
        out.write((const char*)&infoHeader, sizeof(infoHeader));

        vector<ap_uint<8>> tmpData(rowStride * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ap_uint<8> pixel = data[y * width + x];
                tmpData[(y * rowStride) + (x * 3)] = pixel;
                tmpData[(y * rowStride) + (x * 3) + 1] = pixel;
                tmpData[(y * rowStride) + (x * 3) + 2] = pixel;
            }
        }

        out.write((const char*)tmpData.data(), tmpData.size());
    } else {
        cerr << "Error writing BMP file." << endl;
    }
}
