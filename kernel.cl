__kernel void grayscale(__global unsigned char* input_image, __global unsigned char* output_image, const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input_image[idx];
        unsigned char g = input_image[idx + 1];
        unsigned char b = input_image[idx + 2];
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        output_image[y * width + x] = gray;
    }
}

__kernel void laplacian(
    __global const uchar* grayscale_output,
    __global uchar* padded,
    __global uchar* filtered_output,
    int width, int height) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;

    int padded_width = width + 2;
    int padded_height = height + 2;
    if (x < width && y < height) {
        padded[(y + 1) * padded_width + (x + 1)] = grayscale_output[idx];
    }
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        padded[y * padded_width + x] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int laplacian[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};
        int filtered_value = 0;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int pixel = padded[(y + ky + 1) * padded_width + (x + kx + 1)];
                filtered_value += pixel * laplacian[ky + 1][kx + 1];
            }
        }
        filtered_output[idx] = clamp(filtered_value, 0, 255);
    } else {
        filtered_output[idx] = 0;
    }
}

__kernel void sharpen(
    __global const uchar* original_image,
    __global const uchar* filtered_output,
    __global uchar* sharpened_output,
    int width, int height) {

    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;

    int sharpened_value = original_image[idx] + filtered_output[idx];
    sharpened_output[idx] = clamp(sharpened_value, 0, 255);
}
