#define NOMINMAX // so that windows.h does not define min/max macros

#include <algorithm>
#include <iostream>
#include <time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "defines.h"
#include "utils.h"

using namespace aocl_utils;

#define WIDTH 321
#define HEIGHT 481

// OpenCL Global Variables.
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel kernels[3]; // Grayscale, Laplacian, Sharpen
cl_program program;

enum {GRAYSCALE_KERNEL, LAPLACIAN_KERNEL, SHARPEN_KERNEL};

// Global variables.
unsigned char *h_input = NULL;
unsigned char *h_grayscale_output = NULL;
unsigned char *h_filtered_output = NULL;
unsigned char *h_sharpened_output = NULL;
cl_mem input_buffer, grayscale_output_buffer, filtered_output_buffer, sharpened_output_buffer;

std::string imageFilename;
std::string aocxFilename;
std::string deviceInfo;
unsigned char* bmp_header;
int cols, rows;
int width, height;
char outputfile[256];

// Function prototypes.
void process_image();
void initCL();
void cleanup();
void teardown(int exit_status = 1);
void print_usage();

int main(int argc, char **argv) {
    // Parsing command line arguments.
    Options options(argc, argv);

    // Relative path to image filename option.
    if (options.has("img")) {
        imageFilename = options.get<std::string>("img");
    } else {
        print_usage();
        return 0;
    }

    // Relative path to aocx filename.
    if (options.has("aocx")) {
        aocxFilename = options.get<std::string>("aocx");
    } else {
        aocxFilename = "process_image";
    }

    // Load the image.
    bmp_header = (unsigned char*) malloc(BMP_HEADER_SIZE * sizeof(unsigned char));
    if (!read_bmp(imageFilename.c_str(), bmp_header, (struct pixel**)&h_input)) {
        std::cerr << "Error: could not load " << argv[1] << std::endl;
        teardown(-1);
    }
    cols = *(int*)&bmp_header[18];
    rows = *(int*)&bmp_header[22];
    width = *(int*)&bmp_header[18];
    height = *(int*)&bmp_header[22];
    std::cout << "Input image dimensions: " << cols << "x" << rows << std::endl;

    // Ensure the image dimensions are correct
    if (cols != WIDTH || rows != HEIGHT) {
        std::cerr << "Error: image should be " << WIDTH << "x" << HEIGHT << " pixels, but is actually " << cols << "x" << rows << std::endl;
        teardown(-1);
    }

    // Initializing OpenCL and the kernels.
    h_grayscale_output = (unsigned char*) alignedMalloc(sizeof(unsigned char) * rows * cols);
    h_filtered_output = (unsigned char*) alignedMalloc(sizeof(unsigned char) * rows * cols);
    h_sharpened_output = (unsigned char*) alignedMalloc(sizeof(unsigned char) * rows * cols);
    initCL();

    // Start measuring process_image time.
    double start = get_wall_time();

    // Call the process_image.
    process_image();

    // Stop measuring the process_image time.
    double end = get_wall_time();
    printf("TIME ELAPSED: %.2f ms\n", end - start);

    // Write out the processed images.
    snprintf(outputfile, 256, "%s_grayscale.bmp", imageFilename.c_str());
    printf("Writing grayscale image to %s\n", outputfile);
    write_bmp(outputfile, bmp_header, (struct pixel*)h_grayscale_output);

    snprintf(outputfile, 256, "%s_filtered.bmp", imageFilename.c_str());
    printf("Writing filtered image to %s\n", outputfile);
    write_bmp(outputfile, bmp_header, (struct pixel*)h_filtered_output);

    snprintf(outputfile, 256, "%s_sharpened.bmp", imageFilename.c_str());
    printf("Writing sharpened image to %s\n", outputfile);
    write_bmp(outputfile, bmp_header, (struct pixel*)h_sharpened_output);

    // Teardown OpenCL.
    teardown(0);
}

void process_image() {
    cl_int status;

    // SETUP KERNEL ARGUMENTS AND BUFFERS
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * rows * cols * 3, NULL, &status); // 3 channels for RGB
    checkError(status, "Error: could not create input_buffer");
    grayscale_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * rows * cols, NULL, &status);
    checkError(status, "Error: could not create grayscale_output_buffer");
    filtered_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * rows * cols, NULL, &status);
    checkError(status, "Error: could not create filtered_output_buffer");
    sharpened_output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * rows * cols, NULL, &status);
    checkError(status, "Error: could not create sharpened_output_buffer");

    // Copy data to kernel input buffer.
    status = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(unsigned char) * rows * cols * 3, h_input, 0, NULL, NULL);
    checkError(status, "Error: could not copy data into device");

    // Enqueue Grayscale Kernel
    status = clSetKernelArg(kernels[GRAYSCALE_KERNEL], 0, sizeof(cl_mem), (void*)&input_buffer);
    checkError(status, "Error: could not set grayscale kernel arg 0");
    status = clSetKernelArg(kernels[GRAYSCALE_KERNEL], 1, sizeof(cl_mem), (void*)&grayscale_output_buffer);
    checkError(status, "Error: could not set grayscale kernel arg 1");
    status = clSetKernelArg(kernels[GRAYSCALE_KERNEL], 2, sizeof(int), &width);
    checkError(status, "Error: could not set grayscale kernel arg 2");
    status = clSetKernelArg(kernels[GRAYSCALE_KERNEL], 3, sizeof(int), &height);
    checkError(status, "Error: could not set grayscale kernel arg 3");

    size_t global_work_size[2] = {width, height};
    status = clEnqueueNDRangeKernel(queue, kernels[GRAYSCALE_KERNEL], 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    checkError(status, "Error: failed to enqueue grayscale kernel");

    status = clFinish(queue);
    checkError(status, "Failed to finish");
    // Enqueue Laplacian Kernel
    status = clSetKernelArg(kernels[LAPLACIAN_KERNEL], 0, sizeof(cl_mem), (void*)&grayscale_output_buffer);
    checkError(status, "Error: could not set laplacian kernel arg 0");
    status = clSetKernelArg(kernels[LAPLACIAN_KERNEL], 1, sizeof(cl_mem), (void*)&filtered_output_buffer);
    checkError(status, "Error: could not set laplacian kernel arg 1");
    status = clSetKernelArg(kernels[LAPLACIAN_KERNEL], 2, sizeof(int), &width);
    checkError(status, "Error: could not set laplacian kernel arg 2");
    status = clSetKernelArg(kernels[LAPLACIAN_KERNEL], 3, sizeof(int), &height);
    checkError(status, "Error: could not set laplacian kernel arg 3");

    status = clEnqueueNDRangeKernel(queue, kernels[LAPLACIAN_KERNEL], 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    checkError(status, "Error: failed to enqueue laplacian kernel");

    status = clFinish(queue);
    checkError(status, "Failed to finish");

    // Enqueue Sharpen Kernel
    status = clSetKernelArg(kernels[SHARPEN_KERNEL], 0, sizeof(cl_mem), (void*)&grayscale_output_buffer);
    checkError(status, "Error: could not set sharpen kernel arg 0");
    status = clSetKernelArg(kernels[SHARPEN_KERNEL], 1, sizeof(cl_mem), (void*)&filtered_output_buffer);
    checkError(status, "Error: could not set sharpen kernel arg 1");
    status = clSetKernelArg(kernels[SHARPEN_KERNEL], 2, sizeof(cl_mem), (void*)&sharpened_output_buffer);
    checkError(status, "Error: could not set sharpen kernel arg 2");
    status = clSetKernelArg(kernels[SHARPEN_KERNEL], 3, sizeof(int), &width);
    checkError(status, "Error: could not set sharpen kernel arg 3");
    status = clSetKernelArg(kernels[SHARPEN_KERNEL], 4, sizeof(int), &height);
    checkError(status, "Error: could not set sharpen kernel arg 4");

    status = clEnqueueNDRangeKernel(queue, kernels[SHARPEN_KERNEL], 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    checkError(status, "Error: failed to enqueue sharpen kernel");

    status = clFinish(queue);
    checkError(status, "Failed to finish");

    // Read output buffers from kernels.
    status = clEnqueueReadBuffer(queue, grayscale_output_buffer, CL_TRUE, 0, sizeof(unsigned char) * rows * cols, h_grayscale_output, 0, NULL, NULL);
    checkError(status, "Error: could not copy grayscale data from device");

    status = clEnqueueReadBuffer(queue, filtered_output_buffer, CL_TRUE, 0, sizeof(unsigned char) * rows * cols, h_filtered_output, 0, NULL, NULL);
    checkError(status, "Error: could not copy filtered data from device");

    status = clEnqueueReadBuffer(queue, sharpened_output_buffer, CL_TRUE, 0, sizeof(unsigned char) * rows * cols, h_sharpened_output, 0, NULL, NULL);
    checkError(status, "Error: could not copy sharpened data from device");
}

void initCL() {
    cl_int status;

    // Start everything at NULL to help identify errors.
    kernels[GRAYSCALE_KERNEL] = NULL;
    kernels[LAPLACIAN_KERNEL] = NULL;
    kernels[SHARPEN_KERNEL] = NULL;
    queue = NULL;

    // Locate files via relative paths.
    if (!setCwdToExeDir()) {
        teardown();
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA");
    if (platform == NULL) {
        teardown();
    }

    // Get the first device.
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    checkError(status, "Error: could not query devices");

    char info[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
    deviceInfo = info;

    // Create the context.
    context = clCreateContext(0, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "Error: could not create OpenCL context");

    // Create the command queues for the kernels.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);

    // Create the program.
    std::string binary_file = getBoardBinaryFile(aocxFilename.c_str(), device);
    std::cout << "Using AOCX: " << binary_file << "\n";
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 1, &device, "", NULL, NULL);
    checkError(status, "Error: could not build program");

    // Create the kernels - names must match kernel names in the original CL file.
    kernels[GRAYSCALE_KERNEL] = clCreateKernel(program, "grayscale", &status);
    checkError(status, "Failed to create grayscale kernel");

    kernels[LAPLACIAN_KERNEL] = clCreateKernel(program, "laplacian", &status);
    checkError(status, "Failed to create laplacian kernel");

    kernels[SHARPEN_KERNEL] = clCreateKernel(program, "sharpen", &status);
    checkError(status, "Failed to create sharpen kernel");
}

void cleanup() {
    // Called from aocl_utils::check_error, so there's an error.
    teardown(-1);
}

void teardown(int exit_status) {
    if (kernels[GRAYSCALE_KERNEL])
        clReleaseKernel(kernels[GRAYSCALE_KERNEL]);
    if (kernels[LAPLACIAN_KERNEL])
        clReleaseKernel(kernels[LAPLACIAN_KERNEL]);
    if (kernels[SHARPEN_KERNEL])
        clReleaseKernel(kernels[SHARPEN_KERNEL]);
    if (queue)
        clReleaseCommandQueue(queue);

    if (h_input)
        alignedFree(h_input);
    if (h_grayscale_output)
        alignedFree(h_grayscale_output);
    if (h_filtered_output)
        alignedFree(h_filtered_output);
    if (h_sharpened_output)
        alignedFree(h_sharpened_output);
    if (input_buffer)
        clReleaseMemObject(input_buffer);
    if (grayscale_output_buffer)
        clReleaseMemObject(grayscale_output_buffer);
    if (filtered_output_buffer)
        clReleaseMemObject(filtered_output_buffer);
    if (sharpened_output_buffer)
        clReleaseMemObject(sharpened_output_buffer);
    if (program)
        clReleaseProgram(program);
    if (context)
        clReleaseContext(context);

    exit(exit_status);
}

void print_usage() {
    printf("\nUsage:\n");
    printf("\tprocess_image --img=<img> [--aocx=<aocx file>]\n\n");
    printf("Options:\n\n");
    printf("--img=<img>\n");
    printf("\tThe relative path to the input image to be processed.\n\n");
    printf("[--aocx=<aocx file>]\n");
    printf("\tThe relative path to the aocx file without the .aocx suffix (default: process_image).\n\n");
}
