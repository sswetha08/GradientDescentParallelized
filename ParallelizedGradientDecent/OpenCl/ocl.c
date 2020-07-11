#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SOURCE_SIZE (0x100000)

int main ()
{

    int i;
    int LIST_SIZE = 500;


    // 0. Create and initialize the vectors

    int *X = (int*)malloc(sizeof(int) * LIST_SIZE);
    int *Y = (int*)malloc(sizeof(int) * LIST_SIZE);
    float* Deriv = (float *)malloc(sizeof(float) * LIST_SIZE);
    float* Diff = (float *)malloc(sizeof(float) * LIST_SIZE);
    float *Diff_Sqr = (float *)malloc(sizeof(float) * LIST_SIZE);
    float b0 = 0, b1 = 0, alpha =  0.0000205;
    int size_per_thread = 50;

    for(i=0;i<LIST_SIZE;i++)
	{
	X[i] = i;
	Y[i] = 2*i + 4;
	}


    // 1. Load the kernel code

    FILE *kernel_code_file_1 = fopen("cost.cl", "r");
    if (kernel_code_file_1 == NULL)
    {
        printf("Kernel loading failed.");
        exit(1);
    }


    char *source_str_1 = (char *)malloc(MAX_SOURCE_SIZE);
    size_t source_size_1 = fread(source_str_1, 1, MAX_SOURCE_SIZE, kernel_code_file_1);

    fclose(kernel_code_file_1);


    // 2. Get platform and device information

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;

    cl_uint ret_num_devices, ret_num_platforms;

    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);


    // 3. Create an OpenCL context

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);


    // 4. Create a command queue

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, &ret);


    // 5. Create memory buffers on the device for each vector X,Y,Diff,Diff_Sqr

   cl_mem mem_obj_X = clCreateBuffer(context, CL_MEM_READ_WRITE, LIST_SIZE * sizeof(int), NULL, &ret);
   cl_mem mem_obj_Y = clCreateBuffer(context, CL_MEM_READ_WRITE, LIST_SIZE * sizeof(int), NULL, &ret);
   cl_mem mem_obj_Deriv = clCreateBuffer(context, CL_MEM_READ_WRITE, LIST_SIZE * sizeof(float), NULL, &ret);
   cl_mem mem_obj_Diff = clCreateBuffer(context, CL_MEM_READ_WRITE, LIST_SIZE * sizeof(float), NULL, &ret);
   cl_mem mem_obj_Diff_Sqr = clCreateBuffer(context, CL_MEM_READ_WRITE, LIST_SIZE * sizeof(float), NULL, &ret);


    // 6. Copy the list X,Y to the respective memory buffers

    ret = clEnqueueWriteBuffer(command_queue, mem_obj_X, CL_TRUE, 0, LIST_SIZE * sizeof(int), X, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, mem_obj_Y, CL_TRUE, 0, LIST_SIZE * sizeof(int), Y, 0, NULL, NULL);


    // 7. Create a program from kernel source

    cl_program program1 = clCreateProgramWithSource(context, 1, (const char **)&source_str_1,
   (const size_t*)&source_size_1, &ret);


    // 8. Build the kernel program

    ret = clBuildProgram(program1, 1, &device_id, NULL, NULL, NULL);

    // 9. Create the OpenCL kernel object

    cl_kernel kernel1 = clCreateKernel(program1, "calc_cost", &ret);

    // 10. Set the arguments of the kernel

    ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&mem_obj_X);
    ret = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&mem_obj_Y);
    ret = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&mem_obj_Deriv);
    ret = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&mem_obj_Diff);
    ret = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void *)&mem_obj_Diff_Sqr);
    ret = clSetKernelArg(kernel1, 5, sizeof(cl_float), (void *)&b0);
    ret = clSetKernelArg(kernel1, 6, sizeof(cl_float), (void *)&b1);
    ret = clSetKernelArg(kernel1, 7, sizeof(cl_int), (void *)&size_per_thread);


    // 11. Execute the kernel on device

    size_t global_item_size = LIST_SIZE/size_per_thread;
    size_t local_item_size = 1;

    int j=0,count =0;
    float diffSum,derivSum,cost = 1000;
    while(cost>0.001)
    {
    count++;
    ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    ret = clFinish(command_queue);

    ret = clEnqueueReadBuffer(command_queue, mem_obj_Diff, CL_TRUE, 0, LIST_SIZE * sizeof(float), Diff, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, mem_obj_Deriv, CL_TRUE, 0, LIST_SIZE * sizeof(float), Deriv, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, mem_obj_Diff_Sqr, CL_TRUE, 0, LIST_SIZE * sizeof(float), Diff_Sqr, 0,
    NULL,NULL);
    diffSum = 0,derivSum = 0,cost = 0;
   // printf("\n ITER : %d",count);
    for(j=0;j<LIST_SIZE;j++)
	{
	diffSum+= Diff[j];
	derivSum+= Deriv[j];
	cost+= Diff_Sqr[j];

	/*
	printf("\n Diff: %f",Diff[j]);
	printf(" Deriv: %f",Deriv[j]);
	printf(" Diffsq: %f",Diff_Sqr[j]);
	*/

	}
    printf("\n b0: %f b1 : %f\n",b0,b1);
    cost = cost/(2*LIST_SIZE);
    printf("\n Cost: %f",cost);
    b0 = b0 - 50*alpha*diffSum/LIST_SIZE;
    b1 = b1 - alpha*derivSum/LIST_SIZE;
    ret = clSetKernelArg(kernel1, 5, sizeof(cl_float), (void *)&b0);
    ret = clSetKernelArg(kernel1, 6, sizeof(cl_float), (void *)&b1);

    }

    printf("\n Data Trained\n");
    printf("\n b0: %f b1 : %f\n",b0,b1);


    clFlush(command_queue);
    clReleaseKernel(kernel1);
    clReleaseProgram(program1);
    clReleaseMemObject(mem_obj_X);
    clReleaseMemObject(mem_obj_Y);
    clReleaseMemObject(mem_obj_Diff);
    clReleaseMemObject(mem_obj_Deriv);
    clReleaseMemObject(mem_obj_Diff_Sqr);

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(X);
    free(Y);
    free(Diff);
    free(Deriv);
    free(Diff_Sqr);

    int d;
    scanf(" %d", &d);

    return 0;

}
