#include "./cl.hpp"             //C++ bindings
#include <OpenCL/opencl.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mpi.h> // For timing
#include <random>

using namespace std;

//----------------------------------------------------
// Struct for Storing Device Info
// A lot more could be added, just some basics
//----------------------------------------------------

typedef struct
{
    std::string device_name;
    std::string device_vendor;
    std::string device_version;
    cl_device_type device_type;
    cl_bool device_available;
    cl_bool device_compiler_available;
    std::string device_extensions;
    cl_ulong device_global_memory_size; //bytes
    cl_ulong device_local_memory_size; //bytes
    cl_ulong device_max_mem_alloc_size; //bytes
    cl_ulong device_global_mem_cache_size; //bytes
    cl_uint device_global_mem_cacheline_size; //bytes
    cl_uint device_max_compute_units;
    cl_uint device_max_clock_frequency; //MHz
    size_t device_max_work_group_size;
    cl_uint device_max_work_item_dimensions;
    size_t device_max_work_item_sizes[3];   //assuming max dimensions is three
    
}
device_info_t;


//----------------------------------------------------
// PLATFORM SETUP (C++)
// Return a list of available OpenCL Platforms.
//      Uses C++ Bindings
//----------------------------------------------------

cl_int getOpenCLPlatformsCPP(std::vector<cl::Platform>& platforms, bool verbose)
{
    cl_int err;
    
    //-----------------------------------
    // Get Platforms and Handle Errors
    //-----------------------------------
    
    platforms.resize(0);
    err = cl::Platform::get(&platforms);
    
    if (err != CL_SUCCESS)
    {
        cerr << "Unable to get platforms. Error code = " << err << endl;
        return err;
    }
    
    //-----------------------------------
    // Get and Output Platform Information
    //-----------------------------------
    
    if (verbose == true)
    {
        for (unsigned int iplat = 0; iplat < platforms.size(); iplat++)
        {
            std::string platform_vendor = platforms[iplat].getInfo<CL_PLATFORM_VENDOR>(&err); //error can be obtained
            std::string platform_version = platforms[iplat].getInfo<CL_PLATFORM_VERSION>(); //or error can be omitted
            std::string platform_name = platforms[iplat].getInfo<CL_PLATFORM_NAME>();
            std::string platform_extensions = platforms[iplat].getInfo<CL_PLATFORM_EXTENSIONS>();
            
            cout << "Information for Platform " << iplat << ": " << endl;
            cout << "\tVendor        : " << platform_vendor << endl;
            cout << "\tVersion        : " << platform_version << endl;
            cout << "\tName        : " << platform_name << endl;
            cout << "\tExtensions    : " << platform_extensions << endl;
        }
        cout << "\n\n";
    }
    return err;
}


//----------------------------------------------------
// Function for Getting Device Info
//----------------------------------------------------

void getDeviceInfo(cl::Device &device, device_info_t& info)
{
    info.device_name = device.getInfo<CL_DEVICE_NAME>();
    info.device_type = device.getInfo<CL_DEVICE_TYPE>();
    info.device_vendor = device.getInfo<CL_DEVICE_VENDOR>();
    info.device_version = device.getInfo<CL_DEVICE_VERSION>();
    info.device_available = device.getInfo<CL_DEVICE_AVAILABLE>();
    info.device_compiler_available = device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
    info.device_extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    info.device_global_memory_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    info.device_local_memory_size = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    info.device_max_mem_alloc_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    info.device_global_mem_cache_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    info.device_global_mem_cacheline_size = device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    info.device_max_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    info.device_max_clock_frequency = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    info.device_max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    info.device_max_work_item_dimensions = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    
    for (int idim = 0; idim < 3; idim++)
    {
        info.device_max_work_item_sizes[idim] = -1;
    }
    
    for (int idim = 0; idim < info.device_max_work_item_dimensions; idim++)
    {
        info.device_max_work_item_sizes[idim] = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[idim];
    }
}


//----------------------------------------------------
// Function for Printing Device Info
//----------------------------------------------------

void printDeviceInfo(const device_info_t& info)
{
    cout << "\nDevice Information " << endl;
    cout << "\tName                    : " << info.device_name << "\n";
    cout << "\tType                    : ";
    if (info.device_type == CL_DEVICE_TYPE_CPU) std::cout << "CPU\n";
    else if (info.device_type == CL_DEVICE_TYPE_GPU) std::cout << "GPU\n";
    else if (info.device_type == CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR\n";
    else if (info.device_type == CL_DEVICE_TYPE_DEFAULT) std::cout << "DEFAULT\n";
    else std::cout << "(ERROR!)\n";
    cout << "\tVendor                  : " << info.device_vendor << "\n";
    cout << "\tVersion                 : " << info.device_version << "\n";
    cout << "\tAvailable               : " << info.device_available << "\n";
    cout << "\tCompiler Available      : " << info.device_compiler_available << "\n";
    cout << "\tGlobal Memory Size      : " << info.device_global_memory_size/(1024*1024*1024.0) << " GB\n";
    cout << "\tLocal Memory Size       : " << info.device_global_memory_size/(1024*1024*1024.0) << " GB\n";
    cout << "\tMaximum Memory Alloc    : " << info.device_max_mem_alloc_size/(1024*1024.0) << " MB\n";
    cout << "\tGlobal Memory Cache     : " << info.device_global_mem_cache_size << " Bytes\n";
    cout << "\tGlobal Memory CacheLine : " << info.device_global_mem_cacheline_size << " Bytes\n";
    cout << "\tMax Compute Units       : " << info.device_max_compute_units << "\n";
    cout << "\tMax Clock Frequency     : " << info.device_max_clock_frequency << " MHz\n";
    cout << "\tMax Work Group Size     : " << info.device_max_work_group_size << "\n";
    cout << "\tMax Work Group Dim      : " << info.device_max_work_item_dimensions << "\n";
    
    int maximum_work_item_product = 1;
    for (int idim = 0; idim < info.device_max_work_item_dimensions; idim++)
    {
        cout << "\tMax Work Item Size - Dimension " << idim << " : " << info.device_max_work_item_sizes[idim] << "\n";
        maximum_work_item_product *= info.device_max_work_item_sizes[idim];
    }
    cout << "\tMaximum Number of Work Items Globally   : " << maximum_work_item_product << "\n";
}

//----------------------------------------------------
// DEVICE SETUP
// Return a list of Devices from a Platform
//----------------------------------------------------

cl_int getOpenCLDevices(const cl::Platform& platform, std::vector<cl::Device>& devices, cl_device_type device_type, bool verbose)
{
    if (device_type != CL_DEVICE_TYPE_ALL && device_type != CL_DEVICE_TYPE_CPU && device_type != CL_DEVICE_TYPE_GPU)
    {
        cerr << "Unknown Device Type!" << endl;
        cerr << "Assuming ALL Devices" << endl;
        device_type = CL_DEVICE_TYPE_ALL;
    }
    
    cl_int err = platform.getDevices(device_type, &devices);
    
    if (err != CL_SUCCESS)
    {
        cerr << "Error Occurred Calling Platform::getDevices(). Error code = " << err << endl;
        return err;
    }
    
    if (verbose == true)
    {
        for (unsigned int idevice = 0; idevice < devices.size(); idevice++)
        {
            device_info_t device_info;
            getDeviceInfo(devices[idevice], device_info);
            printDeviceInfo(device_info);
        }
    }
    return err;
}

//-------------------
// Main
//-------------------

int main(int argc, char** argv){
    
        //-----------------
        // Get command
        // line arguments
        // ----------------

        int m = atoi(argv[1]); // This will be the 'm' value
        int n = atoi(argv[2]); // This will be the 'n' value

        //cout << m << " " << n << endl;

    
        //----------------------------------------------------
        // PLATFORM SETUP
        //----------------------------------------------------

        cl_int err;
    
        cout << "\n------------------------------------------\n";
        cout << " Getting Platforms\n";
        cout << "------------------------------------------\n";

        std::vector<cl::Platform> platforms;
        err = getOpenCLPlatformsCPP(platforms, true);
    
        if (err == CL_SUCCESS)
            cout << "\nSuccessfully retrieved platform...\n" << endl;
    
    
        //----------------------------------------------------
        // DEVICES SETUP
        //----------------------------------------------------

        std::vector<cl::Device> gpu_devices;

        cout << "-------------------------\n";
        cout << " Get GPU Devices\n";
        cout << "-------------------------\n";

        err = getOpenCLDevices(platforms[0], gpu_devices, CL_DEVICE_TYPE_GPU, true);

        if (err == CL_SUCCESS)
            cout << "Successfully retrieved devices...\n" << endl;
    
    
        //----------------------------------------------------
        // CONTEXT SETUP
        //----------------------------------------------------
    
        cout << "-------------------------\n";
        cout << " Creating GPU Context\n";
        cout << "-------------------------\n";
    
    
        //----------------------
        // Create Context
        //----------------------
    
        cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        cl::Context context(gpu_devices, context_properties, NULL, NULL, &err);
        if (err != CL_SUCCESS)
        {
            cerr << "Error Creating Context. Error Code: " << err << endl;
            return -1;
        }
        else
        {
            cout << "\nSuccessfully created context...\n" << endl;
        }
    
        //----------------------
        // Get Context Devices
        //----------------------

        vector<cl::Device> context_devices = context.getInfo<CL_CONTEXT_DEVICES>();

        //----------------------
        // Print Context Devices
        //----------------------

        cout << "---------------------------------------------\n";
        cout << "Devices in Context: " << std::endl;
        cout << "---------------------------------------------\n";
    
        for (unsigned int idevice = 0; idevice < context_devices.size(); idevice++)
        {
            device_info_t device_info;
            getDeviceInfo(context_devices[idevice], device_info);
            printDeviceInfo(device_info);
        }
    
    
    //----------------------------------------------------
    // COMMAND QUEUE SETUP
    //----------------------------------------------------
    
    cout << "\n---------------------------------------------\n";
    cout << " Creating Command Queue \n";
    cout << "---------------------------------------------\n";
    
    
    //----------------------
    // Create Command Queue
    //----------------------
    
    cl::CommandQueue gpu_command_queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0],0,&err);
    if (err != CL_SUCCESS)
    {
        cerr << "Error Creating Command Queue. Error Code: " << err << endl;
        return -1;
    }
    else
    {
        cout << "\nSuccessfully created command queue...\n" << endl;
    }
    
    
    //-----------------------------------------------
    // Get Context and Device for Queue:
    // This isn't necessary as they should be the same
    // as what we created the command queue with.
    // Just a good check.
    //-----------------------------------------------
    
    cl::Context command_queue_context = gpu_command_queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Device command_queue_device = gpu_command_queue.getInfo<CL_QUEUE_DEVICE>();
    
    
    //--------------------------
    // Output Command Queue Info
    //--------------------------
    
    cout << "-------------------------\n";
    cout << "Command Queue Info: " << std::endl;
    cout << "-------------------------\n";
    
    cout << "\nCommand Queue is for Device: " << std::endl;
    device_info_t device_info;
    getDeviceInfo(command_queue_device, device_info);
    printDeviceInfo(device_info);
    
    
    //----------------------------------------------------
    // PROGRAM SETUP
    //----------------------------------------------------
    
    cout << "\n---------------------------------------------\n";
    cout << "Program Info: " << std::endl;
    cout << "---------------------------------------------\n";
    
    
    //-----------------------------------
    // Read the kernel file into a string
    //----------------------------------
    
    ifstream in_from_source("./kernels.cl");
    string kernel_code(std::istreambuf_iterator<char>(in_from_source), (std::istreambuf_iterator<char>()));
    
    
    //----------------------------------
    // Format the kernel string properly
    // for program constructor
    //----------------------------------
    
    cl::Program::Sources kernel_source(1,std::make_pair(kernel_code.c_str(), kernel_code.length() + 1));
    
    
    //-------------------------------------------------
    // Create a Program in this context for that source
    //-------------------------------------------------
    
    cl::Program program(context, kernel_source, &err);
    
    if (err != CL_SUCCESS)
    {
        cerr << "Error Creating Program. Error Code: " << err << endl;
        return -1;
    }
    else
    {
        cout << "\nSuccessfully created program..." << endl;
    }
    
    //----------------------------
    // Output kernel file to screen
    //----------------------------
    
    //If you have trouble this is a good idea - invalid characters can be a problem.
    cout << "\nKernel Code = " << endl;
    cout << kernel_code << endl;
    
    
    //----------------------------
    // Build the functions in the program for devices
    //----------------------------
	
    err = program.build(context_devices,NULL,NULL,NULL);
    
    if (err != CL_SUCCESS)
    {
        cerr << "Error Building Program. Error Code: " << err << endl;
        return -1;
    }
    else
    {
        cout << "Program Built Successfully." << endl;
    }
    
    vector<size_t> program_binary_sizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    assert(program_binary_sizes.size() == context_devices.size());
    cout << "Binaries size = " << program_binary_sizes[0] << " bytes" <<  endl;
    
    
    //----------------------------
    // Create Kernel
    //----------------------------
    
    cout << "\n---------------------------------------------\n";
    cout << "Creating Kernel: " << std::endl;
    cout << "---------------------------------------------\n";
    
    string kernel_function_name = "transpose";
    cl::Kernel kernel(program,kernel_function_name.c_str(), &err);
    
    if (err != CL_SUCCESS)
    {
        cerr << "Error Creating Kernel Object. Error Code: " << err << endl;
    }
    else
    {
        cout << "\nSucessfully created kernel....\n" << endl;
    }
    cout << "Kernel Function Name = " << kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() << endl;
    cout << "Kernel Number of Arguments = " << kernel.getInfo<CL_KERNEL_NUM_ARGS>() << endl;
    
    
    //----------------------------
    // Kernel Argument Allocation
    //----------------------------
    
    // Create solution space
    
    int N = m*n;       //size of inputMatrix & outputMatrix
    int count = 0;
    
    srand(time(NULL));
    
    vector<int> a(N), aTranspose(N);
    vector <vector <int> > matrix(m);
    for (unsigned int i = 0; i < N; i++)
    {
        a[i] = rand()%1000; // Random integer between 0 and 999
        aTranspose[i] = 0;
        matrix[i/n].push_back(a[i]); //  used for error checking
    }
    
    // Print matrices if 3rd argument presented
    if (argv[3] != NULL){
        
        // Print input matrix
        cout << "\nInput Matrix:" << endl;
        count = 0;
        for (int irow = 0; irow < m; irow++){
            for (int icol = 0; icol < n; icol++){
                cout << a[count] << " ";
                count++;
            }
            cout << endl;
        }

        // PRINT ERROR CHECKING MATRIX (NOT USING OPENCL)
        cout << "\nOutput Matrix (not using OpenCL - error check):" << endl;
        for (int icol = 0; icol < n; icol++){
            for (int irow = 0; irow < m; irow++){
                cout << matrix[irow][icol] << " ";
                count++;
            }
            cout << endl;
        }
    }

    
    //---------------------------
    // Create Buffers
    //---------------------------
    
    cl::Buffer cl_a(context, CL_MEM_READ_ONLY, N*sizeof(int), NULL, &err);
    assert(err == CL_SUCCESS);
    
    cl::Buffer cl_aTranspose(context, CL_MEM_READ_ONLY, N*sizeof(int), NULL, &err);
    assert(err == CL_SUCCESS);
    
    //---------------------------
    // Put Buffers in Write Queue
    // - this passes to GPU
    //---------------------------
    
    err = gpu_command_queue.enqueueWriteBuffer(cl_a, CL_TRUE, 0, N*sizeof(int), &a[0], NULL, NULL);
    assert(err == CL_SUCCESS);
    
    err = gpu_command_queue.enqueueWriteBuffer(cl_aTranspose, CL_TRUE, 0, N*sizeof(int), &aTranspose[0], NULL, NULL);
    assert(err == CL_SUCCESS);
    
    // No need for m and n buffers - can be set using setArg
    
    //---------------------------
    // Set Kernel Arguments
    //---------------------------
    
    err = kernel.setArg(0, cl_a);
    err = kernel.setArg(1, cl_aTranspose);
    err = kernel.setArg(2, m);
    err = kernel.setArg(3, n);
    
    
    //----------------------------------------------------
    // KERNEL EXECUTION
    //----------------------------------------------------
    
    std::cout << "\n---------------------------------------------\n";
    std::cout << "Executing Kernel: " << std::endl;
    std::cout << "---------------------------------------------\n";
    
    double tStart = MPI_Wtime();
    
    err = gpu_command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(m,n), cl::NullRange, NULL, NULL);
    
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error Enqueueing NDRange Kernel. Error Code: " << err << std::endl;
    }
    else
    {
        std::cout << "\nNDRange Kernel Enqueued Successfully..." << std::endl;
    }
    
    
    //---------------------------
    // Read Result
    //---------------------------
    
    err = gpu_command_queue.enqueueReadBuffer(cl_aTranspose, CL_TRUE, 0, N*sizeof(int), &aTranspose[0], NULL, NULL);

    double tStop = MPI_Wtime();
    
    //---------------------------
    // Display Final Results
    //---------------------------
    
    // Print matrix if 3rd argument included
    if (argv[3] != NULL){
        
        // Print output matrix
        cout << "\nOutput Matrix (using OpenCL):" << endl;
        count = 0;
        for (int irow = 0; irow < n; irow++){
            for (int icol = 0; icol < m; icol++){
                cout << aTranspose[count] << " ";
                count++;
            }
            cout << endl;
        }
    }
    
    // Error checking (by subtracting known aTranspose from OpenCL aTranspose)
    count = 0;
    int residualSum = 0;
    for (int icol = 0; icol < n; icol++)
    {
        for (int irow = 0; irow < m; irow++)
        {
            residualSum = aTranspose[count] - matrix[irow][icol];
            count++;
        }
    }
    
    // Print error checking results
    cout << "\nError Check:" << endl;
    cout << "Sum of residuals = " << residualSum << " when comparing OpenCL vs simple matrix transpose - therefore:" << endl;
    if (residualSum == 0)
        cout << "Serial and OpenCL versions of aTranspose are identical." << endl;
    else
        cout << "Serial and OpenCL versions of aTranspose are NOT THE SAME - ERROR OCCURED." << endl;
    
    // Print timing results
    cout << "\nTiming: " << endl;
    double tElapsed = (tStop-tStart)/CLOCKS_PER_SEC;
    cout << "For N = " << N << " elements, OpenCL took " << tElapsed << " seconds" <<endl;
    
    
} // End Main
