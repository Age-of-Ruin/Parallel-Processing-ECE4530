#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>

#define PIXEL_DESCRIPTOR_TAG 99
#define PIXEL_DATA_TAG 100

using namespace std;

void  parallelRange (int globalstart, int globalstop , int irank, int nproc, int &localstart, int &localstop, int &localcount)
{
	int  nrows = globalstop - globalstart  + 1;
	int divisor = nrows/nproc;
	int remainder  = nrows%nproc;
	int offset;
	
	if (irank<remainder) 
		offset = irank;
	else 
		offset = remainder;
	
	localstart = irank*divisor + globalstart + offset;
	localstop =  localstart + divisor - 1;
	
	if (remainder>irank) 
		localstop += 1;
	localcount  =  localstop - localstart + 1;
}


typedef struct
{
	//int rows, cols, type;
	int rowCount;
	int rowStart;
	int rowStop;
	int process_type;
	//add more as needed
} image_descriptor_t;


enum IMAGE_PROCESS_TYPE
{
	IMAGE_PROCESS_TYPE_MIN = 0, 
	IMAGE_PROCESS_TYPE_BLUR, 
	IMAGE_PROCESS_TYPE_ENHANCE,
	IMAGE_PROCESS_TYPE_GREYSCALE, 
	IMAGE_PROCESS_TYPE_MAX 
};


/************** IMPLEMENTED BY Lab ****************/

void imageContrastEnhance(const cv::Mat& in, cv::Mat& out, int minval, int maxval){
		
	// Set output image to input image
	out = in.clone();
		
	// Loop over all rows
	for (int irow = 0; irow < out.rows; irow++) 
	{ 
		// Loop over all cols
		for (int icol = 0; icol < out.cols; icol++) 
		{ 
			// Loop over each channel in pixel
			for (int ichannel = 0; ichannel < 3; ichannel++)
			{
				// Check each channel and adjust accordinly
				if (out.at<cv::Vec3b>(irow,icol).val[ichannel] <= minval)
				{
					out.at<cv::Vec3b>(irow,icol).val[ichannel] = 0;
				}
				else if ((out.at<cv::Vec3b>(irow,icol).val[ichannel] >= maxval))
				{
					out.at<cv::Vec3b>(irow,icol).val[ichannel] = 255;						
				}
				else{
					double slope = 255.0/((double)(maxval - minval));
					
					out.at<cv::Vec3b>(irow,icol).val[ichannel] = (int)(slope*out.at<cv::Vec3b>(irow,icol).val[ichannel] - slope*minval);
				}
			}
		} 
	} 
}



void imageGreyScale(const cv::Mat& in, cv::Mat& out){
	
	// Set output image to input image
	out = in.clone();
		
	// Loop over all rows
	for (int irow = 0; irow < out.rows; irow++) 
	{ 
		// Loop over all cols
		for (int icol = 0; icol < out.cols; icol++) 
		{ 
			// Create variable to store average
			long pixelAvg = 0;
			
			// Loop over each channel and acquire value
			for (int ichannel = 0; ichannel < 3; ichannel++)
			{
				pixelAvg += (long)out.at<cv::Vec3b>(irow,icol).val[ichannel];
			}
			
			// Calculate average
			pixelAvg = pixelAvg / 3;
			
			// Loop over each channel and set value
			for (int ichannel = 0; ichannel < 3; ichannel++)
			{
				out.at<cv::Vec3b>(irow,icol).val[ichannel] = (uchar)pixelAvg;
			}
		} 
	} 
}



void imageBlur(const cv::Mat& in, cv::Mat& out, int level, int rowstart, int rowstop){
	
	// Set output image to input image
	out = in.clone();
		
	// Loop over all rows
	for (int irow = rowstart; irow < rowstop; irow++) 
	{ 
		// Loop over all cols
		for (int icol = 0; icol < out.cols; icol++) 
		{ 
			
			// Bounds
			int blurRowStart = 0;
			int blurRowStop = 0;
			int blurColStart = 0;
			int blurColStop = 0;
			
			// Establish bounds
			if (irow - (2 * level + 1) >= rowstart)
			{
				blurRowStart = irow - (2 * level + 1);
			}
			else
			{
				blurRowStart =  rowstart;	
			}
			
			if (irow + (2 * level + 1) < rowstop)
			{
				blurRowStop = irow + (2 * level + 1);
			}
			else
			{
				blurRowStop = rowstop;
			}
			
			if (icol - (2 * level + 1) >= 0)
			{
				blurColStart = icol - (2 * level + 1);
			}
			else
			{
				blurColStart = 0;
			}
			
			if (icol + (2 * level + 1) < out.cols)
			{
				blurColStop = icol + (2 * level + 1);
			}
			else
			{
				blurColStop = out.cols;
			}
			
			
			// Loop over all channels
			double channelAvg;
			int count;
			for (int ichannel = 0; ichannel < 3; ichannel++){
				//Reset parameters
				channelAvg = 0;
				count = 0;
				
				// Loop over surrounding pixels and calculate average for 1 channel
				for (int blurRowIndex = blurRowStart; blurRowIndex <= blurRowStop; blurRowIndex++)
				{
					for (int blurColIndex = blurColStart; blurColIndex <= blurColStop; blurColIndex++)
					{
						channelAvg += (long)out.at<cv::Vec3b>(blurRowIndex,blurColIndex).val[ichannel];
						count++;				
					}
					
				}
				// Calculate average
				channelAvg = channelAvg / count;
				
				// Set channel value to average
				out.at<cv::Vec3b>(irow,icol).val[ichannel] = (uchar) channelAvg;
			}
		}
	}
}



int main(int argc, char** argv)  { 

	// MPI Declarations
	int rank = 0;						
	int nproc = 1;		
	
	//Initialize MPI
	MPI_Init(&argc, &argv);								
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// OpenCV declarations
	cv::Mat image; 
	
	// Used to enumerate process and hold parameters 
	int imageProcessID = 0;	
	int blurLevel = 0;
	int enhanceMin = 0;
	int enhanceMax = 255;
	
	// Create contiguous data type for 1 pixel  - need 3 bytes (1 per channel)
	MPI_Datatype cv_contig_Vec3b;
	MPI_Type_contiguous(3, MPI_BYTE, &cv_contig_Vec3b);
	MPI_Type_commit(&cv_contig_Vec3b);
	
	// Holds an image descriptor
	image_descriptor_t imgD;
	
	//------------------
	// RANK 0 OPERATIONS
	//------------------
	
	if(rank == 0){

		string imageName;
		string imageProcess;
		
		// Enumerate image process and store parameters
		if(argc > 2){
			
			// Parse command line arguements
			imageName = argv[1];
			imageProcess = argv[2];
			
			if(imageProcess == "min")
			{
				imageProcessID = IMAGE_PROCESS_TYPE_MIN;
			}
			else if(imageProcess == "blur" && argv[3] != NULL)
			{
				imageProcessID = IMAGE_PROCESS_TYPE_BLUR;
				blurLevel = atoi(argv[3]);
			}
			else if(imageProcess == "enhance" && argv[3] != NULL && argv[4] != NULL)
			{
				imageProcessID = IMAGE_PROCESS_TYPE_ENHANCE;
				enhanceMin = atoi(argv[3]);
				enhanceMax = atoi(argv[4]);
			}
			else if(imageProcess == "greyscale")
			{
				imageProcessID = IMAGE_PROCESS_TYPE_GREYSCALE;
			}
			else if(imageProcess == "max")
			{
				imageProcessID = IMAGE_PROCESS_TYPE_MAX;
			}
			else
			{
				cout << "No proper process selected, file name incorrect, or arguments are missing - terminating program." << endl;
				exit(1);
			} 
		}
		else
		{
			cout << "No proper process selected, file name incorrect, or arguments are missing - terminating program." << endl;
			exit(1);
		} 
	
		cout << "Rank " << rank << " reading image " << imageName << " and performing " << imageProcess << "." << endl;
		
		// Read image file (in color) - assumes picture in current directory
		string imageLocation = "./" + imageName;
		image = cv::imread(imageLocation,1);
		
		// Partition the image by rows (using parallel range)
		vector <int> localstart(nproc), localstop(nproc), localcount(nproc);
		for (int irank = 0; irank < nproc; irank++)
		{
			parallelRange(0, image.rows-1, irank, nproc, localstart[irank], localstop[irank], localcount[irank]);
		}
		
		// Create image descriptor for each processor
		vector <image_descriptor_t> imgDs(nproc);
		for (int irank = 0; irank < nproc; irank++)
		{
			imgDs[irank].rowStart = localstart[irank];
			imgDs[irank].rowStop = localstop[irank];
			imgDs[irank].rowCount = localcount[irank];
			imgDs[irank].process_type = imageProcessID;
		}
		
		// Set rank 0 image descriptor
		imgD = imgDs[0];
		
		// Send relevant image descriptor to all processors (except 0 - already has it)
        for (int irank = 1; irank < nproc; irank++)
        {
            MPI_Send(&imgDs[irank], imgDs.size()*sizeof(imgDs[irank]), MPI_BYTE, irank, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD);
        }
		
		// Extract pixels
		long pixelCount = 0;
		long proc1PixelCount = 0;
		vector<cv::Vec3b> allPixels(image.rows * image.cols);
		for (unsigned int irow = 0; irow < image.rows; irow++) 
		{ 
			for (unsigned int icol = 0; icol < image.cols; icol++) 
			{ 
				allPixels[pixelCount] = image.at<cv::Vec3b>(irow,icol);
				pixelCount++;
			}  
		}
		
		cout << "Total pixel count = " << pixelCount << endl;
		
		// Determine number of pixels per processor
		vector <long> procPixelCounts(nproc);
        for (int irank = 0; irank < nproc; irank++)
        {
            procPixelCounts[irank] = imgDs[irank].rowCount * image.cols;
			cout << "Rank "<< irank << " will process " << procPixelCounts[irank] << " pixels" << endl;
        }
		
		//
		

		// Send pixels to other processors using cv_contig_Vec3b
		for (int irank = 1; irank < nproc; irank++)
        {
            MPI_Send(&allPixels[procPixelCounts[0]], procPixelCounts[irank], cv_contig_Vec3b, irank, PIXEL_DATA_TAG, MPI_COMM_WORLD);
        }
		
		// Recieve processed image and write out to file
		
		
	}
	
	
	
	
	
	
	

	
	// Receive image descriptors
	if (rank != 0)
	{
		int recv_count;
        MPI_Status status;
        MPI_Probe(0, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &recv_count);
        
		MPI_Recv(&imgD, recv_count, MPI_BYTE, 0, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD, &status); //Receive
	}
	
	cout << "Rank " << rank << " will process this many rows " << imgD.rowCount << endl;
	
	// Receive pixel data
	vector <cv::Vec3b> procPixels;
	if (rank != 0)
	{
		int recv_count;
        MPI_Status status;
        MPI_Probe(0, PIXEL_DATA_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, cv_contig_Vec3b, &recv_count);  
		
		procPixels.resize(recv_count);
		
		cout << "Hello from rank " <<  rank << " received " << recv_count << " pixels." << endl;
		
		MPI_Recv(&procPixels[0], recv_count, cv_contig_Vec3b, 0, PIXEL_DATA_TAG, MPI_COMM_WORLD, &status); //Receive
	}
	
	// Free MPI datatype (no longer needed)
	MPI_Type_free(&cv_contig_Vec3b);
	

	
	//--------------------
	// ALL RANK OPERATIONS
	//--------------------
	
	
	// Send image descriptors back
	
	
	// Send partial image back
	
	
	
	//--------------
	// Process Image
	//--------------
	
	if (!image.data) 
	{
		//cout << "Rank " << rank << " could not open or ﬁnd the image." << endl;
	}
	else 
	{
		// Print image info
		cout << "Image dimension = " << image.rows << " x " << image.cols << endl;
		cout << "Image type = " << image.type() << endl;
		
		// Create clone of image to work on
		cv::Mat image_clone(image.rows, image.cols, image.type());

		// Copy original image to clone
		assert(image.type() == CV_8UC3); //these types of images have pixels stored as cv::Vec3b
		for (int irow = 0; irow < image.rows; irow++) 
		{ 
			for (int icol = 0; icol < image.cols; icol++) 
			{ 
				image_clone.at<cv::Vec3b>(irow,icol) = image.at<cv::Vec3b>(irow,icol);
			} 
		} 
		
		// Write orignal image
		cv::imwrite("./Original.jpg",image_clone);
		
		// Write processed image based on process selection
		cv::Mat processedImage;
		switch(imageProcessID){
			case IMAGE_PROCESS_TYPE_MIN:
			
			break;
			
			case IMAGE_PROCESS_TYPE_BLUR:
				imageBlur(image_clone, processedImage, blurLevel, 0, image_clone.rows); // Blur
				cv::imwrite("./blur.jpg",processedImage);
				break;
			
			case IMAGE_PROCESS_TYPE_ENHANCE:
				imageContrastEnhance(image_clone, processedImage, enhanceMin, enhanceMax); // Contrast
				cv::imwrite("./enhance.jpg",processedImage);
				break;
			case IMAGE_PROCESS_TYPE_GREYSCALE:
				imageGreyScale(image_clone, processedImage); // Greyscale
				cv::imwrite("./greyscale.jpg",processedImage);
				break;
			
			case IMAGE_PROCESS_TYPE_MAX:
			
			break;	
		}
	}
	
	// Clean up MPI
	MPI_Finalize(); 
}