#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>

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
	int rowCount = 0;
	int rowStart = 0;
	int rowStop = 0;
	//int process_type;
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
	for (unsigned int irow = 0; irow < out.rows; irow++) 
	{ 
		// Loop over all cols
		for (unsigned int icol = 0; icol < out.cols; icol++) 
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
	for (unsigned int irow = 0; irow < out.rows; irow++) 
	{ 
		// Loop over all cols
		for (unsigned int icol = 0; icol < out.cols; icol++) 
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
	for (unsigned int irow = rowstart; irow < rowstop; irow++) 
	{ 
		// Loop over all cols
		for (unsigned int icol = 0; icol < out.cols; icol++) 
		{ 
			
			// Bounds
			unsigned int blurRowStart = 0;
			unsigned int blurRowStop = 0;
			unsigned int blurColStart = 0;
			unsigned int blurColStop = 0;
			
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
				for (unsigned int blurRowIndex = blurRowStart; blurRowIndex <= blurRowStop; blurRowIndex++)
				{
					for (unsigned int blurColIndex = blurColStart; blurColIndex <= blurColStop; blurColIndex++)
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
	MPI_Status status;
	
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
	
	// Used to hold single image desriptor
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
		for (unsigned int irank = 0; irank < nproc; irank++)
		{
			parallelRange(0, image.rows, irank, nproc, localstart[irank], localstop[irank], localcount[irank]);
		}
		
		// Create image descriptor for each processor
		vector <image_descriptor_t> imgDs(nproc);
		for (unsigned int irank = 0; irank < nproc; irank++)
		{
			imgDs[irank].rowStart = localstart[irank];
			imgDs[irank].rowStop = localstop[irank];
			imgDs[irank].rowCount = localcount[irank];
		}
		
		// Set rank 0 image descriptor
		imgD = imgDs[0];
		
		// Send relevant image descriptor to all processors (except 0 - already has it)
        for (int irank = 1; irank < nproc; irank++)
        {
            MPI_Send(&imgDs[irank], imgDs.size()*sizeof(imgDs[irank]), MPI_BYTE, irank, 99, MPI_COMM_WORLD);
        }
		
	
	
	
	}
	
	
	
	
	
	
	
	//--------------------
	// ALL RANK OPERATIONS
	//--------------------
	
	
	
	// Recieve
	if (rank != 0)
	{
		int recv_count;
        MPI_Status status;
        MPI_Probe(0, 99, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &recv_count);
        
		MPI_Recv(&imgD, recv_count, MPI_BYTE, 0, 99, MPI_COMM_WORLD, &status); //Receive
	}
	
	
	cout << "Rank " << rank << " will process this many rows " << imgD.rowCount << endl;
	

	// Output image (if it exists)
	if (!image.data) 
	{
		cout << "Rank " << rank << " could not open or ﬁnd the image." << endl;
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
		for (unsigned int irow = 0; irow < image.rows; irow++) 
		{ 
			for (unsigned int icol = 0; icol < image.cols; icol++) 
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