#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include <string>

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
	int rowCount;
	int rowStart;
	int rowStop;
	int colCount;
	int imgProcessID;
	int contrastMin;
	int contrastMax;
	int blurLvl;
	int haloRowStart;
	int haloRowStop;
	int haloRowCount;
	
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
	
	// Create contiguous data type for 1 pixel  - need 3 bytes (1 per channel)
	MPI_Datatype cv_contig_Vec3b;
	MPI_Type_contiguous(3, MPI_BYTE, &cv_contig_Vec3b);
	MPI_Type_commit(&cv_contig_Vec3b);
	
	// Declare an image descriptor for each processor
	image_descriptor_t imgD;
	
	// Holds pixel data on each processor
	vector <cv::Vec3b> procPixels;
	
	//------------------
	// RANK 0 OPERATIONS
	//------------------
	
	if(rank == 0){
		
		// Used to parse & read image
		cv::Mat image; 
		string imageName;
		string imageProcess;
		
		// Used to enumerate process and hold parameters 
		int imageProcessID = 0;	
		int blurLevel = 0;
		int enhanceMin = 0;
		int enhanceMax = 255;
		
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
				cout << "Arguments are missing - terminating program." << endl;
				exit(1);
			} 
		}
		else
		{
			cout << "Arguments are missing - terminating program." << endl;
			exit(1);
		} 
	
		cout << "Rank " << rank << " reading image " << imageName << " and performing " << imageProcess << "." << endl;
		
		// Read image file (in color) - assumes picture in current directory
		string imageLocation = "./" + imageName;
		image = cv::imread(imageLocation,1);
		
		// Print image info
		cout << "Image dimension = " << image.rows << " x " << image.cols << endl;
		cout << "Image type = " << image.type() << endl;
		
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
			imgDs[irank].colCount = image.cols;
			imgDs[irank].imgProcessID = imageProcessID;
			
			if (imageProcessID == IMAGE_PROCESS_TYPE_ENHANCE)
			{
				imgDs[irank].contrastMin = enhanceMin;
				imgDs[irank].contrastMax = enhanceMax;
			}
			
			if (imageProcessID == IMAGE_PROCESS_TYPE_BLUR)
			{
				imgDs[irank].blurLvl = blurLevel;
				
				// Determine rows to send (for halo rows)
				if (localstart[irank] - (2 * blurLevel + 1) < 0)
					imgDs[irank].haloRowStart = 0;
				else
					imgDs[irank].haloRowStart = localstart[irank] - (2 * blurLevel + 1);

				if (localstop[irank] + (2 * blurLevel + 1) >= image.rows)
					imgDs[irank].haloRowStop = image.rows-1;
				else
					imgDs[irank].haloRowStop = localstop[irank] + (2 * blurLevel + 1);
			
				imgDs[irank].haloRowCount = imgDs[irank].haloRowStop - imgDs[irank].haloRowStart + 1;
			
				// Print stuff
				//cout << localcount[irank] << " " << localstart[irank] << " " << localstop[irank] << endl;
				
				//cout << imgDs[irank].haloRowCount << " " << imgDs[irank].haloRowStart << " " << imgDs[irank].haloRowStop << endl;
			
			}

		}
		
		// Set rank 0 image descriptor
		imgD = imgDs[0];
		
		// Send image descriptor to processors 1..n
        for (int irank = 1; irank < nproc; irank++)
        {
            MPI_Send(&imgDs[irank], sizeof(imgDs[irank]), MPI_BYTE, irank, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD);
        }
		
		
		// Extract all pixels
		long pixelCount = 0;
		vector<cv::Vec3b> allPixels(image.rows * image.cols);
		for (int irow = 0; irow < image.rows; irow++) 
		{ 
			for (int icol = 0; icol < image.cols; icol++) 
			{ 
				allPixels[pixelCount] = image.at<cv::Vec3b>(irow,icol);
				pixelCount++;
			}  
		}
		
		//--------------
		// SEND ROUTINES
		//--------------

		if (imageProcessID == IMAGE_PROCESS_TYPE_BLUR)
		{
/* 			// Determine number of pixels per processor
			for (int irank = 0; irank < nproc; irank++)
			{
				procPixelCounts[irank] = imgDs[irank].haloRowCount * image.cols;
			}
				
			// Store pixel data for rank 0
			procPixels.resize(procPixelCounts[0]);
			for (long i = 0; i < procPixelCounts[0]; i++)
			{
				procPixels[i] = allPixels[i];
			}

			// Send pixels to other processors 1..n using cv_contig_Vec3b
			long procPixelIndex = imgDs[0].rowCount * image.cols;
			for (int irank = 1; irank < nproc; irank++)
			{
				MPI_Send(&allPixels[procPixelIndex], procPixelCounts[irank], cv_contig_Vec3b, irank, PIXEL_DATA_TAG, MPI_COMM_WORLD);
				
				//cout << procPixelIndex << endl;
				
				procPixelIndex += imgDs[irank].rowCount * image.cols;
			} */
		
		}
		else // ENHANCE or GREYSCALE 
		{			
			// Store pixel data for rank 0
			procPixels.resize(imgDs[0].rowCount * imgDs[0].colCount);
			for (long i = 0; i < imgDs[0].rowCount * imgDs[0].colCount; i++)
			{
				procPixels[i] = allPixels[i];
			}

			// Send pixels to other processors 1..n using cv_contig_Vec3b
			long procPixelIndex = imgDs[0].rowCount * imgDs[0].colCount;
			for (int irank = 1; irank < nproc; irank++)
			{
				MPI_Send(&allPixels[procPixelIndex], imgDs[irank].rowCount * imgDs[irank].colCount, cv_contig_Vec3b, irank, PIXEL_DATA_TAG, MPI_COMM_WORLD);
				
				procPixelIndex += imgDs[irank].rowCount * imgDs[irank].colCount;
			}
		}
	} // End of Rank 0 Operations

	
	//-------------------
	// RECEIVE OPERATIONS
	//-------------------
	
	// Receive image descriptors
	if (rank != 0)
	{
        MPI_Status status;
        MPI_Probe(0, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD, &status);
		
		MPI_Recv(&imgD, sizeof(imgD), MPI_BYTE, 0, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD, &status); 
	}	
	
	// Receive pixel data
	if (rank != 0)
	{
		int recv_count;
        MPI_Status status;
        MPI_Probe(0, PIXEL_DATA_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, cv_contig_Vec3b, &recv_count);  
		
		procPixels.resize(recv_count);
		
		MPI_Recv(&procPixels[0], recv_count, cv_contig_Vec3b, 0, PIXEL_DATA_TAG, MPI_COMM_WORLD, &status);
	}

	
	//--------------------
	// ALL RANK OPERATIONS
	//--------------------
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Create cv::Mats (matrices) used for processing
	cv::Mat procImageIn(imgD.rowCount, imgD.colCount, CV_8UC3);
	cv::Mat procImageOut(imgD.rowCount, imgD.colCount, CV_8UC3);
	
	// Copy pixels into cv::Mat (matrix)
	long pixelCount = 0;
	for (int irow = 0; irow < imgD.rowCount; irow++) 
	{ 
		for (int icol = 0; icol < imgD.colCount; icol++) 
		{ 
			procImageIn.at<cv::Vec3b>(irow,icol) = procPixels[pixelCount];
			pixelCount++;
		} 
	} 

	// Check if images have data
	if (!procImageIn.data) 
	{
		cout << "Rank " << rank << " could not open or ﬁnd the image." << endl;
	}
	else 
	{
		// Process Image
		ostringstream os;
		string fileName;
		switch(imgD.imgProcessID){
			case IMAGE_PROCESS_TYPE_MIN:
			
			break;
			
			case IMAGE_PROCESS_TYPE_BLUR:
				//imageBlur(procImageIn, procImageOut, imgD.blurLvl, imgD.rowStart, imgD.rowStop);
				os << "./blur" << rank << ".jpg";
				fileName = os.str();
				//cv::imwrite(fileName,procImageOut);
				break;
			
			case IMAGE_PROCESS_TYPE_ENHANCE:
				imageContrastEnhance(procImageIn, procImageOut, imgD.contrastMin, imgD.contrastMax);
				os << "./enhance" << rank << ".jpg";
				fileName = os.str();
				cv::imwrite(fileName,procImageOut);
				break;
			
			case IMAGE_PROCESS_TYPE_GREYSCALE:
				imageGreyScale(procImageIn, procImageOut);
				os << "./greyscale" << rank << ".jpg";
				fileName = os.str();
				cv::imwrite(fileName,procImageOut);
				break;
			
			case IMAGE_PROCESS_TYPE_MAX:
			
			break;	
		}
			
		// Send image descriptors to rank 0
		if (rank != 0) 
		{
			MPI_Send(&imgD, sizeof(imgD), MPI_BYTE, 0, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD);
		}
		
		// Build vector of pixels
		long pixelCount = 0;
		for (int irow = 0; irow < procImageOut.rows; irow++) 
		{ 
			for (int icol = 0; icol < procImageOut.cols; icol++) 
			{ 
				procPixels[pixelCount] = procImageOut.at<cv::Vec3b>(irow,icol);
				pixelCount++;
			} 
		} 
		
		// Send pixel data to rank 0
		if (rank != 0) 
		{
			// Send pixels to other processors n using cv_contig_Vec3b
			MPI_Send(&procPixels[0], imgD.rowCount * imgD.colCount, cv_contig_Vec3b, 0, PIXEL_DATA_TAG, MPI_COMM_WORLD);			
		}
	}
	
	
	//------------------
	// RANK 0 OPERATIONS
	//------------------
	
	if (rank == 0)
	{		
		// Receive image descriptor and acquire size
		MPI_Status status;
		vector <image_descriptor_t> imgDs(nproc);
		imgDs[0] = imgD;
		long numPixels = imgDs[0].rowCount * imgDs[0].colCount;
		int numRows = imgDs[0].rowCount;
		for (int irank = 1; irank < nproc; irank ++)
		{
			MPI_Recv(&imgDs[irank], sizeof(imgDs[irank]), MPI_BYTE, irank, PIXEL_DESCRIPTOR_TAG, MPI_COMM_WORLD, &status);	

			numPixels += imgDs[irank].rowCount * imgDs[irank].colCount;
			numRows += imgDs[irank].rowCount;
		}	
		
		
		// Store pixel data for rank 0
		vector <cv::Vec3b> allPixels(numPixels);
		for (long i = 0; i < imgDs[0].rowCount * imgDs[0].colCount; i++)
		{
			allPixels[i] = procPixels[i];
		}
		
		// Receive processed image
		long procPixelIndex = imgDs[0].rowCount * imgDs[0].colCount;
		for (int irank = 1; irank < nproc; irank ++)
		{			
			MPI_Recv(&allPixels[procPixelIndex], procPixelIndex, cv_contig_Vec3b, irank, PIXEL_DATA_TAG, MPI_COMM_WORLD, &status);
			
			procPixelIndex += imgDs[irank].rowCount * imgDs[irank].colCount;
		
		}
		
		// Rebuild Image from procImages
		long pixelCount = 0;
		cv::Mat finalImage(numRows, imgDs[0].colCount, CV_8UC3);
		for (int irow = 0; irow < numRows; irow++) 
		{ 
			for (int icol = 0; icol < imgDs[0].colCount; icol++) 
			{ 
				finalImage.at<cv::Vec3b>(irow,icol) = allPixels[pixelCount];
				pixelCount++;
			} 
		} 
		
		// Print final image
		cv::imwrite("./finalImage.jpg", finalImage);
	}
	 
	 
	// Clean up MPI
	MPI_Type_free(&cv_contig_Vec3b);
	MPI_Barrier(MPI_COMM_WORLD);
	cout << "Rank = " << rank << " - Terminating. " << std::endl;
	MPI_Finalize(); 
}