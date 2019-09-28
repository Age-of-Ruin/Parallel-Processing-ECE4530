#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <mpi.h>

#define MAX_SIZE 1024

using namespace std;

// Ever faithful parallelrange 
void  parallelRange (int globalstart, int globalstop , int irank, int nproc, int &localstart, int &localstop, int &localcount)
{
	int nrows = globalstop - globalstart  + 1;
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

int main(int argc, char** argv)
{	

	//-----------------------
	// Initialize MPI
	//-----------------------
	int rank = 0;						
	int nproc = 1;		
	
	MPI_Init(&argc, &argv);								
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Structure to hold each processors data
	vector <vector <unsigned char>> localRows;
	
	//-----------------------
	// Convert Command Line
	//-----------------------
				
	int ny = atoi(argv[1]);
	int nx = atoi(argv[2]);
	int maxiter = atoi(argv[3]);
	
	assert(argc == 4);
	assert(ny <= MAX_SIZE);
	assert(nx <= MAX_SIZE);
	
	// Create Solution space on rank 0
	if (rank == 0)
	{	
		//---------------
		// Generate the 
		// initial image 
		//---------------
		
		vector <vector <unsigned char>> solutionSpace(ny); 
		srand(clock());
		for (unsigned int iy = 0; iy < ny; iy++)
		{
			solutionSpace[iy].resize(nx);
			
			for (unsigned int ix = 0; ix < nx; ix++)
			{
				//seed a 1/2 density of alive (just arbitrary really)
				int state = rand()%2;
				if (state == 0) solutionSpace[iy][ix] = 255; //dead
				else solutionSpace[iy][ix] = 0; //alive
			}
		}
		
		//------------
		// Partition
		//-----------
		
		// Determine local rows for each processor
		vector <int> localStart(nproc), localStop(nproc), localCount(nproc);
		for (int irank = 0; irank < nproc; irank++){
			
			parallelRange(0, ny-1, irank, nproc, localStart[irank], localStop[irank], localCount[irank]);
			
			// Add extra rows where neccessary 
			if (irank == 0){
				localStop[irank]++;
				localCount[irank]++;
			}
			else if(irank == nproc-1)
			{
				localStart[irank]--;
				localCount[irank]++;
			}
			else
			{
				localStart[irank]--;
				localStop[irank]++;
				localCount[irank]+=2;	
			}
			
			//cout << "Proc " << irank << " will be sent " << localCount[irank] << " rows and "  << localCount[irank]*nx << " values" << endl;
		}
		
		//-------
		// Send
		//-------
		
		// Store rank 0 rows first
		localRows.resize(localCount[0]);
		for (int irow = localStart[0]; irow <= localStop[0]; irow++)
		{
			localRows[irow] = solutionSpace[irow];
		}
		
		// Send rows to processors 1...nproc
		vector <unsigned char> unrolledVals;
		int count = 0;
		for (int irank = 1; irank < nproc; irank++)
		{
			// Unroll 2D array into 1D vector
			unrolledVals.resize(localCount[irank]*nx);
			count = 0;
			for (int irow = localStart[irank]; irow <= localStop[irank]; irow++)
			{
				for (int icol = 0; icol < nx; icol++)
				{
					unrolledVals[count] = solutionSpace[irow][icol];
					count++;
				}
			}
			
			// Send these unrolled values
			MPI_Send(&unrolledVals[0], unrolledVals.size(), MPI_UNSIGNED_CHAR, irank , 99, MPI_COMM_WORLD);
		}
	} // End of rank 0 operations
	
	
	//----------------
	// Receive on
	// all processors
	// (except rank 0)
	//----------------
	
	int recvCount, numRows, count;
	vector <unsigned char> recvVals;
	MPI_Status status;
	if (rank != 0){

		MPI_Probe(0, 99, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &recvCount);  
		
		recvVals.resize(recvCount);
		
		MPI_Recv(&recvVals[0], recvCount, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &status);
	
		// Reconstruct 2d array of local rows from unrolled vals
		numRows = recvCount/nx;
		localRows.resize(numRows);
		count = 0;
		for (int irow = 0; irow < numRows; irow++)
		{
			localRows[irow].resize(nx);
			for (int icol = 0; icol < nx; icol++)
			{			
				localRows[irow][icol] = recvVals[count];
				count++;
			}			
		}
		
		cout << "Proc " << rank << " has received " << numRows << " rows and "  << recvCount << " values" << endl;
	} // End of receive operations
	
	
	
	//------------
	// Reconstruct 
	// image
	//------------
	
	// Unroll 2D matrix into 1D vector
	vector <unsigned char> unrolledVals;
	unrolledVals.resize(localRows.size()*nx);
	count = 0;
	for (int irow = 0; irow < localRows.size(); irow++)
	{
		for (int icol = 0; icol < nx; icol++)
		{
			unrolledVals[count] = localRows[irow][icol];
			count++;
		}
	}
	
	// Send approriate rows to rank 0 (ie exclude halo rows)
	if (rank != 0 && rank != nproc-1)
	{
		MPI_Send(&unrolledVals[nx], unrolledVals.size()-2*nx, MPI_UNSIGNED_CHAR, 0, 102, MPI_COMM_WORLD);
	}
	else if (rank == nproc-1)
	{
		MPI_Send(&unrolledVals[nx], unrolledVals.size()-nx, MPI_UNSIGNED_CHAR, 0, 102, MPI_COMM_WORLD);
	}
	
	// Reconstruct image on rank 0
	if (rank == 0)
	{				
		// Place all unrolled values into vector
		std::vector <unsigned char> allUnrolledVals(nx*ny);
		int valCount = 0;
		while (valCount < unrolledVals.size()-nx)
		{
			allUnrolledVals[valCount] = unrolledVals[valCount];
			valCount++;
		}
		
		// Receive from ranks 1 to nproc-1
		for (int irank = 1; irank < nproc; irank++)
		{
			MPI_Probe(irank, 102, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &recvCount); 

			MPI_Recv(&allUnrolledVals[valCount], recvCount, MPI_UNSIGNED_CHAR, irank, 102, MPI_COMM_WORLD, &status);
			
			valCount += recvCount;
			
			cout << "Rank 0 received " << recvCount << " from rank " << irank << endl;
		}
		
		// Reconstruct 2D image/matrix
		cv::Mat population(ny, nx, CV_8UC1);
		valCount = 0;
		for (int irow = 0; irow < ny; irow++)
		{
			for (int icol = 0; icol < nx; icol++)
			{
				population.at<uchar>(irow,icol) = allUnrolledVals[valCount];
				valCount++;
			}
		}
		
		// Resize image
		cv::Mat image_to_view(MAX_SIZE,MAX_SIZE,CV_8UC1);
		cv::resize(population,image_to_view,image_to_view.size(), cv::INTER_LINEAR);
		
		// Rename image
		ostringstream fNameStream;
		fNameStream << "Population_" << 0 << ".jpg";
		string fName = fNameStream.str();
		
		// Output image		
		cv::imwrite(fName, image_to_view);
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	// cv::Mat population(ny, nx, CV_8UC1);
	// cv::Mat image_for_viewing(MAX_SIZE,MAX_SIZE,CV_8UC1);
	// cv::Mat newPopulation;
	
	//-----------------------
	// Game of Life 
	// (on all processors)
	//-------------------------
	
//	for (int iter = 0; iter < maxiter; iter++)
//	{
//		// Rename image
//		ostringstream fNameStream;
//		fNameStream << "Population_" << iter << ".jpg";
//		string fName = fNameStream.str();
//		
//		// Resize and output population matrix to image		
//		cv::resize(population,image_for_viewing,image_for_viewing.size(),cv::INTER_LINEAR);
//		cv::imwrite(fName, image_for_viewing);
//		
//		// Loop over all points in matrix
//		for (int iy = 0; iy < ny; iy++)
//		{
//			for (int ix = 0; ix < nx; ix++)
//			{
//				
//				// Find (and count) any alive neighbors 
//				int occupied_neighbours = 0;
//
//				for (int jy = iy - 1; jy <= iy + 1; jy++)
//				{
//					for (int jx = ix - 1; jx <= ix + 1; jx++)
//					{
//						if (jx == ix && jy == iy) continue;
//						
//						int row = jy;
//						if (row == ny) row = 0;
//						if (row == -1) row = ny-1;
//						
//						int col = jx;
//						if (col == nx) col = 0;
//						if (col == -1) col = nx - 1;
//						
//						if (population.at<uchar>(row,col) == 0) occupied_neighbours++;
//					}
//				}
//			
//				// Update each element according to the number alive neighbors
//				if (population.at<uchar>(iy,ix) == 0)   // update alive elements
//				{
//					if (occupied_neighbours <= 1 || occupied_neighbours >= 4) newPopulation.at<uchar>(iy,ix) = 255; // dies (over-crowding)
//					if (occupied_neighbours == 2 || occupied_neighbours == 3) newPopulation.at<uchar>(iy,ix) = 0; // stays alive (perfect amount of neighbors)
//				}
//				else if (population.at<uchar>(iy,ix) == 255) // update dead elements
//				{
//					if (occupied_neighbours == 3)
//					{
//						newPopulation.at<uchar>(iy,ix) = 0; // reproduction
//					}
//				}
//			}
//		}
//		
//		// Copy updated population for next iteration
//		population = newPopulation.clone(); 
//	}
	
	// Clean up MPI and exit
	MPI_Finalize();
	exit(0);
}