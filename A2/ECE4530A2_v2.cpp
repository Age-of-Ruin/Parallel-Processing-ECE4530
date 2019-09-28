#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> //for resizing
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <mpi.h>

#define MAX_SIZE 1024

using namespace std;

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
	
	// Create Solution space on r
	if (rank == 0)
	{
	
		//-----------------------
		// Convert Command Line
		//-----------------------
					
		int ny = atoi(argv[1]);
		int nx = atoi(argv[2]);
		int maxiter = atoi(argv[3]);
		
		assert(argc == 4);
		assert(ny <= MAX_SIZE);
		assert(nx <= MAX_SIZE);
		
		//-----------------------
		// Generate the 
		// initial image 
		//----------------------
		
		cv::Mat population(ny, nx, CV_8UC1);
		cv::Mat image_for_viewing(MAX_SIZE,MAX_SIZE,CV_8UC1);
		cv::Mat newPopulation;

		srand(clock());
		for (unsigned int iy = 0; iy < ny; iy++)
		{
			for (unsigned int ix = 0; ix < nx; ix++)
			{
				//seed a 1/2 density of alive (just arbitrary really)
				int state = rand()%2;
				if (state == 0) population.at<uchar>(iy,ix) = 255; //dead
				else population.at<uchar>(iy,ix) = 0;   //alive
			}
		}
		newPopulation = population.clone();
		
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
			
			cout << "Proc " << irank << " will take " << localCount[irank] << " rows" << endl;
		}
		
			//-------
			// Send
			//-------
			
	//		// Store rank 0 rows first
	//		localRows.resize(localCount[0]);
	//		for (int irow = localStart[0]; irow <= localStop[0]; irow++)
	//		{
	//			localRows[irow] = T[irow];
	//		}
	//		
	//		// Send rows to processors 1...nproc
	//		std::vector <double> unrolledVals;
	//		int count = 0;
	//		for (int irank = 1; irank < nproc; irank++)
	//		{
	//			// Unroll 2D array into 1D vector
	//			unrolledVals.resize(localCount[irank]*Nx);
	//			count = 0;
	//			for (int irow = localStart[irank]; irow <= localStop[irank]; irow++)
	//			{
	//				for (int icol = 0; icol < Nx; icol++)
	//				{
	//					unrolledVals[count] = T[irow][icol];
	//					count++;
	//				}
	//			}
	//			
	//			// Send these unrolled values
	//			MPI_Send(&unrolledVals[0], unrolledVals.size(), MPI_DOUBLE, irank , 99, MPI_COMM_WORLD);
	//		}
	
	
	//-----------------------
	// Game of Life 
	// (on all processors)
	//-------------------------
	
		for (int iter = 0; iter < maxiter; iter++)
		{
			// Rename image
			ostringstream fNameStream;
			fNameStream << "Population_" << iter << ".jpg";
			string fName = fNameStream.str();
			
			// Resize and output population matrix to image		
			cv::resize(population,image_for_viewing,image_for_viewing.size(),cv::INTER_LINEAR);
			cv::imwrite(fName, image_for_viewing);
			
			// Loop over all points in matrix
			for (int iy = 0; iy < ny; iy++)
			{
				for (int ix = 0; ix < nx; ix++)
				{
					
					// Find (and count) any alive neighbors 
					int occupied_neighbours = 0;

					for (int jy = iy - 1; jy <= iy + 1; jy++)
					{
						for (int jx = ix - 1; jx <= ix + 1; jx++)
						{
							if (jx == ix && jy == iy) continue;
							
							int row = jy;
							if (row == ny) row = 0;
							if (row == -1) row = ny-1;
							
							int col = jx;
							if (col == nx) col = 0;
							if (col == -1) col = nx - 1;
							
							if (population.at<uchar>(row,col) == 0) occupied_neighbours++;
						}
					}
				
					// Update each element according to the number alive neighbors
					if (population.at<uchar>(iy,ix) == 0)   // update alive elements
					{
						if (occupied_neighbours <= 1 || occupied_neighbours >= 4) newPopulation.at<uchar>(iy,ix) = 255; // dies (over-crowding)
						if (occupied_neighbours == 2 || occupied_neighbours == 3) newPopulation.at<uchar>(iy,ix) = 0; // stays alive (perfect amount of neighbors)
					}
					else if (population.at<uchar>(iy,ix) == 255) // update dead elements
					{
						if (occupied_neighbours == 3)
						{
							newPopulation.at<uchar>(iy,ix) = 0; // reproduction
						}
					}
				}
			}
			
			// Copy updated population for next iteration
			population = newPopulation.clone(); 
		}
	}
	
	// Clean up MPI and exit
	MPI_Finalize();
	exit(0);
}