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
		
	//-----------------------
	// Convert Command Line
	//-----------------------
				
	int ny = atoi(argv[1]);
	int nx = atoi(argv[2]);
	int maxiter = atoi(argv[3]);
	
	assert(argc >= 4);
	assert(ny <= MAX_SIZE);
	assert(nx <= MAX_SIZE);
	
	// Timing Variables
	double startPTime, stopPTime, elapsedPTime;
	double startSTime, stopSTime, elapsedSTime;

	// Structure to hold each processors data
	vector <vector <unsigned char>> localRows;
	
	// Create Solution space on rank 0
	if (rank == 0)
	{	
		//---------------
		// Generate the 
		// initial image 
		//---------------
		
		vector <vector <unsigned char>> solutionSpace(ny); // Holds entire initial population/matrix/solution
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
		
		//-----------------------
		// Serial Comparision
		//-----------------------
		
		// Include extra argument for serial comparison
		if (argv[4] != NULL || nproc == 1)
		{
			
			//-----------------------
			// Generate the initial 
			// image 
			//----------------------
			
			cv::Mat population(ny, nx, CV_8UC1);
			cv::Mat image_for_viewing(MAX_SIZE,MAX_SIZE,CV_8UC1);
			cv::Mat newPopulation;
			
			for (unsigned int iy = 0; iy < ny; iy++)
			{
				for (unsigned int ix = 0; ix < nx; ix++)
				{
					population.at<uchar>(iy,ix) = solutionSpace[iy][ix]; // Copy solution space
				}
			}
			newPopulation = population.clone();
				
			// Start serial timing
			startSTime = clock();
				
			for (int iter = 0; iter < maxiter; iter++)
			{
				// Rename image
				ostringstream fNameStream;
				fNameStream << "Population_serial_" << iter << ".jpg";
				string fName = fNameStream.str();
				
				// Resize and output population matrix to image		
				cv::resize(population,image_for_viewing,image_for_viewing.size(),cv::INTER_LINEAR);
				cv::imwrite(fName, image_for_viewing);
				
				for (int iy = 0; iy < ny; iy++)
				{
					for (int ix = 0; ix < nx; ix++)
					{
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
					
						if (population.at<uchar>(iy,ix) == 0)   //alive
						{
							if (occupied_neighbours <= 1 || occupied_neighbours >= 4) newPopulation.at<uchar>(iy,ix) = 255; //dies
							if (occupied_neighbours == 2 || occupied_neighbours == 3) newPopulation.at<uchar>(iy,ix) = 0; //same as population
						}
						else if (population.at<uchar>(iy,ix) == 255) //dead
						{
							if (occupied_neighbours == 3)
							{
								newPopulation.at<uchar>(iy,ix) = 0; //reproduction
							}
						}
					}
				}
				population = newPopulation.clone(); // Copy new population
			}
			
			// Finalize timing results
			stopSTime = clock();
			elapsedSTime = (stopSTime-startSTime)/CLOCKS_PER_SEC;
			cout << "Total Elapsed Time (Serial): " << elapsedSTime << " seconds" << endl;
			
			// Exit if only 1 processor
			if (nproc == 1)
			{
				MPI_Finalize();
				exit(0);
			}
		} // End of serial comparison
		
		
		//-----------------------
		//  Parallel
		//-----------------------
		
		// Start parallel timing
		startPTime = clock();
		
		//------------
		// Partition
		//-----------
		
		// Determine local rows for each processor
		vector <int> localStart(nproc), localStop(nproc), localCount(nproc);
		for (int irank = 0; irank < nproc; irank++){
			
			parallelRange(0, ny-1, irank, nproc, localStart[irank], localStop[irank], localCount[irank]);
			
			// Add extra rows to accomodate halo rows 
			localStart[irank]--;
			localStop[irank]++;
			localCount[irank]+=2;	
			
			//cout << "Proc " << irank << " will be sent " << localCount[irank] << " rows and "  << localCount[irank]*nx << " values" << endl;
		}
		
		//-------
		// Send
		//-------
		
		// Store rank 0 rows first
		int startRow = localStart[0]; // local row start
		int stopRow = localStop[0]; // local row stop
		int rowCount = localCount[0]; // local row count
		int count = 0; // counter
		
		// Resize rows
		localRows.resize(localCount[0]);
		
		// Create rows local to rank 0
		for (int irow = startRow+1; irow < rowCount; irow++)
		{				
			// Resize columns
			localRows[irow].resize(nx);
	

			for (int icol = 0; icol < nx; icol++)
			{
				// Store global last row in 1st row of rank 0
				if (irow == 0)
					localRows[irow][icol] = solutionSpace[ny-1][icol];
				
				// Store other elements (including 1 'middle' halo row)
				else
					localRows[irow][icol] = solutionSpace[irow-1][icol];
			}
		}

		//cout << "Proc 0 stored " << localRows.size()*localRows[localRows.size()-1].size() << " values" << endl;
		
		// Send rows to processors 1...nproc
		vector <unsigned char> unrolledVals;
		for (int irank = 1; irank < nproc; irank++)
		{
			// Update start and stop values & send vector size
			unrolledVals.resize(localCount[irank]*nx);
			startRow = localStart[irank];
			stopRow = localStop[irank];
			rowCount = localCount[irank];
			
	
			// Unroll 2D array into 1D vector
			count = 0;
			for (int irow = startRow; irow <= stopRow; irow++)
			{
				for (int icol = 0; icol < nx; icol++)
				{
					// Adjust for first halo row (ie store global 1st row in last row of nproc-1)		
					if (irow == stopRow && irank == nproc-1)
						unrolledVals[count] = solutionSpace[0][icol];
					
					// Store other elements (including 1 'middle' halo row)
					else
						unrolledVals[count] = solutionSpace[irow][icol];
					
					count++;
				}
			}
			
			// Send these unrolled values
			MPI_Send(&unrolledVals[0], unrolledVals.size(), MPI_UNSIGNED_CHAR, irank , 99, MPI_COMM_WORLD);
			
			//cout << "Proc " << irank << " was sent " << unrolledVals.size() << " values" << endl;
		}
	} // End of rank 0 operations
	
	
	//----------------
	// Receive on
	// all processors
	// (except rank 0)
	//----------------
	
	int recvCount, numRows, count; // Count variables
	vector <unsigned char> recvVals; // Receive buffer
	MPI_Status status; // MPI status structure
	if (rank != 0){

		// Probe for receive size
		MPI_Probe(0, 99, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &recvCount);  
		
		// Resize the receive buffer
		recvVals.resize(recvCount);
		
		// Receive the values
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
		
		//cout << "Proc " << rank << " has received " << numRows << " rows and "  << recvCount << " values" << endl;
	} // End of receive operations
	
	
	
	// Print each partial image from each processor
	
//		MPI_Barrier(MPI_COMM_WORLD);
//		
//		// Matrix, image and vector structures
//		cv::Mat population1(localRows.size(), nx, CV_8UC1, 255);
//		cv::Mat image_to_view1(MAX_SIZE,MAX_SIZE,CV_8UC1);
//		vector <unsigned char> unrolledVals1(localRows.size()*nx);
//		
//		// Reconstruct 2D image/matrix
//		for (int irow = 0; irow < localRows.size(); irow++)
//		{
//			for (int icol = 0; icol < nx; icol++)
//			{
//				population1.at<uchar>(irow,icol) = localRows[irow][icol];
//			}
//		}
//		
//		// Resize image
//		cv::resize(population1,image_to_view1,image_to_view1.size(), cv::INTER_LINEAR);
//		
//		// Rename image
//		ostringstream fNameStream;
//		fNameStream << "Population_rank_" << rank << ".jpg";
//		string fName = fNameStream.str();
//		
//		// Output image		
//		cv::imwrite(fName, image_to_view1);
	

	
	//-----------------------
	// Game of Life 
	// (on all processors)
	//-------------------------
	
	// Matrix, image and vector structures
	vector <vector <unsigned char>> newLocalRows(localRows.size()); // holds updated localRows after iteration/computation
	for (int irow = 0; irow < newLocalRows.size(); irow++) // resize this matrix to same size as localRows
	{
		newLocalRows[irow].resize(nx);
		
		// Copy original solution into clone
		for (int icol = 0; icol < nx; icol++)
		{
			newLocalRows[irow][icol] = localRows[irow][icol];
		}
	}
	
	cv::Mat population(ny, nx, CV_8UC1, 255); // holds entire population matrix
	cv::Mat image_to_view(MAX_SIZE,MAX_SIZE,CV_8UC1); // holds image scaled to MAX_SIZE x MAX_SIZE
	vector <unsigned char> unrolledVals((localRows.size()-2)*nx); // holds local rows in 1D vector
	vector <unsigned char> allUnrolledVals(ny*nx); // holds entire population in 1D vector
	
	for (int iter = 0; iter < maxiter; iter++)
	{
		//------------
		// Reconstruct 
		// image (on
		// rank 0)
		//------------
		
		// Unroll 2D matrix into 1D vector & EXCLUDE HALO ROWS (ie 1st and last)
		count = 0;
		for (int irow = 1; irow < localRows.size()-1; irow++)
		{
			for (int icol = 0; icol < nx; icol++)
			{
				unrolledVals[count] = localRows[irow][icol];
				count++;
			}
		}
		
		// Send these unrolled values to rank 0 
		if (rank != 0)
			MPI_Send(&unrolledVals[0], unrolledVals.size(), MPI_UNSIGNED_CHAR, 0, 102, MPI_COMM_WORLD);
		
		// Reconstruct image on rank 0
		if (rank == 0)
		{				
			// Place all unrolled values into vector
			// Start with rank 0
			int valCount = 0;
			while (valCount < unrolledVals.size())
			{
				allUnrolledVals[valCount] = unrolledVals[valCount];
				valCount++;
			}
			
			// Receive from ranks 1 to nproc-1 and place into same vector
			for (int irank = 1; irank < nproc; irank++)
			{
				MPI_Probe(irank, 102, MPI_COMM_WORLD, &status);
				MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &recvCount); 

				MPI_Recv(&allUnrolledVals[valCount], recvCount, MPI_UNSIGNED_CHAR, irank, 102, MPI_COMM_WORLD, &status);
				
				valCount += recvCount;
				
				//cout << "Rank 0 received " << recvCount << " from rank " << irank << endl;
			}
			
			// Reconstruct 2D image/matrix using this vector
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
			cv::resize(population,image_to_view,image_to_view.size(), cv::INTER_LINEAR);
			
			// Rename image
			ostringstream fNameStream;
			fNameStream << "Population_parallel_" << iter << ".jpg";
			string fName = fNameStream.str();
			
			// Output image		
			cv::imwrite(fName, image_to_view);
		}

		//------------
		// Perform 
		// updates
		//------------		
		
		// Loop over all local points in matrix EXCLUDE HALO ROWS
		for (int iy = 1; iy < localRows.size()-1; iy++)
		{
			for (int ix = 0; ix < nx; ix++)
			{	
				// Find (and count) any alive neighbors 
				int occupied_neighbours = 0;

				// Loop over all neighbors
				for (int jy = iy - 1; jy <= iy + 1; jy++)
				{
					for (int jx = ix - 1; jx <= ix + 1; jx++)
					{
						// Skip current point
						if (jx == ix && jy == iy) continue;
						
						// Wrap edge cases (only wrap columns - rows are handled via halo rows)		
						int row = jy;
						int col = jx;
						if (col == nx) col = 0;
						if (col == -1) col = nx - 1;
						
						// Count the neighbor if it is alive
						if (localRows[row][col] == 0) occupied_neighbours++;
					}
				}
			
				// Update each element according to the number alive neighbors
				if (localRows[iy][ix] == 0)   // update alive elements
				{
					if (occupied_neighbours <= 1 || occupied_neighbours >= 4) newLocalRows[iy][ix] = 255; // dies (over-crowding/starvation)
					if (occupied_neighbours == 2 || occupied_neighbours == 3) newLocalRows[iy][ix] = 0; // stays alive (perfect amount of neighbors)
				}
				else if (localRows[iy][ix] == 255) // update dead elements
				{
					if (occupied_neighbours == 3)
					{
						newLocalRows[iy][ix] = 0; // reproduction
					}
				}
			}
		} 

		// Store new configuration
		for (int irow = 0; irow < newLocalRows.size(); irow++)
		{
			for (int icol = 0; icol < nx; icol++)
			{
				localRows[irow][icol] = newLocalRows[irow][icol];
			}
		}

		// Synchronize Halo Rows under new configuraion
		
		// INNER HALO ROWS
		// Replace first row of Pi+1 (halo row) with 2nd last row of Pi (ie send from Pi to Pi+1)
		if (rank != nproc-1)
			MPI_Send(&localRows[localRows.size()-2][0], nx, MPI_UNSIGNED_CHAR, rank+1, 100, MPI_COMM_WORLD);
		
		if (rank != 0)
			MPI_Recv(&localRows[0][0], nx, MPI_UNSIGNED_CHAR, rank-1, 100, MPI_COMM_WORLD, &status);
		
		// Replace last row of Pi (halo row) with second row of Pi+1 (ie send from Pi+1 to Pi)
		if (rank != 0)
			MPI_Send(&localRows[1][0], nx, MPI_UNSIGNED_CHAR, rank-1, 101, MPI_COMM_WORLD);
		
		if (rank != nproc-1)
			MPI_Recv(&localRows[localRows.size()-1][0], nx, MPI_UNSIGNED_CHAR, rank+1, 101, MPI_COMM_WORLD, &status);
		
		
		// OUTER HALO ROWS (IE WRAP EDGES)
		// Send 2nd last row of nproc-1 to 1st row of rank 0
		// Send 2nd row of rank 0 to last row of nproc-1
		if (rank == nproc-1)
		{
			MPI_Send(&localRows[localRows.size()-2][0], nx, MPI_UNSIGNED_CHAR, 0, 100, MPI_COMM_WORLD);
			MPI_Recv(&localRows[localRows.size()-1][0], nx, MPI_UNSIGNED_CHAR, 0, 101, MPI_COMM_WORLD, &status);
		}
		if (rank == 0)
		{
			MPI_Recv(&localRows[0][0], nx, MPI_UNSIGNED_CHAR, nproc-1, 100, MPI_COMM_WORLD, &status);
			MPI_Send(&localRows[1][0], nx, MPI_UNSIGNED_CHAR, nproc-1, 101, MPI_COMM_WORLD);
		}

	} // End of parllel processing/iterations
	
	// Finalize and print timing results
	if (rank == 0)
	{
		stopPTime = clock();
		elapsedPTime = (stopPTime - startPTime)/CLOCKS_PER_SEC;
		cout << "Total Elapsed Time (Parallel): " << elapsedPTime << " seconds" << endl;
	}
	
	// Clean up MPI and exit
	MPI_Finalize();
	exit(0);
}