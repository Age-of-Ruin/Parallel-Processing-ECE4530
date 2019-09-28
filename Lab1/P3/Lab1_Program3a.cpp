#include <iostream>
#include <mpi.h>					//Use C-version of mpi, include mpi++.h for C++ bindings.
#include <math.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <string.h>

using namespace std;

int main(int argc, char** argv)
{
	int rank = 0;										// Declare rank			
	int nproc = 1;										// Declare number of processors
	int namelength;										// Declare name length
	char processor_name[MPI_MAX_PROCESSOR_NAME];		// Declare character array to hold processor name
	
	MPI_Init(&argc, &argv);								// Initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);				// Get the number of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);				// Get my rank
	MPI_Get_processor_name(processor_name,&namelength);	// Get processor name	

	int nvalues = 0;									// Declare int to hold selected number of random values

	if (rank == 0)
	{
		// Retrieve (from user) how many random numbers to generate 
		std::cout << "Enter the number of random values that each processor should generate " << std::endl;
		std::cin >> nvalues;
		
		//Inform all processors of the number of values it should provide
		MPI_Bcast(&nvalues, 1, MPI_INT, rank, MPI_COMM_WORLD);		
	}
	else
	{
		// Send broadcast back to processor 0 so it can continue
		MPI_Bcast(&nvalues, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// Generate random number based on seed calculated from rank and number of processors and store in vector
	srand(rank + nproc); 
	vector<double> values(nvalues);
	
	for (int ival = 0; ival < nvalues; ival++)
	{
		values[ival] = rand();
	}

	//Open and Output Values To File Here. Use a newline delimeter.
	
	// Read File based on rank (1st attempt)
/* 	string line									// Declare string to hold line
	int MAX_FILE_NAME_SIZE = 20;				// Declare max file name
	char fileName[MAX_FILE_NAME_SIZE];			// Declare character array for file name
	sprintf(fileName, "P%dfile.txt", rank);		// Create unique name based on rank
	ifstream in_from_file(fileName);			// Open file with unique file name */
	
	// Read File based on rank (2nd attempt)
	string line;									// Declare string to hold line
	ostringstream filenamestream;					// Declare string stream to hold file name
	filenamestream << "./P" << rank << "file.txt";	// Push name to string stream
	ifstream in_from_file(filenamestream.str());	// Read file based on file name
	
	// Print file contents to std out
/* 	if(in_from_file.is_open()) {
		while(getline(in_from_file, line))
			cout << line << endl;
		in_from_file.close();
	} */
	
	// Read file contents into string
	string fileString;							// Declare string to hold each line read from file
	if(in_from_file.is_open()) {				// Check that file is open
		while(getline(in_from_file, line))		// Loop over entire file and capture each line as a string
			fileString = fileString + line;		// Concatenate all lines into 1 big string
		in_from_file.close();					// Close files
	}
	
	//cout << fileString << endl;		// Print file string
	
	// Convert string to character array
	char fileArray[fileString.size() + 1];
	strcpy(fileArray, fileString.c_str());
	
	//cout << fileArray << endl;			// Print character array
	
    
	// Compute local processor sum
	long localSum = 0;
	for (int i = 0; i < nvalues; i++){
		localSum += values[i];
	}
	
	usleep(10000*rank); // Format output by sleeping based on rank
	
	cout << "Local sum for processor " << rank << " is " << localSum << endl; // Print local sum
	
	// Compute global sum
	long globalSum = 0;
	MPI_Allreduce(&localSum, &globalSum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	
	usleep(10000*rank); // Format output by sleeping based on rank
	
	cout << "Global sum for processor " << rank << " is " << globalSum << endl; // Print global sum
    
    MPI_Finalize(); // Finalize MPI
    
	return 0;
		
}	