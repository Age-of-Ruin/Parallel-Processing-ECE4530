#include <iostream>
#include <fstream>					// Needed for 1st and 2nd attempt (ofstream out_to_file)
#include <sstream>					// Needed for 2nd attempt (ostringstream fileNameStream)
#include <mpi.h>					// Use C-version of mpi, include mpi++.h for C++ bindings.

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
	
	ofstream out_to_file;								// You'll need an include for this (need <fstream> class)
	
	//Open the output file HERE using a unique name based on rank and nproc
	//Hint: you can use ostringstream to produce the filename as a combination of 
	//numbers and strings pretty easily (you'll need an include)

	// Open file with unique name (1st attempt)
/* 	int MAX_FILE_NAME_SIZE = 20;				// Declare max file name
	char fileName[MAX_FILE_NAME_SIZE];			// Declare character array for file name
	sprintf(fileName, "P%dfile.txt", rank);		// Create unique name based on rank
	out_to_file.open(fileName);					// Open file with unique file name */
	
	// Open file with unique name (2nd attempt)
	ostringstream fileNameStream;					// Declare output string stream
	fileNameStream << "P" << rank << "file.txt";	// Push name into string stream
	out_to_file.open(fileNameStream.str());		// Create file using file name
	
	// Print output to file and close
	out_to_file << "My name is " << processor_name << " and I am processor " << rank << " of " << nproc << std::endl;
	out_to_file.close();
									
	MPI_Finalize(); // Finalize MPI

	return 0;
		
}	