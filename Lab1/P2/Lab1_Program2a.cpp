#include <iostream>
#include <mpi.h>					// Use C-version of mpi, include mpi++.h for C++ bindings.

using namespace std;

int main(int argc, char** argv)
{
	int rank = 0;										// Declare rank		
	int nproc = 1;										// Declare number of processors
	int namelength;										// Delcare name length
	char processor_name[MPI_MAX_PROCESSOR_NAME];		// Declare character array for processor name
	
	MPI_Init(&argc, &argv);								// Initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);				// Get the number of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);				// Get my rank
	MPI_Get_processor_name(processor_name,&namelength); // Get processor name
	
	if (rank == 0)
	{
		// Print number of processors and rank 0's output
		std::cout << "Number of processors is " << nproc << std::endl;
		std::cout << "Node " << rank << " is processor " << processor_name << std::endl;
		
        
		char host[MPI_MAX_PROCESSOR_NAME]; // Declare 
		
		for (int irank = 1; irank < nproc; irank++)
		{
			// Rank 0 receives names from processors 1 to nproc-1
			MPI_Status status;
			MPI_Recv(&host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, irank, 1, MPI_COMM_WORLD, &status);
			
			// Output each name received
			std::cout << "Node " << irank << " is processor " << host << std::endl;
		}
	}
	else
	{
		// Every process (with rank 1 to nproc-1) sends its name to rank 0
		MPI_Send(&processor_name, namelength, MPI_CHAR, 0 , 1, MPI_COMM_WORLD);
	}
	
				
	MPI_Finalize(); // Finalize MPI

	return 0;
		
}	