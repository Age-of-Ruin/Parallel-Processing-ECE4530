#include <iostream>
#ifdef HAVEMPI		// Only include mpi.h if defined when executed
#include <mpi.h>
#endif

using namespace std;

int main(int argc, char** argv)
{
	int rank = 0;	// Declare rank		
	int nproc = 1;	// Declare number of processors	
	int namelength; // Delcare name length
	
	
	// Parallel code (dependant on MPI - mpi.h must be included)
	#ifdef HAVEMPI
		char processor_name[MPI_MAX_PROCESSOR_NAME];		// Declare character array for processor name
		
		MPI_Init(&argc, &argv);								// Initialize MPI
		MPI_Comm_size(MPI_COMM_WORLD, &nproc);				// Get the number of processors
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);				// Get my rank
		MPI_Get_processor_name(processor_name,&namelength);	// Get processor name
		
		// Output for parallel execution
		std::cout << "Hello, my name is " << processor_name << " and I am number " << rank <<  " of " << nproc << std::endl;
		
		MPI_Finalize();	// Finalize MPI
		
	// Serial Code (do not include mpi.h)
	#else
		// Output for serial execution
		std::cout << "Hello from processor " << rank <<  " of " << nproc << std::endl;
	#endif
	
	return 0;
		
}	