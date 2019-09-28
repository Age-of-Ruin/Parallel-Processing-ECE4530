#include <iostream>
#include <vector>
#include <mpi.h>
#include <string>
#include <algorithm>
#include "./Mesh.h"
#include <random>
#include <unistd.h>

int main(int argc, char** argv)
{
    //----------------------------
    // MPI Setup
    //----------------------------

    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
	
    //----------------------------
    // Parse Command Line
    //----------------------------
    
    if (argc < 2)
    {
        if (rank == 0) std::cerr << "You must specify a .msh gmsh file name as a command line parameter" << std::endl;
        MPI_Finalize();
        exit(1);
    }
    string filename = argv[1];
    
    //---------------------------------------------------
    // Instantiate Mesh - Performs Read and Partitioning
    //---------------------------------------------------

    Mesh pmesh(filename);
	
	
	//------------------------
	// Write the original mesh
	//------------------------
	
    std::vector<double> element_ranks(pmesh.getElement3DCount(),rank);
	pmesh.writeMesh("./MeshBefore", "Ranks", element_ranks);
	
	// Time Statistics
	double startTime = 0;
	double stopTime = 0;
	double elapsedTime = 0;
	
	// Print Statistics
	if (rank == 0)
	{
		std::cout << "\n***** Input Mesh *****" << std::endl;
    }
	
	pmesh.outputStatistics();
	
	//----------
	// Partition
	//----------
	
	// Synchnronize
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Partition
	startTime = clock();
	pmesh.partitionMesh();
	stopTime = clock();
	
	// Synchnronize & print time statistics
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(10000*rank);
	

	elapsedTime = (stopTime-startTime)/CLOCKS_PER_SEC;
	std::cout << "\nRank " << rank << " took " << elapsedTime << " seconds to partition." << std::endl;
	
    //---------------------------------------------------
    // Write Mesh - You will want to assign a value of
    // rank to every element in the mesh.
    // You can use Mesh::getElement3DCount() to obtain
    // the number of 3D elements in the mesh.
    //---------------------------------------------------
	
	element_ranks.resize(pmesh.getElement3DCount(),rank);
    pmesh.writeMesh("./MeshAfter", "Ranks", element_ranks);
	
	// Print Statistics
	if (rank == 0)
	{
		cout << "\n***** Output Mesh *****" << std::endl;
	}
	
	pmesh.outputStatistics();
    
	
	// Finalize and exit
    MPI_Finalize();
    
    return 0;
}

