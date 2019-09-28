#include "Utilities.h"
#include "TreeNode.h"
#include <random> 
#include <unistd.h>

using namespace std;

int main(int argc, char** argv)
{
	
	//-----------
    // MPI Setup
    //-----------

    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	
	//-------------------
    // Parse Command Line
    //-------------------
    
    if (argc < 3)
    {
        if (rank == 0) 
			std::cerr << "Must specify: 1) numPoints per processor and 2)error parameter (theta)." << std::endl;
        
		MPI_Finalize();
        exit(1);
    }
	
	int numProcPoints = atoi(argv[1]); // number of points per processor
	double theta = strtod(argv[2], NULL); // theta aka error control aka when to use COM
	
	// Print input info
	if (rank == 0)
	{
		cout << "\n***** INPUT PARAMETERS *****" << endl;
		cout << "numProcPoints = " << numProcPoints << endl;
		cout << "theta = " << theta << endl; 
		if (argv[3] != NULL) cout << "Direct Method Comparison" << endl;
	}
	
	
	//-----------------
    // Timing Variables
    //-----------------
	
	double directStartTime, directStopTime, directElapsedTime;
	double serialStartTime, serialStopTime, serialElapsedTime;
	double parallelStartTime, parallelStopTime, parallelElapsedTime;
	double partitionStartTime, partitionStopTime, partitionElapsedTime;
	double LETStartTime, LETStopTime, LETElapsedTime;
	double forceStartTime, forceStopTime, forceElapsedTime;
	
	
	//----------------
	// Force Variables
	//----------------
	
	double forceMag;
	double localForceSum;
	double localForceMin;
	double localForceMax;
	double localForceAvg;
	double globalForceSum;
	double globalForceMin;
	double globalForceMax;
	double globalForceAvg;
	
	//-------------------------------------------
	// Generate random points (in 3d) within unit 
	// cube with random weight 1kg to 1000kg
	//------------------------------------------
	
	// Print - diagnostics
	// usleep(10000*rank);
	// cout << "Proc " << rank << endl;
	
	srand(100*rank + time(NULL));
		
	int weightMax = 1000;
	int weightMin = 1;
	std::vector <body_t> localBodies(numProcPoints);
	for (unsigned int ibody = 0; ibody < numProcPoints; ibody++)
	{
		// Generate points with random x, y, z, and mass
		localBodies[ibody].r[0] = (double)rand()/(double)RAND_MAX;
		localBodies[ibody].r[1] = (double)rand()/(double)RAND_MAX;
		localBodies[ibody].r[2] = (double)rand()/(double)RAND_MAX;
		localBodies[ibody].m = (double)(rand()%(weightMax-weightMin)+1);
		
		// Print - diagnostics
		// cout << "Body " << ibody << endl;
		// cout << "x = " << localBodies[ibody].r[0] << "\ny = " << localBodies[ibody].r[1] << "\nz = " << localBodies[ibody].r[2] <<  "\nm = " << localBodies[ibody].m << endl;
	}

	
	//***********************
	// Serial Direct Solution 
	//***********************
		
	if (argv[3] != NULL && nproc == 1)
	{
		// Start Time
		directStartTime = clock();
		
		vector <double3_t> directForces (localBodies.size());
		for (int ibody = 0; ibody < localBodies.size(); ibody++)
		{
			directForces[ibody].r[0] = 0.0;
			directForces[ibody].r[1] = 0.0;
			directForces[ibody].r[2] = 0.0;
			for (int iforce = 0; iforce < directForces.size(); iforce++)
			{
				if (ibody == iforce) continue;
				
				double Rx = localBodies[ibody].r[0]-localBodies[iforce].r[0];
				double Ry = localBodies[ibody].r[1]-localBodies[iforce].r[1];
				double Rz = localBodies[ibody].r[2]-localBodies[iforce].r[2];
				double R = sqrt(Rx*Rx + Ry*Ry + Rz*Rz);
				directForces[ibody].r[0] += -G*localBodies[ibody].m*localBodies[iforce].m*Rx/(R*R*R);
				directForces[ibody].r[1] += -G*localBodies[ibody].m*localBodies[iforce].m*Ry/(R*R*R);
				directForces[ibody].r[2] += -G*localBodies[ibody].m*localBodies[iforce].m*Rz/(R*R*R);
			}
		}
		
		forceMag = 0;
		localForceSum = 0;
		localForceAvg = 0;
		localForceMin = DBL_MAX;
		localForceMax = DBL_MIN;
		for (int ibody = 0; ibody < directForces.size(); ibody++)
		{
			forceMag = sqrt(directForces[ibody].r[0]*directForces[ibody].r[0] + directForces[ibody].r[1]*directForces[ibody].r[1] + directForces[ibody].r[2]*directForces[ibody].r[2]);
			
			// Print - diagnostics
			//cout << "Force Mag " << forceMag << endl;
			
			localForceSum += forceMag;
			if (forceMag < localForceMin) localForceMin = forceMag;
			if (forceMag > localForceMax) localForceMax = forceMag;
		}
		
		localForceAvg = localForceSum/directForces.size();
		
		// Stop Time
		directStopTime = clock();
		directElapsedTime = (directStopTime-directStartTime)/CLOCKS_PER_SEC;
		
		// Print solution results
		cout << "\n****** SERIAL DIRECT *******" << endl;

		cout << "Average Force\t\t: " << localForceAvg << " Newtons" << endl;
		cout << "Minimum Force\t\t: " << localForceMin << " Newtons" << endl;
		cout << "Maximum Force\t\t: " << localForceMax << " Newtons" << endl;
		
		cout << "\nDirect Elapsed Time\t: " << directElapsedTime << " seconds" << endl;
	}
	
	
	//*******************
	// Serial BH Solution
	//*******************
	
	if (nproc == 1)
	{
		// Start serial timing
		serialStartTime = clock();
		
		// ALREADY HAVE ALL BODIES NECCESSARY FOR LET (ie localtree is Local Essential Tree (LET))
		// NO PARTITIONING REQUIRED
		// NOW CONSTRUCT LET
		
		// Start LET Timing
		LETStartTime = clock();
		
		// Acquire global extent across all domains
		domain_t globalDomain;
		double globalDimMin, globalDimMax;
		getPointExtent(localBodies, globalDomain, globalDimMin, globalDimMax, 1);
		
		// Construct localtree/LET using global extent
		TreeNode localRoot(globalDomain.min[0], globalDomain.max[0], globalDomain.min[1], globalDomain.max[1], globalDomain.min[2], globalDomain.max[2]);
		for (int ibody = 0; ibody < localBodies.size(); ibody++)
		{
			localRoot.addBody(localBodies[ibody]);
		}
		
		// Recursively compute COM of all LET nodes
		localRoot.computeCoM();
		
		// Stop LET Timing
		LETStopTime = clock();
	
		// Force calculation start time
		forceStartTime = clock();
	
		// Compute forces over all local (now LET) bodies
		vector <double3_t> serialForces (localBodies.size());
		for (int ibody = 0; ibody < localBodies.size(); ibody++)
		{
			localRoot.computeForceOnBody(localBodies[ibody], theta, serialForces[ibody]);
		}
		
		// Calculate magnitude and find average, min and max force
		forceMag = 0;
		localForceSum = 0;
		localForceAvg = 0;
		localForceMin = DBL_MAX;
		localForceMax = DBL_MIN;
		for (int ibody = 0; ibody < serialForces.size(); ibody++)
		{
			forceMag = sqrt(serialForces[ibody].r[0]*serialForces[ibody].r[0] + serialForces[ibody].r[1]*serialForces[ibody].r[1] + serialForces[ibody].r[2]*serialForces[ibody].r[2]);
			
			// Print - diagnostics
			//cout << "Force Mag " << forceMag << endl;
			
			localForceSum += forceMag;
			if (forceMag < localForceMin) localForceMin = forceMag;
			if (forceMag > localForceMax) localForceMax = forceMag;
		}
		
		localForceAvg = localForceSum/serialForces.size();
		
		// Force calculation stop time
		forceStopTime = clock();
	
		// Stop serial timing
		serialStopTime = clock();
		
		// Determine elapsed times
		serialElapsedTime = (serialStopTime-serialStartTime)/CLOCKS_PER_SEC;
		LETElapsedTime = (LETStopTime-LETStartTime)/CLOCKS_PER_SEC;
		forceElapsedTime = (forceStopTime-forceStartTime)/CLOCKS_PER_SEC;
		
		// Print solution results
		cout << "\n******** SERIAL BH *********" << endl;
		cout << "Average Force\t\t: " << localForceAvg << " Newtons" << endl;
		cout << "Minimum Force\t\t: " << localForceMin << " Newtons" << endl;
		cout << "Maximum Force\t\t: " << localForceMax << " Newtons" << endl;
		
		parallelElapsedTime = (parallelStopTime-parallelStartTime)/CLOCKS_PER_SEC;
		partitionElapsedTime = (partitionStopTime-partitionStartTime)/CLOCKS_PER_SEC;
		LETElapsedTime = (LETStopTime-LETStartTime)/CLOCKS_PER_SEC;
		forceElapsedTime = (forceStopTime-forceStartTime)/CLOCKS_PER_SEC;
	
		cout << "\nLET Construction Time\t: " << LETElapsedTime << " seconds" << endl;
		cout << "Force Calculation Time\t: " << forceElapsedTime << " seconds" << endl;
		cout << "Serial Elapsed Time\t: " << serialElapsedTime << " seconds" << endl;
		cout << endl;
		
		// Exit Program
		MPI_Finalize();
		exit(0);
	}
		
	
	//*********************
	// Parallel BH Solution
	//*********************
	
	// Start overall parallel timing
	parallelStartTime = clock();
	
	//--------------------------------
	// Partition bodies (in space) and 
	// send to appropriate processor 
	//---------------------------------
	
	// Start partition timing
	partitionStartTime = clock();
	
	// Get indices of elements destined for each processor (using ORB)
    std::vector <std::vector<int>>  indices_for_each_proc;
	ORB(nproc, localBodies, indices_for_each_proc);
	
	// Convert indices to bodies/points
	std::vector <std::vector <body_t>> bodies_to_be_swapped(nproc);
	for (int irank = 0; irank < nproc; irank++)
	{
		bodies_to_be_swapped[irank].resize(indices_for_each_proc[irank].size());

		for (int ibody = 0; ibody < indices_for_each_proc[irank].size(); ibody++)
		{
			bodies_to_be_swapped[irank][ibody] = localBodies[indices_for_each_proc[irank][ibody]];
		}
	}
	
	// Send appropriate bodies to each processor
	MPI_Alltoall_vecvecT(bodies_to_be_swapped, bodies_to_be_swapped);
	
	// Stop partition timing
	partitionStopTime = clock();
	
	// Print - diagnostics
	// usleep(10000*rank);
	// if (rank == 0) cout << "\n************After Partition*****************\n";
	// cout << "Proc " << rank << endl;
	
	// Overwrite current list of bodies with new/partitioned list of elements
	// int bodyCount = 0;
	localBodies.resize(0);
	for (int irank = 0; irank < nproc; irank++)
	{		
		for (int ibody = 0; ibody < bodies_to_be_swapped[irank].size(); ibody++)
		{
			localBodies.push_back(bodies_to_be_swapped[irank][ibody]);
			
			// Print - diagnostics
			// cout << "Body " << bodyCount << endl;
			// cout << "x = " << localBodies[bodyCount].r[0] << "\ny = " << localBodies[bodyCount].r[1] << "\nz = " << localBodies[bodyCount].r[2] <<  "\nm = " << localBodies[bodyCount].m << endl;
			// bodyCount++;
		}	
	}
	
	
	//---------------------
	// Build the local tree 
	//---------------------
	
	// Start LET Timing
	LETStartTime = clock();
	
	// Acquire global extent across all domains
	domain_t globalDomain;
	double globalDimMin, globalDimMax;
	getPointExtent(localBodies, globalDomain, globalDimMin, globalDimMax, 1);
	
	// Construct local tree using global extent
	TreeNode localRoot(globalDomain.min[0], globalDomain.max[0], globalDomain.min[1], globalDomain.max[1], globalDomain.min[2], globalDomain.max[2]);
	for (int ibody = 0; ibody < localBodies.size(); ibody++)
	{
		localRoot.addBody(localBodies[ibody]);
	}
	
	// Recursively compute COM of all localtree nodes
	localRoot.computeCoM();
	
	//------------
	// DIAGNOSTICS
	//------------
	
	// int numBodies = 0;
	// int numNodes = 0;
	// int maxLvl = 0;
	
	// MPI_Barrier(MPI_COMM_WORLD);
	// usleep(100000*rank);
	// cout << "\n*******Diagnostics (Local Tree)********" << endl;
	
	// body_t localCOM;
	// localRoot.getCoM(localCOM);
	// cout << "Proc " << rank << " has COM " << localCOM.m << endl;
	
	// localRoot.diagnostics(0, maxLvl, numBodies, numNodes);
	// cout << "Proc " << rank << " has " << numBodies << " bodies and " << numNodes << " nodes within " << maxLvl << " levels" << endl;
	
	
	//-------------------------------------
	// Obtain bodies from other processors
	// neccessary for force calcuations
	// and put into local tree (ie build LET)
	//-------------------------------------
	
	// Aqcuire local extent across local/partitioned domain
	domain_t localDomain;
	double localDimMin, localDimMax;
	getPointExtent(localBodies, localDomain, localDimMin, localDimMax, 0);
	
	// Communicate LOCAL extent to other processors
	vector <vector <domain_t>> allDomains(nproc);
	for (int irank = 0; irank < nproc; irank++)
	{
		allDomains[irank].resize(1);
		
		if (irank != rank)
			allDomains[irank][0] = localDomain;
	}
	
	MPI_Alltoall_vecvecT(allDomains, allDomains);
	
	// Find bodies that need to be sent to other processors (and send)
	vector <vector <body_t>> bodies_to_swap(nproc);
	for(int irank = 0; irank < nproc; irank++)
	{	
		bodies_to_swap[irank].resize(0);
		
		if (irank != rank)
		{
			localRoot.LETBodies(allDomains[irank][0], theta, bodies_to_swap[irank]);
		}
	}
	
	MPI_Alltoall_vecvecT(bodies_to_swap, bodies_to_swap);
	
	// Add bodies from other ranks to LET
	for (int irank = 0; irank < nproc; irank++)
	{
		if(irank != rank)
		{
			for (int ibody = 0; ibody < bodies_to_swap[irank].size(); ibody++)
			{
				localRoot.addBody(bodies_to_swap[irank][ibody]);
			}
		}
	}
	
	// Recursively compute COM of all LET nodes
	localRoot.computeCoM();
	
	// Stop LET Timing
	LETStopTime = clock();
	
	//-----------------------------------------
	// DIAGNOSTICS (print COM and see if equal)
	//-----------------------------------------
	
	// MPI_Barrier(MPI_COMM_WORLD);
	// usleep(100000*rank);
	// cout << "\n***********Diagnostics (LET)************" << endl;
	
	// localRoot.getCoM(localCOM);
	// cout << "Proc " << rank << " has COM x = " << localCOM.r[0] << " y = " << localCOM.r[1] << " z = " << localCOM.r[2] << endl;
	
	// maxLvl = 0; numBodies = 0; numNodes = 0;
	// localRoot.diagnostics(0, maxLvl, numBodies, numNodes);
	// cout << "Proc " << rank << " has " << numBodies << " bodies and " << numNodes << " nodes within " << maxLvl << " levels" << endl;
	
	
	//-------------------------------
	// Compute FORCE on local bodies
	//-------------------------------
	
	// Force calculation start time
	forceStartTime = clock();
	
	// Compute forces over all local bodies using LET 
	vector <double3_t> parallelForces (localBodies.size());
	for (int ibody = 0; ibody < localBodies.size(); ibody++)
	{
		localRoot.computeForceOnBody(localBodies[ibody], theta, parallelForces[ibody]);
	}
	
	
	//---------------------------
	// Determine avg, min and max 
	// force over all processors
	//---------------------------
	
	// Calculate magnitude and find sum, min and max force
	forceMag = 0;
	localForceSum = 0;
	localForceMin = DBL_MAX;
	localForceMax = DBL_MIN;
	for (int ibody = 0; ibody < parallelForces.size(); ibody++)
	{
		forceMag = sqrt(parallelForces[ibody].r[0]*parallelForces[ibody].r[0] + parallelForces[ibody].r[1]*parallelForces[ibody].r[1] + parallelForces[ibody].r[2]*parallelForces[ibody].r[2]);
		
		// Print - diagnostics
		//cout << "Force Mag " << forceMag << endl;
		
		localForceSum += forceMag;
		if (forceMag < localForceMin) localForceMin = forceMag;
		if (forceMag > localForceMax) localForceMax = forceMag;
	}

	// REDUCE ALL VALUES ON RANK 0
	// Find Global average
	globalForceSum = 0;
	globalForceAvg = 0;
	double localNumBodies = (double)localBodies.size();
	double globalNumBodies = 0;
	MPI_Reduce(&localForceSum, &globalForceSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&localNumBodies, &globalNumBodies, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	globalForceAvg = globalForceSum/globalNumBodies;
	
	// Find Global min
	globalForceMin = 0;
	MPI_Reduce(&localForceMin, &globalForceMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	
	// Find Global max
	globalForceMax = 0;
	MPI_Reduce(&localForceMax, &globalForceMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	// Force stop timing
	forceStopTime = clock();
	
	// Final Timing Results
	parallelStopTime = clock();
		
	// Print PARALLEL force and time results
	if (rank == 0) {
		cout << "\n****** PARALLEL BH *******" << endl;
		cout << "Average Force\t\t: " << globalForceAvg << " Newtons" << endl;
		cout << "Minimum Force\t\t: " << globalForceMin << " Newtons" << endl;
		cout << "Maximum Force\t\t: " << globalForceMax << " Newtons" << endl;
		
		parallelElapsedTime = (parallelStopTime-parallelStartTime)/CLOCKS_PER_SEC;
		partitionElapsedTime = (partitionStopTime-partitionStartTime)/CLOCKS_PER_SEC;
		LETElapsedTime = (LETStopTime-LETStartTime)/CLOCKS_PER_SEC;
		forceElapsedTime = (forceStopTime-forceStartTime)/CLOCKS_PER_SEC;
		
		cout << "\nPartition Time\t\t: " << partitionElapsedTime << " seconds" << endl;
		cout << "LET Construction Time\t: " << LETElapsedTime << " seconds" << endl;
		cout << "Force Calculation Time\t: " << forceElapsedTime << " seconds" << endl;
		cout << "Parallel Elapsed Time\t: " << parallelElapsedTime << " seconds" << endl;
		cout << endl;
	}
	
	//*************************
	// Parallel Direct Solution 
	//*************************
		
	if (argv[3] != NULL && nproc > 1)
	{
		// Start Time
		directStartTime = clock();
		
		vector <vector <body_t>> allBodies (nproc);
		for (int irank = 0; irank < nproc; irank++)
		{
			if (irank != rank)
				allBodies[irank] = localBodies;	
		}
		
		MPI_Alltoall_vecvecT(allBodies, allBodies);
		
		// Loop over local bodies
		vector <double3_t> directForces (localBodies.size());
		for (int ibody = 0; ibody < localBodies.size(); ibody++)
		{
			directForces[ibody].r[0] = 0.0;
			directForces[ibody].r[1] = 0.0;
			directForces[ibody].r[2] = 0.0;
			for (int irank = 0; irank < nproc; irank++)
			{
				for (int iOtherBody = 0; iOtherBody < allBodies[irank].size(); iOtherBody++)
				{
					if (irank == rank) continue;
					
					double Rx = localBodies[ibody].r[0]-allBodies[irank][iOtherBody].r[0];
					double Ry = localBodies[ibody].r[1]-allBodies[irank][iOtherBody].r[1];
					double Rz = localBodies[ibody].r[2]-allBodies[irank][iOtherBody].r[2];
					double R = sqrt(Rx*Rx + Ry*Ry + Rz*Rz);
					directForces[ibody].r[0] += -G*localBodies[ibody].m*allBodies[irank][iOtherBody].m*Rx/(R*R*R);
					directForces[ibody].r[1] += -G*localBodies[ibody].m*allBodies[irank][iOtherBody].m*Ry/(R*R*R);
					directForces[ibody].r[2] += -G*localBodies[ibody].m*allBodies[irank][iOtherBody].m*Rz/(R*R*R);
				}
			}
		}
		
		forceMag = 0;
		localForceSum = 0;
		localForceAvg = 0;
		localForceMin = DBL_MAX;
		localForceMax = DBL_MIN;
		for (int ibody = 0; ibody < directForces.size(); ibody++)
		{
			forceMag = sqrt(directForces[ibody].r[0]*directForces[ibody].r[0] + directForces[ibody].r[1]*directForces[ibody].r[1] + directForces[ibody].r[2]*directForces[ibody].r[2]);
			
			// Print - diagnostics
			//cout << "Force Mag " << forceMag << endl;
			
			localForceSum += forceMag;
			if (forceMag < localForceMin) localForceMin = forceMag;
			if (forceMag > localForceMax) localForceMax = forceMag;
		}
		
		localForceAvg = localForceSum/directForces.size();
		
		// Communicate results if neccessary - REDUCE ON RANK 0
		if (nproc > 1)
		{
			globalForceSum = 0;
			globalForceAvg = 0;
			localNumBodies = (double)localBodies.size();
			globalNumBodies = 0;
			MPI_Reduce(&localForceSum, &globalForceSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&localNumBodies, &globalNumBodies, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			
			globalForceAvg = globalForceSum/globalNumBodies;
			
			// Find Global min
			globalForceMin = 0;
			MPI_Reduce(&localForceMin, &globalForceMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
			
			// Find Global max
			globalForceMax = 0;
			MPI_Reduce(&localForceMax, &globalForceMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		}
		
		// Stop Time
		directStopTime = clock();
		directElapsedTime = (directStopTime-directStartTime)/CLOCKS_PER_SEC;
		
		// Print solution results
		if (rank == 0)
		{
			cout << "\n***** DIRECT PARALLEL ******" << endl;

			cout << "Average Force\t\t: " << globalForceAvg << " Newtons" << endl;
			cout << "Minimum Force\t\t: " << globalForceMin << " Newtons" << endl;
			cout << "Maximum Force\t\t: " << globalForceMax << " Newtons" << endl;
		
			cout << "\nDirect Elapsed Time\t: " << directElapsedTime << " seconds" << endl;
		}
	}
	
	
	// Cleanup & exit
	MPI_Finalize();
	exit(0);
	
} // End Main