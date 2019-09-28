#include <vector>
#include <iostream>
#include <math.h>
#include <cassert>
#include <mpi.h>
#include <queue>
#include <algorithm>
#include <float.h>
#include <unistd.h>
#include "./MeshUtilities.h"


//-----------------------------------------------------
// Usual Parallel Partitioning Code
//-----------------------------------------------------
void parallelRange(int globalstart, int globalstop, int irank, int nproc, int& localstart, int& localstop, int& localcount)
{
	int nvals = globalstop - globalstart + 1;
	int divisor = nvals/nproc;
	int remainder = nvals%nproc;
	int offset;
	if (irank < remainder) offset = irank;
	else offset = remainder;
    
	localstart = irank*divisor + globalstart + offset;
	localstop = localstart + divisor - 1;
	if (remainder > irank) localstop += 1;
	localcount = localstop - localstart + 1;
}

//-----------------------------------------------------
// Extract a single coordinate list from a double3_t list
//-----------------------------------------------------
std::vector<double> getSingleCoordinateListFromPoints(const std::vector<double3_t>& points, int dim)
{
    std::vector<double> coordinate_list(points.size());
    if (dim > 3 || dim < 0)
    {
        std::cerr << "Requested dimension " << dim << " is out of bounds at line " << __LINE__ << " of file " << __FILE__ << " for function " << __FUNCTION__ << std::endl;
        return coordinate_list;
    }
    
    for (unsigned int ipoint = 0; ipoint < points.size(); ipoint++)
    {
        coordinate_list[ipoint] = points[ipoint].r[dim];
    }
    
    //Want to see the list?
    /*
    for (unsigned int ipoint = 0; ipoint < coordinate_list.size(); ipoint++)
    {
        std::cout << coordinate_list[ipoint] << std::endl;
    }
    */
    return coordinate_list;
}


//-----------------------------------------------------
// Implementation of ORB.
//
// Inputs:
//
// P the number of domains to produce.
//
// points a list of type double3_t to partition.
//
// Output:
// points_in_orb_domains is a vector of vectors where
// points_in_orb_domains[iproc] stores the local element
// indexes of all processors in subdomain iproc (i.e.,
// destined for iproc).
//-----------------------------------------------------
void ORB(int P, const std::vector<double3_t>& points, std::vector<std::vector<int> >& points_in_orb_domains)
{
    int rank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	//--------------------------------------------------------------------------------
    // Initialize the output list to size 0
    //--------------------------------------------------------------------------------
    points_in_orb_domains.resize(0);
    
    //--------------------------------------------------------------------------------
    // Initialize the list of indexes to all of our points
    //--------------------------------------------------------------------------------
    
    std::vector<int> point_indexes_in_domain(points.size());
    for (unsigned int ipoint = 0; ipoint < points.size(); ipoint++)
    {
        //code missing to fill out point_indexes_in_domain[ipoint]
        //this should just be the local list of indexes for all points (trivial)
		point_indexes_in_domain[ipoint] = ipoint;
    }
    
    //--------------------------------------------------------------------------------
    // Start by setting dim = 0 (x) and setup the required queues
    //--------------------------------------------------------------------------------
    int dim = 0;
    
    std::queue<double> weight_queue;      //a queue for the weights of incomplete domains
    std::queue<std::vector<int> > points_in_domain_queue; //all of the local points in a given domain
    std::queue<int> dim_queue;            //we need to keep track of the next dim to partition with
    
    weight_queue.push(P);                 //start with P as the initial weight (should be nproc but we pass it in)
    dim_queue.push(dim);                  //start with dim as the initial dimension
    points_in_domain_queue.push(point_indexes_in_domain);   //start with all of the indexes of the points
    
    //--------------------------------------------------------------------------------
    // Now recursively (using queues) process each domain until all weights are 1
    //--------------------------------------------------------------------------------
    while (!weight_queue.empty())
    {
        int weight = weight_queue.front();      //pull the next domain weight
        weight_queue.pop();                     //remove from queue
        
        dim = dim_queue.front();                //pull the next domain partition dimension
        dim_queue.pop();                        //remove from queue
        
        std::vector<int> domain_point_indexes = points_in_domain_queue.front();     //pull the next domain local point indexes
        points_in_domain_queue.pop();           //remove from queue
        
		
        int weight00, weight01;
        
        //Code missing: compute the weights.
		weight00 = (int) floor(weight/2.0);
        weight01 = (int) ceil(weight/2.0);
        
        std::vector<double3_t> domain_points(domain_point_indexes.size());
        for (unsigned int ipoint = 0; ipoint < domain_point_indexes.size(); ipoint++)
        {
            //Code missing: create a list of the actual points in the domain (domain_points) from the original "points" passed in.
			domain_points[ipoint] = points[domain_point_indexes[ipoint]];
        }

        std::vector<double> coords, sorted_coords;
        
        //Code missing: use an existing utility function to extract the "dim" coordinate of the domain_points
		coords = getSingleCoordinateListFromPoints(domain_points, dim);
		
        //Code missing: parallelBucketSort to generate sorted_coords on this processor.
		parallelBucketSort(coords, sorted_coords);
		
        int M0; //the total number of global points in the domain
        int n_local_points_in_domain = domain_point_indexes.size(); //the number of local points in the domain
        
        //Code missing: determine M0.
        MPI_Allreduce(&n_local_points_in_domain, &M0, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);	
		
        int M00;
        
        //Code missing: determine M00, the global index of the weighted median.
		M00 = (int) floor(M0*(double)weight00/(double)weight);		

        //----------------------------------------------------------------
        //Tricky bit of code to get the M00th entry of the list of sorted
        //coordinates distributed across all processors.
        //----------------------------------------------------------------
        
        //First determine how many sorted coordinates ended up on each processor, i.e. the bucket size on each proc.
        //This gets stored in the nproc-sized array "processor_sorted_coords_count". You should understand how this
        //works.
        std::vector<int> n_local_sorted_coords_vec(nproc, sorted_coords.size());
        std::vector<int> processor_sorted_coords_counts(nproc,1);
        MPI_Alltoall(&n_local_sorted_coords_vec[0], 1, MPI_INT, &processor_sorted_coords_counts[0], 1, MPI_INT, MPI_COMM_WORLD);
        
        //Once we know the number of sorted coordinates on each processor we can locally determine
        //Which processor has the M00th index overall.
        int proc_with_weighted_median = -1;
        int index_of_weighted_median_on_proc = -1;
        int running_sum = 0;
        for (int iproc = 0; iproc < nproc; iproc++)
        {
            //Code Missing which can be described as follows:
            
            //IF: the runnnig sum of points over all previous processors
            //plus the count on the current iproc is M00 or greater, then the M00th
            //entry lies on processor iproc. Further, the index of the local entry
            //on iproc corresponding to M00 overall is (M00 - running sum).
            //Set proc_with_weighted_median = iproc and the index_of_weighted_median accordingly.
            //Break the loop.
            if ((running_sum + processor_sorted_coords_counts[iproc]) >= M00)
			{
				proc_with_weighted_median = iproc;
				index_of_weighted_median_on_proc = M00 - running_sum;
				break;
			}
			//ELSE: increment the running sum.
			else
			{
				running_sum += processor_sorted_coords_counts[iproc];			
			}
        }

        //If we have the local weighted median index determine the weighted median value
        double local_weighted_median = 0;
        if (rank == proc_with_weighted_median)
        {
            local_weighted_median = sorted_coords[index_of_weighted_median_on_proc];
        }
		
        //Here we need to inform all processors of the weighted median. One trick for doing this
        //is to recognize that we set the median value to 0 above on all processors except the
        //one that has the median. So what happens if we all-reduce those values?
        
		double weighted_median;
        
		//Code missing to determine the weighted median on all processors.
        MPI_Allreduce(&local_weighted_median, &weighted_median, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
        
        //We will divide the points into two lists according to how they compare by the "dim"
        //coordinate to the median.
        std::vector<int> domain_00_point_indexes(0);
        std::vector<int> domain_01_point_indexes(0);
        
        //Code missing, fill out domain_00_point_indexes and domain_01_point_indexes.
        //Their sizes should sum to the domain_point_indexes.size()
		for (int i = 0; i < domain_point_indexes.size(); i++)
		{
			if(coords[i] <= weighted_median)
			{
				domain_00_point_indexes.push_back(domain_point_indexes[i]);
			}
			else
			{
				domain_01_point_indexes.push_back(domain_point_indexes[i]);
			}
		}
		
        assert(domain_00_point_indexes.size() + domain_01_point_indexes.size() == domain_point_indexes.size());
       
        if (weight00 == 1)
        {
            //we are done with domain_00 - put it in the list of domain.
            //Code missing
			points_in_orb_domains.push_back(domain_00_point_indexes);
        }
        else
        {
            //we need to put domain_00 in the queue for another bisection call.
            //this means adding the weight, new dim and domain_00 to the
            //appropriate queues.
            
            //Code missing
			weight_queue.push(weight00);
			dim_queue.push((dim + 1) % 3);
			points_in_domain_queue.push(domain_00_point_indexes); 
        }
        
        //Code missing to handle weight01 case.
		if (weight01 == 1)
        {
            //we are done with domain_01 - put it in the list of domain.
            //Code missing
			points_in_orb_domains.push_back(domain_01_point_indexes);
        }
        else
        {
            //we need to put domain_01 in the queue for another bisection call.
            //this means adding the weight, new dim and domain_00 to the
            //appropriate queues.
            
            //Code missing
			weight_queue.push(weight01);                 
			dim_queue.push((dim + 1) % 3);
			points_in_domain_queue.push(domain_01_point_indexes);
        }	
    }
    
}


//-----------------------------------------------------
// Implementation of Parallel Bucket Sort.
//
// Inputs:
//
// The values to sort.
//
// Outputs:
//
// A subset of the sorted values. Note that the entries
// in sorted values are not the same as values to sort
// on any given processor. They are a subset of the
// total set of sorted values where rank 0 will contain
// the lowest sorted numbers.
//-----------------------------------------------------
void parallelBucketSort(const std::vector<double>& values_to_sort, std::vector<double>& sorted_values)
{
    int rank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	// Find localMin/localMax to create ranges for buckets
	double localMin = DBL_MAX;
	double localMax = -DBL_MAX;
	for (int i = 0; i < values_to_sort.size(); i++)
	{
		if (values_to_sort[i] < localMin)
			localMin = values_to_sort[i];
		if (values_to_sort[i] > localMax)
			localMax = values_to_sort[i];
	}
	
	// Gather global max & min from all processors
	double globalMax;
	double globalMin;
	MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	
	// Find the ranges of the buckets
	double deltaBucket = (globalMax-globalMin) / nproc; // Note: numBuckets = nproc
	std::vector <double> bucketRanges(nproc);
	for (int i = 0; i < nproc; i++)
	{
		bucketRanges[i] = globalMin + (i * deltaBucket);
	}
	
	// Initialize vectors for sending buckets (1 small bucket for each processor)
	std::vector <std::vector <double>> sendBuckets(nproc);
	std::vector <int> sendCounts(nproc);
	for (int i = 0; i < nproc; i++)
	{
		sendBuckets[i].resize(0);
		sendCounts[i] = 0;
	}
	
	// Place into buckets to be sent to processors
	// Loop over all elements and place into appropriate bucket
	bool placed = false;
	for (int i = 0; i < values_to_sort.size(); i++)
	{
		// Reset placed boolean for next value to sort
		placed = false;
		
		// Loop over all bucket ranges
		for (int j = 1; j < nproc && placed == false; j++)
		{
			
			if (values_to_sort[i] >= bucketRanges[j-1] && values_to_sort[i] < bucketRanges[j])
			{
				sendBuckets[j-1].push_back(values_to_sort[i]);	
				sendCounts[j-1]++;
				placed = true;
			}
			else if (values_to_sort[i] >= bucketRanges[nproc-1])
			{
				sendBuckets[nproc-1].push_back(values_to_sort[i]);
				sendCounts[nproc-1]++;
				placed = true;
			}
		}
	}

	// Send/Receive buckets using AlltoAll function
	std::vector <std::vector <double>> rcvBuckets(nproc);
	MPI_Alltoall_vecvecT(sendBuckets, rcvBuckets);

	
	// Put into large buckets
	std::vector <double> lrgBucket;
	for (int i = 0; i < nproc; i ++)
	{	
		for (int j = 0; j < rcvBuckets[i].size(); j++)
		{
			lrgBucket.push_back(rcvBuckets[i][j]);			
		}
	}
	
	// Sort large buckets
	sort(lrgBucket.begin(), lrgBucket.end());
		
	// Return sorted bucket/vector
	sorted_values = lrgBucket;
}

