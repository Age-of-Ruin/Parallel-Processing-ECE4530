





void ORB(int P, const std::vector<double3_t>& points, std::vector<std::vector<int> >& points_in_orb_domains)
{
    //--------------------------------------------------------------------------------
    // As this is a function outside of a class we need to get the rank and nproc again
    // (or we could pass them as input)
    //--------------------------------------------------------------------------------
    
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
        
        
        std::vector<double3_t> domain_points(domain_point_indexes.size());
        for (unsigned int ipoint = 0; ipoint < domain_point_indexes.size(); ipoint++)
        {
            //Code missing: create a list of the actual points in the domain (domain_points) from the original "points" passed in.
        }
        
        
        std::vector<double> coords, sorted_coords;
        
        //Code missing: use an existing utility function to extract the "dim" coordinate of the domain_points
        //Code missing: parallelBucketSort to generate sorted_coords on this processor.

        int M0; //the total number of global points in the domain
        int n_local_points_in_domain = domain_point_indexes.size(); //the number of local points in the domain
        
        //Code missing: determine M0.
        
        int M00;
        
        //Code missing: determine M00, the global index of the weighted median.

        //----------------------------------------------------------------
        //Tricky bit of code to get the M00th entry of the list of sorted
        //coordinates distributed across all processors.
        //----------------------------------------------------------------
        
        //First determine how many sorted coordinates ended up on each processor, i.e. the bucket size on each proc.
        //This gets stored in the nproc-sized array "processor_sorted_coords_count". You should understand how this
        //works.
        std::vector<int> n_local_sorted_coords_vec(nproc, sorted_coords.size());
        std::vector<int> processor_sorted_coords_counts(nproc,1);
        MPI_Alltoall(&n_local_sorted_coords_vec[0], 1, MPI_INT,&processor_sorted_coords_counts[0], 1, MPI_INT, MPI_COMM_WORLD);
        
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
            
            //ELSE: increment the running sum.
        
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
        
        
        //We will divide the points into two lists according to how they compare by the "dim"
        //coordinate to the median.
        std::vector<int> domain_00_point_indexes(0);
        std::vector<int> domain_01_point_indexes(0);
        
        //Code missing, fill out domain_00_point_indexes and domain_01_point_indexes.
        //Their sizes should sum to the domain_point_indexes.size()

        
        assert(domain_00_point_indexes.size() + domain_01_point_indexes.size() == domain_point_indexes.size());
        
        
        if (weight00 == 1)
        {
            //we are done with domain_00 - put it in the list of domain.
            //Code missing
        }
        else
        {
            //we need to put domain_00 in the queue for another bisection call.
            //this means adding the weight, new dim and domain_00 to the
            //appropriate queues.
            
            //Code missing
        }
        
        //Code missing to handle weight01 case.
    }
    
}
