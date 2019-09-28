#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <cassert>
#include <mpi.h>
#include <sstream>
#include <map>
#include <algorithm>
#include "./Mesh.h"
#include "./MeshUtilities.h"
#include <unistd.h>

using namespace std;

//-------------------------------------
// Constructor
//-------------------------------------
Mesh::Mesh(string filename)
{
    vertices_.resize(0);
    elements_3d_.resize(0);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc_);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank_);
    
    readMesh(filename);
    
	getElementVertices();   //complete the vertices we have for local elements
    
    createElementCentroidsList();    //create the list of centroids as private member
    
}

//-------------------------------------
// Destructor - free dynamically allocated
// memory here
//-------------------------------------
Mesh::~Mesh()
{
    
}


//-------------------------------------
// Get a pointer to a vertex from the
// mesh id (mid). Returns NULL if not
// found.
//-------------------------------------
const vertex_t* Mesh::getVertexFromMID(int vertex_mid) const
{
    vertex_t search_vertex;
    search_vertex.mid = vertex_mid;
    void* result = bsearch(&search_vertex, &vertices_[0], vertices_.size(), sizeof(vertex_t), searchVertexByMIDPredicate);
    if (result != NULL) // found vertex
    {
        const vertex_t* found_vertex = (const vertex_t*)result;
        return found_vertex;
    }
    else
    {
        return NULL;
    }
}

//-------------------------------------
// Read a Gmsh formatted (version 2+)
// mesh. Creates list of vertices and
// elements. Not guaranteed to have
// vertices for local elements.
//-------------------------------------
void Mesh::readMesh(string filename)
{
    ifstream in_from_file(filename.c_str(),std::ios::in);
	
    if (!in_from_file.is_open())
	{
		if (rank_ == 0) std::cerr << "Mesh File " << filename << " Could Not Be Opened!" << std::endl;
        MPI_Finalize();
        exit(1);
	}
    
    std::string parser = "";
	in_from_file >> parser;
    
    if (parser != "$MeshFormat" )
	{
        if (rank_ == 0) std::cerr << "Invalid Mesh Format/File" << std::endl;
        MPI_Finalize();
        exit(1);
	}
    
	double dummy;
    in_from_file >> dummy >> dummy >> dummy;
    in_from_file >> parser >> parser;       //skip $EndMeshFormat and $Nodes
	
    //Vertex Read
    int n_vertices_in_file;
	in_from_file >> n_vertices_in_file;
	int local_vert_start;
	int local_vert_stop;
    int local_vert_count;
	parallelRange(0, n_vertices_in_file - 1, rank_, nproc_, local_vert_start, local_vert_stop, local_vert_count);
    
	vertices_.resize(0);
	vertex_t vertex;
	
    
	for (int ivert = 0; ivert < n_vertices_in_file; ivert++)
	{
		if (ivert >= local_vert_start && ivert <= local_vert_stop)
		{
			in_from_file >> vertex.mid >> vertex.r[0] >> vertex.r[1] >> vertex.r[2];
			vertices_.push_back(vertex);
		}
		else    //skip over the line
		{
			int dummy;
			in_from_file >> dummy;
			in_from_file.ignore(1000,'\n');
		}
	}

    //3D Tetrahedral Element Read
    
	elements_3d_.resize(0);
    
	in_from_file >> parser >> parser;   //skip $EndNodes and $Elements
    
    if (parser != "$Elements")
    {
        std::cerr << "Something has gone very wrong" << std::endl;
        assert(0==1);
    }
    
	int n_elements_in_file = 0;
	in_from_file >> n_elements_in_file;
    
	int local_ele_start;
	int local_ele_stop;
    int local_ele_count;
	parallelRange(0, n_elements_in_file - 1, rank_, nproc_, local_ele_start, local_ele_stop, local_ele_count);
    
    int element_num_tags;
	element_t element;
    
	int n_global_3d_elements = 0;
    
	for (int i_ele = 0; i_ele < n_elements_in_file; i_ele++)
	{
		int ele_in_range = (i_ele >= local_ele_start && i_ele <= local_ele_stop);
		
        in_from_file >> element.mid >> element.type >> element_num_tags >> dummy >> dummy;
        for (int itag = 0; itag < element_num_tags - 2; itag++)
        {
            in_from_file >> dummy;
        }
        
        //we are going to skip over anything that isn't a tetrahedral element so we just peel off the vertex mids
		if (element.type == FV_MESH_GMESH_ELEMENT_POINT)
		{
			in_from_file >> dummy;
		}
		else if (element.type == FV_MESH_GMESH_ELEMENT_FIRST_ORDER_LINE)
		{
			in_from_file >> dummy >> dummy;
		}
		else if (element.type == FV_MESH_GMESH_ELEMENT_FIRST_ORDER_TRIANGLE)
		{
			in_from_file >> dummy >> dummy >> dummy;
		}
		else if (element.type == FV_MESH_GMESH_ELEMENT_FIRST_ORDER_QUADRANGLE)
		{
			in_from_file >> dummy >> dummy >> dummy >> dummy;
		}
		else if (element.type == FV_MESH_GMESH_ELEMENT_FIRST_ORDER_TETRAHEDRAL)
		{
			in_from_file >> element.vertex_mids[0];
			in_from_file >> element.vertex_mids[1];
			in_from_file >> element.vertex_mids[2];
			in_from_file >> element.vertex_mids[3];
            element.nvert = 4;
			if (ele_in_range) elements_3d_.push_back(element);
		}
		else if (element.type == FV_MESH_GMESH_ELEMENT_FIRST_ORDER_HEXAHEDRAL)
		{
			in_from_file >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy;
		}
		else
		{
            std::cerr << "Hit an unsupported element type" << std::endl;
            assert(0==1);
		}
	}//for each element
    
	in_from_file.close();
}


//--------------------------------------------------------------------------
// Construct the list of element centroids.
//--------------------------------------------------------------------------

void Mesh::createElementCentroidsList()
{
    element_3d_centroids_.resize(elements_3d_.size());
    const vertex_t* vertex_pointer;

    for (unsigned int iele = 0; iele < elements_3d_.size(); iele++)
    {
        element_3d_centroids_[iele].r[0] = 0.0;
        element_3d_centroids_[iele].r[1] = 0.0;
        element_3d_centroids_[iele].r[2] = 0.0;

        //The centroid of a simplex is just the average of the vertex coordinates
        for (unsigned int ivert = 0; ivert < elements_3d_[iele].nvert; ivert++)
        {
            vertex_pointer = getVertexFromMID(elements_3d_[iele].vertex_mids[ivert]);
            element_3d_centroids_[iele].r[0] += vertex_pointer->r[0];
            element_3d_centroids_[iele].r[1] += vertex_pointer->r[1];
            element_3d_centroids_[iele].r[2] += vertex_pointer->r[2];
        }
        
        element_3d_centroids_[iele].r[0] /= (double)elements_3d_[iele].nvert;
        element_3d_centroids_[iele].r[1] /= (double)elements_3d_[iele].nvert;
        element_3d_centroids_[iele].r[2] /= (double)elements_3d_[iele].nvert;
    }
    
    //Sanity check - the average of the centroids should be the "center" of the mesh.
    
    double3_t centroid_sum;
    centroid_sum.r[0] = 0.0;
    centroid_sum.r[1] = 0.0;
    centroid_sum.r[2] = 0.0;
    
    for (unsigned int iele = 0; iele < element_3d_centroids_.size(); iele++)
    {
        centroid_sum.r[0] += element_3d_centroids_[iele].r[0];
        centroid_sum.r[1] += element_3d_centroids_[iele].r[1];
        centroid_sum.r[2] += element_3d_centroids_[iele].r[2];
    }
    
    
    double3_t total_centroid_sum;
    MPI_Reduce(&centroid_sum.r[0], &total_centroid_sum.r[0], 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int centroid_count = element_3d_centroids_.size();
    int total_centroid_count;
    
    MPI_Reduce(&centroid_count, &total_centroid_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank_ == 0)
    {
        double3_t average_centroid = total_centroid_sum;
        average_centroid.r[0] /= (double)total_centroid_count;
        average_centroid.r[1] /= (double)total_centroid_count;
        average_centroid.r[2] /= (double)total_centroid_count;
        
        std::cout << "The average centroid across all processors is (" << average_centroid.r[0] <<  ", " << average_centroid.r[1] << ", " << average_centroid.r[2] << ")" << std::endl;
    }
}


//--------------------------------------------------------------------------
// Obtain the list of unique vertices that completes our elements on
// every processor.
//--------------------------------------------------------------------------
void Mesh::getElementVertices()
{
    if (nproc_ == 1) return;
    
    //determine the vertices that we need to complete our local elements
    std::vector<int> vertex_mid_requests(0);
    for (unsigned int iele = 0; iele < elements_3d_.size(); iele++)
    {
        for (unsigned int ivert = 0; ivert < elements_3d_[iele].nvert; ivert++)
        {
            vertex_mid_requests.push_back(elements_3d_[iele].vertex_mids[ivert]);
        }
    }
    
    //make the list unique
    sort(vertex_mid_requests.begin(), vertex_mid_requests.end());
	vertex_mid_requests.erase(unique(vertex_mid_requests.begin(), vertex_mid_requests.end()), vertex_mid_requests.end());

    //now we know all the vertices we need to actually complete our elements
    
    //should already be sorted based on sequential read from mesh - but just to be safe
    sort(vertices_.begin(), vertices_.end(),sortVertexByMIDPredicate);

    
    //having collected the vertex set, we can now obtain the vertices we require
    //we send the vertices we need to every process (we could do better here)
    
    std::vector<std::vector<int> > outgoing_vertex_requests(0);
    for (unsigned int iproc = 0; iproc < nproc_; iproc++) outgoing_vertex_requests.push_back(vertex_mid_requests);
    
    std::vector<std::vector<int> > incoming_vertex_requests(0);
    MPI_Alltoall_vecvecT(outgoing_vertex_requests, incoming_vertex_requests);
    
    std::vector<std::vector<vertex_t> > outgoing_vertices(nproc_);
    vertex_t search_vertex;
    for (unsigned int iproc = 0; iproc < nproc_; iproc++)
    {
        outgoing_vertices[iproc].resize(0);
        for (unsigned int ivert = 0; ivert < incoming_vertex_requests[iproc].size(); ivert++)
        {
            int vertex_mid = incoming_vertex_requests[iproc][ivert];
            search_vertex.mid = vertex_mid;
            void* result = bsearch(&search_vertex, &vertices_[0], vertices_.size(), sizeof(vertex_t), searchVertexByMIDPredicate);
            if (result != NULL) // found vertex
            {
                const vertex_t* found_vertex = (const vertex_t*)result;
                search_vertex.r[0] = found_vertex->r[0];
                search_vertex.r[1] = found_vertex->r[1];
                search_vertex.r[2] = found_vertex->r[2];
                outgoing_vertices[iproc].push_back(search_vertex);
            }
        }
    }

    std::vector<std::vector<vertex_t> > incoming_vertices;
    MPI_Alltoall_vecvecT(outgoing_vertices, incoming_vertices);
    
    vertices_.resize(0);
    std::map<int,int> vertex_map;
    for (unsigned int iproc = 0; iproc < nproc_; iproc++)
    {
        for (unsigned int ivert = 0; ivert < incoming_vertices[iproc].size(); ivert++)
        {
            if (vertex_map.find(incoming_vertices[iproc][ivert].mid) == vertex_map.end())
            {
                vertices_.push_back(incoming_vertices[iproc][ivert]);
                vertex_map[incoming_vertices[iproc][ivert].mid] = 1;
            }
        }
    }
    
    sort(vertices_.begin(), vertices_.end(), sortVertexByMIDPredicate);
    
    //now we should be able to find the vertices for any element we have locally.
    //sanity check.
    for (unsigned int iele = 0; iele < elements_3d_.size(); iele++)
    {
        for (unsigned int ivert = 0; ivert < elements_3d_[iele].nvert; ivert++)
        {
            const vertex_t* vertex = getVertexFromMID(elements_3d_[iele].vertex_mids[ivert]);
            assert(vertex != NULL);
        }
    }
    
    
}

//--------------------------------------------------------------------------
// Write unstructured mesh to Paraview XML format. This will create P+1 files
// on P processors. The .vtu files are pieces of the mesh. The .pvtu file is a
// single wrapper file that can be loaded in paraview such that every .vtu file with
// the corresponding names will be opened simultaneously.
//
// Inputs:
//
// The filename should be a complete path with NO extension (.vtu and .pvtu
// will be added.
//
// Value label is a string that will be written to the vtu file labeling the values
// that you are writing for each element (e.g. "rank").
//
// Values is a vector with length elements_3d_.size() corresponding to a single
// scalar value to be associated with each element in the mesh.
//--------------------------------------------------------------------------
void Mesh::writeMesh(string filename, std::string value_label, const vector<double>& values) const
{
    ofstream vtu_out, pvtu_out;
    
    std::ostringstream converter;
    converter << filename << "_P" << nproc_ << "_R" << rank_;
    std::string vtu_filename = converter.str() + ".vtu";
    
    //--------------------------------------------------------------------------------
    // Open the VTU file (All ranks)
    //--------------------------------------------------------------------------------

    vtu_out.open(vtu_filename.c_str());
    if (!vtu_out.is_open())
    {
        std::cerr << "Could not open vtu file" << std::endl;
        assert(0==1);
    }
    
    vtu_out << "<?xml version=\"1.0\"?>" << std::endl;
    vtu_out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
        
    //--------------------------------------------------------------------------------
    // Open the PVTU file (Rank 0)
    //--------------------------------------------------------------------------------
        
    if (rank_==0)
    {
        std::ostringstream pconverter;
        pconverter << filename << "_P" << nproc_;
        std::string pvtu_filename = pconverter.str() + ".pvtu";
        pvtu_out.open(pvtu_filename.c_str());
        if (!pvtu_out.is_open())
        {
            std::cerr << "Could not open pvtu file" << std::endl;
            assert(0==1);
        }
            
        pvtu_out << "<?xml version=\"1.0\"?>" << std::endl;
        pvtu_out << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    }
        
    //--------------------------------------------------------------------------------
    // Write the 3D Mesh Elements to File
    //--------------------------------------------------------------------------------
        
    int n_elements = elements_3d_.size();
    
    //--------------------------------------------------------------------------------
    // VTU Mesh
    //--------------------------------------------------------------------------------
    
    //Preamble
    vtu_out << "<UnstructuredGrid>" << std::endl;
    vtu_out << "<Piece NumberOfPoints=\"" << vertices_.size() << "\" NumberOfCells=\"" << n_elements << "\">" << std::endl;
    
    //Vertices
    vtu_out << "<Points>" << std::endl;
    vtu_out << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    for (unsigned int ivert = 0; ivert < (int)vertices_.size(); ivert++)
    {
        vtu_out << vertices_[ivert].r[0] << " " << vertices_[ivert].r[1] << " " << vertices_[ivert].r[2] << " ";
    }
    vtu_out << std::endl;
    vtu_out << "</DataArray>" << std::endl;
    vtu_out << "</Points>" << std::endl;
        
    vtu_out << "<Cells>" << std::endl;
        
    //Element Connectivity
    vtu_out << "<DataArray type=\"Int32\" Name=\"connectivity\">" << std::endl;
    for (unsigned int iele = 0; iele < (int)elements_3d_.size(); iele++)
    {
        for (unsigned int ivert = 0; ivert < elements_3d_[iele].nvert; ivert++)
        {
            const vertex_t* vertex = getVertexFromMID(elements_3d_[iele].vertex_mids[ivert]);
            assert(vertex != NULL);
            int vertex_lid = (vertex - &vertices_[0]);
            assert(vertices_[vertex_lid].mid == elements_3d_[iele].vertex_mids[ivert]);
            assert(vertex_lid >= 0 && vertex_lid < vertices_.size());
            vtu_out << vertex_lid << " ";
        }
    }
    vtu_out << std::endl;
    vtu_out << "</DataArray>" << std::endl;
        
    //Offsets
    vtu_out << "<DataArray type=\"Int32\" Name=\"offsets\">" << std::endl;
    int vert_sum = 0;
    for (unsigned int iele = 0; iele < elements_3d_.size(); iele++)
    {
        vert_sum += elements_3d_[iele].nvert;
        vtu_out << vert_sum << " ";
    }
    vtu_out << std::endl;
    vtu_out << "</DataArray>" << std::endl;
        
    //Types
    vtu_out << "<DataArray type=\"UInt8\" Name=\"types\">" << std::endl;
    for (unsigned int iele = 0; iele < (int)elements_3d_.size(); iele++)
    {
        vtu_out << "10 ";
    }
    vtu_out << std::endl;
        
    vtu_out << "</DataArray>" << std::endl;
    vtu_out << "</Cells>" << std::endl;
        
        
    //--------------------------------------------------------------------------------
    // PVTU Mesh
    //--------------------------------------------------------------------------------
        
    if (rank_ == 0)
    {
        pvtu_out << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
        
        pvtu_out << "<PPoints>" << std::endl;
        pvtu_out << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
        
        pvtu_out << "</PDataArray>" << std::endl;
        pvtu_out << "</PPoints>" << std::endl;
        
        pvtu_out << "<PCells>" << std::endl;
        
        //Connectivity
        pvtu_out << "<PDataArray type=\"Int32\" Name=\"connectivity\">" << std::endl;
        pvtu_out << "</PDataArray>" << std::endl;
        
        //Offsets
        pvtu_out << "<PDataArray type=\"Int32\" Name=\"offsets\">" << std::endl;
        pvtu_out << "</PDataArray>" << std::endl;
        
        //Types
        pvtu_out << "<PDataArray type=\"UInt8\" Name=\"types\">" << std::endl;
        pvtu_out << "</PDataArray>" << std::endl;
        pvtu_out << "</PCells>" << std::endl;
    }
    
    //--------------------------------------------------------------------------------
    // VTU Cell Data Open
    //--------------------------------------------------------------------------------
        
    vtu_out << "<CellData>" << std::endl;

    vtu_out << "<DataArray type=\"Float32\" format=\"ascii\" Name=\"" << value_label << "\">" << std::endl;
    assert(values.size() == elements_3d_.size());
    for (int iele = 0; iele < (int)elements_3d_.size(); iele++)
    {
        vtu_out << values[iele] << " ";
    }
    vtu_out << "</DataArray>" << std::endl;
    vtu_out << "</CellData>" << std::endl;
    
    //--------------------------------------------------------------------------------
    // PVTU Cell Data Open
    //--------------------------------------------------------------------------------
        
    if (rank_ == 0)
    {
        pvtu_out << "<PCellData>" << std::endl;
        pvtu_out << "<PDataArray type=\"Float32\" format=\"ascii\" Name=\"" << value_label << "\">" << std::endl;
        pvtu_out << "</PDataArray>" << std::endl;
        pvtu_out << "</PCellData>" << std::endl;
    }
    
    
    //--------------------------------------------------------------------------------
    // VTU Close
    //--------------------------------------------------------------------------------
    
    vtu_out << "</Piece>" << std::endl;
    vtu_out << "</UnstructuredGrid>" << std::endl;
    vtu_out << "</VTKFile>" << std::endl;
    vtu_out.close();
    
    //--------------------------------------------------------------------------------
    // PVTU Close
    //--------------------------------------------------------------------------------
    
    if (rank_ == 0)
    {
        for (int iproc = 0; iproc < nproc_; iproc++)
        {
            std::ostringstream vtu_converter;
            //vtu_converter << vtkfilename << "_Run" << iRun << "_N" << num_proc_ <<"_P" << iproc << ".vtu";
            //We always assume the pvtu file exists in the same directory as the other files so here vtkfilename must only be the relative name
            vtu_converter << filename << "_P" << nproc_ << "_R" << iproc << ".vtu";
            pvtu_out << "<Piece Source=\"" << vtu_converter.str() << "\"/>" << std::endl;
        }
        
        pvtu_out << "</PUnstructuredGrid>" << std::endl;
        pvtu_out << "</VTKFile>" << std::endl;
        pvtu_out.close();
    }
}


//-------------------------------------
// Mesh partitioning using ORB. You are responsible
// for writing this function. It should call the
// ORB function you will write in MeshUtilities.cpp.
// That function will return labels for each element
// indicating which processor they should be moved to.
// Then you should move the elements and obtain the vertices.
//-------------------------------------
void Mesh::partitionMesh()
{
	// Get indices of elements destined for each processor (using ORB)
    std::vector <std::vector<int>>  indices_for_each_proc;
	ORB(nproc_, element_3d_centroids_, indices_for_each_proc);
	
	// Convert indices to centroids/elements
	std::vector <std::vector <element_t>> elements_to_be_sent(nproc_);
	std::vector <std::vector <element_t>> elements_to_be_rcvd(nproc_);
	for (int i = 0; i < nproc_; i ++)
	{
		elements_to_be_sent[i].resize(indices_for_each_proc[i].size());
		elements_to_be_rcvd[i].resize(indices_for_each_proc[i].size());

		for (int j = 0; j < indices_for_each_proc[i].size(); j++)
		{
			elements_to_be_sent[i][j] = elements_3d_[indices_for_each_proc[i][j]];
		}
	}
	
	// Send appropriate elements/centroids to each processor
	MPI_Alltoall_vecvecT(elements_to_be_sent, elements_to_be_rcvd);
	
	// Overwrite current list of elements with new/partitioned list of elements
	int elementCount = 0;
	elements_3d_.resize(0);
	for (int i = 0; i < nproc_; i ++)
	{		
		for (int j = 0; j < elements_to_be_rcvd[i].size(); j++)
		{
			elements_3d_.push_back(elements_to_be_rcvd[i][j]);
			elementCount++;
		}	
	}
	
	// Obtain the vertices of the elements now placed on appropriate processor
	getElementVertices();
}

//-------------------------------------
// Output some statistics including
// number of elements/vertices on each
// processor and global number of
// elements/vertices. Nicely formatted.
//-------------------------------------
void Mesh::outputStatistics() const
{
	// Synchronize
	MPI_Barrier(MPI_COMM_WORLD);  
	usleep(10000*rank_);
	
	// Print local statistics
	int localElementCount = elements_3d_.size();
	int localVertexCount = vertices_.size();
	
	std::cout << "Rank " << rank_ << std::endl;
	std::cout << "Number of Elements: " << localElementCount << std::endl;
	std::cout << "Number of Vertices: " << localVertexCount << "\n" << std:: endl;
	
	// Acquire global statistics
	int globalElementCount = 0;
	int globalVertexCount = 0; 
	MPI_Allreduce(&localElementCount, &globalElementCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localVertexCount, &globalVertexCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	
	// Synchronize
	MPI_Barrier(MPI_COMM_WORLD);  
	usleep(10000*rank_);
	
	// Print global statistics	
	if(rank_ == 0)
	{
		std::cout << "Global" << std::endl;
		std::cout << "Number of Elements: " << globalElementCount << std::endl;
		std::cout << "Number of Vertices: " << globalVertexCount << std:: endl;
	}
}