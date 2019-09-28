#include <vector>
#include <math.h>
#include <iostream>
#include <cassert>
#include <sstream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mpi.h>

#define MAX_SIZE 1024

using namespace std;

void colorPixelFromScalar(double f, cv::Vec3b& pixel)
{
	assert(f >= 0 - 1e-9 && f <= 1.0 + 1e-9);
	
	if (f < 0.03125) {pixel.val[2] = 59; pixel.val[1] = 76; pixel.val[0] = 192;}
	else if (f < 0.0625) {pixel.val[2] = 68; pixel.val[1] = 90; pixel.val[0] = 204;}
	else if (f < 0.09375) {pixel.val[2] = 77; pixel.val[1] = 104; pixel.val[0] = 215;}
	else if (f < 0.125) {pixel.val[2] = 87; pixel.val[1] = 117; pixel.val[0] = 225;}
	else if (f < 0.15625) {pixel.val[2] = 98; pixel.val[1] = 130; pixel.val[0] = 234;}
	else if (f < 0.1875) {pixel.val[2] = 108; pixel.val[1] = 142; pixel.val[0] = 241;}
	else if (f < 0.21875) {pixel.val[2] = 119; pixel.val[1] = 154; pixel.val[0] = 247;}
	else if (f < 0.25) {pixel.val[2] = 130; pixel.val[1] = 165; pixel.val[0] = 251;}
	else if (f < 0.28125) {pixel.val[2] = 141; pixel.val[1] = 176; pixel.val[0] = 254;}
	else if (f < 0.3125) {pixel.val[2] = 152; pixel.val[1] = 185; pixel.val[0] = 255;}
	else if (f < 0.34375) {pixel.val[2] = 163; pixel.val[1] = 194; pixel.val[0] = 255;}
	else if (f < 0.375) {pixel.val[2] = 174; pixel.val[1] = 201; pixel.val[0] = 253;}
	else if (f < 0.40625) {pixel.val[2] = 184; pixel.val[1] = 208; pixel.val[0] = 249;}
	else if (f < 0.4375) {pixel.val[2] = 194; pixel.val[1] = 213; pixel.val[0] = 244;}
	else if (f < 0.46875) {pixel.val[2] = 204; pixel.val[1] = 217; pixel.val[0] = 238;}
	else if (f < 0.5) {pixel.val[2] = 213; pixel.val[1] = 219; pixel.val[0] = 230;}
	else if (f < 0.53125) {pixel.val[2] = 221; pixel.val[1] = 221; pixel.val[0] = 221;}
	else if (f < 0.5625) {pixel.val[2] = 229; pixel.val[1] = 216; pixel.val[0] = 209;}
	else if (f < 0.59375) {pixel.val[2] = 236; pixel.val[1] = 211; pixel.val[0] = 197;}
	else if (f < 0.625) {pixel.val[2] = 241; pixel.val[1] = 204; pixel.val[0] = 185;}
	else if (f < 0.65625) {pixel.val[2] = 245; pixel.val[1] = 196; pixel.val[0] = 173;}
	else if (f < 0.6875) {pixel.val[2] = 247; pixel.val[1] = 187; pixel.val[0] = 160;}
	else if (f < 0.71875) {pixel.val[2] = 247; pixel.val[1] = 177; pixel.val[0] = 148;}
	else if (f < 0.75) {pixel.val[2] = 247; pixel.val[1] = 166; pixel.val[0] = 135;}
	else if (f < 0.78125) {pixel.val[2] = 244; pixel.val[1] = 154; pixel.val[0] = 123;}
	else if (f < 0.8125) {pixel.val[2] = 241; pixel.val[1] = 141; pixel.val[0] = 111;}
	else if (f < 0.84375) {pixel.val[2] = 236; pixel.val[1] = 127; pixel.val[0] = 99;}
	else if (f < 0.875) {pixel.val[2] = 229; pixel.val[1] = 112; pixel.val[0] = 88;}
	else if (f < 0.90625) {pixel.val[2] = 222; pixel.val[1] = 96; pixel.val[0] = 77;}
	else if (f < 0.9375) {pixel.val[2] = 213; pixel.val[1] = 80; pixel.val[0] = 66;}
	else if (f < 0.96875) {pixel.val[2] = 203; pixel.val[1] = 62; pixel.val[0] = 56;}
	else if (f < 1.0) {pixel.val[2] = 192; pixel.val[1] = 40; pixel.val[0] = 47;}
	else {pixel.val[2] = 180; pixel.val[1] = 4; pixel.val[0] = 38;}
}

void convertMatrixToImage(const std::vector<std::vector<double> >& matrix, double max_scale, cv::Mat& image)
{
    assert(matrix.size() == image.rows);
    for (int irow = 0; irow < image.rows; irow++)
    {
        assert(matrix[irow].size() == image.cols);
        for (int icol = 0; icol < image.cols; icol++)
        {
            double value = matrix[irow][icol];
            //assume value between min_scale and max_scale. Scale to [0 1]
            if (value < 0) value = 0;   //shouldn't happen
            if (value > max_scale) value = max_scale;
            
            value /= max_scale;
            
            colorPixelFromScalar(value,image.at<cv::Vec3b>(irow,icol));
        }
    }
}



int main(int argc, char** argv)
{
	// MPI Declarations
	int rank = 0;						
	int nproc = 1;		
	
	//Initialize MPI
	MPI_Init(&argc, &argv);								
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    //-------------------------------------
    // Parse Command Line
    //-------------------------------------

    if (argc < 9)
    {
        std::cerr << "Input arguments are not complete. Required arguments are " << std::endl;
        std::cerr << "\tNx" << std::endl;
        std::cerr << "\tNy" << std::endl;
        std::cerr << "\tBC at xmin " << std::endl;
        std::cerr << "\tBC at xmax " << std::endl;
        std::cerr << "\tBC at ymin " << std::endl;
        std::cerr << "\tBC at ymax " << std::endl;
        std::cerr << "\tValue of alpha " << std::endl;
        std::cerr << "\tMaximum time " << std::endl;

        exit(1);
    }
    
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    double xmin_bc = strtod(argv[3],NULL);  //boundary condition at xmax
    double xmax_bc = strtod(argv[4],NULL);  //boundary condition at xmin
    double ymin_bc = strtod(argv[5],NULL);  //boundary condition at ymin
    double ymax_bc = strtod(argv[6],NULL);  //boundary condition at ymax
    double alpha = strtod(argv[7],NULL);    //value of alpha
    double tmax = strtod(argv[8],NULL);     //maximum time

    std::cout << "Running Finite Difference Heat Equation with : " << std::endl;
    std::cout << "\tNx              : " << Nx << std::endl;
    std::cout << "\tNy              : " << Ny << std::endl;
    std::cout << "\tBC at xmin      : " << xmin_bc << std::endl;
    std::cout << "\tBC at xmax      : " << xmax_bc << std::endl;
    std::cout << "\tBC at ymin      : " << ymin_bc << std::endl;
    std::cout << "\tBC at ymax      : " << ymax_bc << std::endl;
    std::cout << "\tValue of alpha  : " << alpha << std::endl;
    std::cout << "\tMaximum time    : " << tmax << std::endl;

    
    if (xmin_bc < 0 || ymin_bc < 0 || xmax_bc < 0 || ymax_bc < 0)
    {
        std::cout << "Temperatures are in Kelvin and should not be below zero" << std::endl;
        return -1;
    }
    
    //----------------------------------------
    // Solution Space Creation
    //----------------------------------------

    int N = Nx*Ny;  //global number of unknowns
    double dx = 1.0/(Nx - 1);
    double dy = 1.0/(Ny - 1);
    
    std::vector<std::vector<double> > T(Ny, std::vector<double>(Nx,0.0));
    std::vector<std::vector<double> > Tnew = T;

    //take 90 percent of maximum
    double dt = 0.9*1.0/(2.0*alpha*(1.0/dx/dx + 1.0/dy/dy));
    int Nt = (int)ceil(tmax/dt);
    
    std::cout << "Time step computed as " << dt << std::endl;
    std::cout << "Simulation requires Nt = " << Nt << " time steps " << std::endl;
    
    //----------------------------------------
    // Place Boundary Conditions in T
    // Initial condition elsewhere assumed 0
    //----------------------------------------

    for (unsigned int iy = 0; iy < Ny; iy++)
    {
        T[iy][0] = xmin_bc;
        T[iy][Nx-1] = xmax_bc;
    }
    for (unsigned int ix = 0; ix < Nx; ix++)
    {
        T[0][ix] = ymin_bc;
        T[Ny-1][ix] = ymax_bc;
    }
    
    //copy boundary conditions to Tnew
    Tnew = T;
    
    //----------------------------------------
    // Setup Image for Display
    //----------------------------------------
    double min_scale = min(xmin_bc, min(xmax_bc, min(ymin_bc, ymax_bc)));
    double max_scale = max(xmin_bc, max(xmax_bc, max(ymin_bc, ymax_bc)));
    
    cv::Mat image(Ny,Nx,CV_8UC3);
    cv::Mat image_to_view(MAX_SIZE, MAX_SIZE, CV_8UC3);
    convertMatrixToImage(T,max_scale, image);
    cv::resize(image,image_to_view,image_to_view.size(), cv::INTER_LINEAR);
    
    //cv::namedWindow("Heat Distribution", cv::WINDOW_AUTOSIZE );
    //cv::imshow("Heat Distribution", image_to_view);
	cv::imwrite("./Heat_Original.jpg", image_to_view);
	
    //----------------------------------------
    // Perform Time Marching
    //----------------------------------------

    double t_start = (double)clock()/(double)CLOCKS_PER_SEC;
    
    double invdxdx = 1.0/dx/dx;
    double invdydy = 1.0/dy/dy;
    
    for (unsigned int it = 0; it < Nt; it++)
    {
        std::cout << "\rTime step " << it+1 << " of " << Nt << " time = " << it*dt;
        
        //iterate over points excluding boundaries
        for (unsigned int iy = 1; iy < Ny-1; iy++)
        {
            for (unsigned int ix = 1; ix < Nx-1; ix++)
            {
                //Insert update code here
				Tnew[iy][ix] = T[iy][ix] + (alpha*dt*invdxdx) * (T[iy][ix-1]-2*T[iy][ix]+T[iy][ix+1]) + (alpha*dt*invdydy) * (T[iy-1][ix]-2*T[iy][ix]+T[iy+1][ix]);
            }
        }
        
        T = Tnew;
        
        //Output every 100 time steps
        if (it%1000 == 0)
        {
            convertMatrixToImage(T,max_scale, image);
            cv::resize(image,image_to_view,image_to_view.size(), cv::INTER_LINEAR);
            //cv::imshow("Heat Distribution", image_to_view);
			ostringstream fStream;
			fStream << "./Heat_" << it << ".jpg";
			string fName = fStream.str();
			cv::imwrite(fName, image_to_view);
        }
    }
    
    cv::imwrite("./Heat_Final.jpg", image_to_view);
    double t_stop = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << std::endl;
    std::cout << "Total Solution Time           : " << t_stop - t_start << " seconds " << std::endl;
    std::cout << "Average Time Per Iteration    : " << (t_stop - t_start)/(double)Nt << " seconds " << std::endl;
    
}