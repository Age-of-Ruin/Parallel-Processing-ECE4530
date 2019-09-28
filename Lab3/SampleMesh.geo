// Gmsh project created on Fri Jun 10 11:47:24 2011

c=3e8;

fmax = 1e9/2;
fmin = 1e9/2;
lmin = c/fmax;
lmax = c/fmin;

cl = lmin/3;


// Boundary Sphere //
br = 3*lmin;

p = newp;
Point(p) = {0,0,0,cl};
Point(p+1) = {br,0,0,cl};
Point(p+2) = {0,br,0,cl};
Point(p+3) = {0,-br,0,cl};
Point(p+4) = {-br,0,0,cl};
Point(p+5) = {0,0,br,cl};
Point(p+6) = {0,0,-br,cl};

Circle(1) = {5, 1, 6};
Circle(2) = {6, 1, 2};
Circle(3) = {2, 1, 7};
Circle(4) = {7, 1, 5};
Circle(5) = {5, 1, 3};
Circle(6) = {3, 1, 2};
Circle(7) = {2, 1, 4};
Circle(8) = {4, 1, 5};
Line Loop(9) = {5, 6, -2, -1};
Ruled Surface(10) = {-9};
Line Loop(11) = {2, 7, 8, 1};
Ruled Surface(12) = {-11};
Line Loop(13) = {7, 8, -4, -3};
Ruled Surface(14) = {13};
Line Loop(15) = {3, 4, 5, 6};
Ruled Surface(16) = {15};

//PEC Sphere
sr = 1*lmin;

p = newp;
Point(p) = {0,0,0,cl};
Point(p+1) = {sr,0,0,cl};
Point(p+2) = {0,sr,0,cl};
Point(p+3) = {0,-sr,0,cl};
Point(p+4) = {-sr,0,0,cl};
Point(p+5) = {0,0,sr,cl};
Point(p+6) = {0,0,-sr,cl};
Circle(17) = {12, 1, 13};
Circle(18) = {13, 1, 9};
Circle(19) = {9, 1, 14};
Circle(20) = {14, 1, 12};
Circle(21) = {12, 1, 10};
Circle(22) = {10, 1, 9};
Circle(23) = {9, 1, 11};
Circle(24) = {11, 1, 12};
Line Loop(25) = {21, 22, -18, -17};
Ruled Surface(26) = {25};
Line Loop(27) = {17, 18, 23, 24};
Ruled Surface(28) = {27};
Line Loop(29) = {24, -20, -19, 23};
Ruled Surface(30) = {-29};
Line Loop(31) = {20, 21, 22, 19};
Ruled Surface(32) = {-31};
Surface Loop(33) = {16, 14, 12, 10};
Surface Loop(34) = {32, 30, 28, 26};
Volume(35) = {33, 34};



Physical Surface(1000) = {32, 30, 26, 28}; //PEC Sphere
Physical Volume(2000) = {35}; //Volume
