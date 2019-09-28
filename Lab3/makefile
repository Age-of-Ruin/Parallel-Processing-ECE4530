MeshDriver: Mesh.cpp MeshUtilities.cpp MeshDriverMain.cpp
	mpicxx -c Mesh.cpp -o Mesh.o
	mpicxx -c MeshUtilities.cpp -o MeshUtilities.o
	mpicxx -c MeshDriverMain.cpp -o MeshDriverMain.o
	mpicxx -o MeshDriver.out Mesh.o MeshUtilities.o MeshDriverMain.o