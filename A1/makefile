compile: BarnesHutMain.cpp TreeNode.cpp Utilities.cpp
	mpicxx -c TreeNode.cpp -o TreeNode.o
	mpicxx -c Utilities.cpp -o Utilities.o
	mpicxx -c BarnesHutMain.cpp -o BarnesHutMain.o
	mpicxx -o barnes BarnesHutMain.o Utilities.o TreeNode.o