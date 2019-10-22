# Divide and Conquer: Barnes Hut
This program computes the force interactions between N bodies. It works by:

– Reading the number of points per processor and the value of θ from the command line.
– Generates that many random points in three-dimensions on each processor over the unit cube with random masses between 1kg and 1000kg.
– Partitions the points using ORB and distributes them.
– Builds a local tree from the ORB’d points.
– Obtains the locally essential bodies required to complete the tree and adds them to the tree 
- Examines the local tree on a given processor and determine which of its points are required for the given domain.
– Computes the force on the bodies.
– Determines the average, minimum and maximum absolute force (each component) over all processors and outputs it to the screen (part of verification).
– Times the important modules of the computations including the overall time and displays the timing results.