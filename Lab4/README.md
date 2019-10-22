# Parallel Heat Equation Simulation
This lab simulates the explicit finite-difference solution to the Heat Equation. It takes the following input arguments:

• Nx the number of points over x
• Ny the number of points over y
• xmin bc the boundary condition at xmin (note that for anything to happen the boundary values should not all be zero)
• xmax bc the boundary condition at xmax
• ymin bc the boundary condition at ymin
• ymax bc the boundary condition at ymax
• alpha the value of α in the Heat Equation (note that the choice affects how fast the solution moves in time; α = 1 is a good starting point).
• tmax the maximum time up to which the simulation should be run (a value of α = 1 will have a solution that settles down to steady-state in less than a second).

The program then visualizes each time-step by coloring pixels based upon their heat (calculated over time by utilizing the supplied boundary conditions).