# ACAS-Simulator

This simulator was developed and tested on python 3.9.
In order to run it requires the ONNXruntime, MatPlotLib, and NumPy modules

A simulator that takes in 5 initial conditions;
  • rho - intial separation in feet between 10000 and 60000
  • theta - angle between heading of ownship (initially north) and intruder in radians between 0 and 2*pi
  • psi - angle between heading of ownship and heading of intruder in radians between 0 and 2*pi
  • v_own - velocity of ownship in feet/second between 100 and 1200
  • v_int - velocity of intruder in feet/second between 100 and 1200
It the uses the ACAS-Xu network to produce commands for the ownship to try and prevent a collision
Produces an animated simulation of the encounter, with the colour of the ownships flight path showing what command it is currently obeying;
  • Clear of Conflict (red) - carry on straight
  • Weak Left (orange) - turn left at a rate of 1.5°/second
  • Weak Right (purple) - turn right at a rate of 1.5°/second
  • Strong Left (yellow) - turn left at a rate of 3.0°/second
  • Strong Right (blue) - turn right at a rate of 3.0°/second

To run the simulator without generating a visualisation, call the run_simulation function from within simulation.py
Pass into it the 5 conditions, and a False value at the end.
This function return the paths of each aircraft, the commands followed by the ownship at each coordinate, and the separation at each coordinate.

To create a visualisation from a previously stored set of data use the animate function from visualiser.py.
This takes in the ownship's path, the intruder's path, and the path commands.

The test.py class contains unit tests for all of the functions in both simulator.py and visualiser.py.
It also includes a system test that will run the entire system, and print out the results returned from run_simulation in an easily understandable order.