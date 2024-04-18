# ACAS-Simulator

A simulator that takes in 5 initial conditions;
  • intial seperation in feet
  • angle between heading of ownship (initially north) and intruder in radians
  • angle between heading of ownship and heading of intruder in radians
  • velocity of ownship in feet/second
  • velocity of intruder in feet/second
It the uses the ACAS-Xu network to produce commands for the ownship to try and prevent a collision
Produces an animated simulation of the encounter, with the colour of the ownships flight path showing what command it is currently obeying;
  • Clear of Conflict (red) - carry on straight
  • Weak Left (orange) - turn left at a rate of 1.5°/second
  • Weak Right (purple) - turn right at a rate of 1.5°/second
  • Strong Left (yellow) - turn left at a rate of 3.0°/second
  • Strong Right (blue) - turn right at a rate of 3.0°/second

This simulator was developed and tested on python 3.9.
In order to run it requires the ONNXruntime, MatPlotLib, and NumPy modules downloaded on the device.
