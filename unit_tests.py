import math
import numpy as np
import main

## Test create_initial_conditions ######################################################################################


## Test find_theta ################################################################################
coords_loop = [[0,-1,-1,-1,0,1,1,1],[1,1,0,-1,-1,-1,0,1]]
own_state = [0,0,0,0]
int_state = [0,0,0,0]
end = False
prev = (-math.pi/4)
for i in range(0,8):
    for j in range(0, len(coords_loop[0])):
        int_state[0] = coords_loop[0][j]
        int_state[1] = coords_loop[1][j]
        own_state[2] = math.pi * (i/4)
        ##print(f"({coords_loop[0][j]},{coords_loop[1][j]}) - {math.degrees(own_state[2])} = {math.degrees(main.find_theta(own_state, int_state))}")
        if main.find_theta(own_state, int_state) != ((j-i)*(math.pi/4)) and main.find_theta(own_state, int_state) != ((j-i+8)*(math.pi/4)):
            print(f"find_theta is not working correctly")
            print(f"returned {main.find_theta(own_state, int_state)}, should have returned {(j-i)*(math.pi/4)} or {(j-i+8)*(math.pi/4)}")
            end = True
            break
    if end == True:
        break

## Test find_psi ##################################################################################
own_state = [0,0,0,0]
int_state = [0,0,0,0]

## Test find_rho ##################################################################################
own_coords_loop = [[0,1,1,1,0,-1,-1,-1],[1,1,0,-1,-1,-1,0,1]]
int_coords_loop = [[0,3,5,3,0,-3,-5,-3],[5,4,0,-4,-5,-4,0,4]]
own_state = [0,0,0,0]
int_state = [0,0,0,0]
end = False
for i in range(0, len(own_coords_loop[0])):
    for j in range(0, len(own_coords_loop[0])):
        own_state[0] = own_coords_loop[0][i]
        own_state[1] = own_coords_loop[1][i]
        int_state[0] = own_state[0] + int_coords_loop[0][j]
        int_state[1] = own_state[1] + int_coords_loop[1][j]
        rho = main.find_rho(own_state, int_state)
        if rho != 5:
            print(f"find_rho is not working correctly")
            print(f"returned {rho}, should have returned 5.0")
            end = True
            break
    if end == True:
        break

## Test find_time_to_closest_approach_number ######################################################

TIME_TO_CLOSEST_APPROACH = [0,1,5,10,20,40,60,80,100]

for i in range(0, len(TIME_TO_CLOSEST_APPROACH)):
    own_state = [0,0,0, 100]
    int_state = [0,0,0, 50]
    int_state[1] = 499 + ((TIME_TO_CLOSEST_APPROACH[i] + 0.3) * (own_state[3]-int_state[3]))
    returned_index = main.find_time_to_closest_approach_number(own_state, int_state)
    if returned_index != i+1:
        print(f"find_time_to_closest_approach_number function is not working correctly")
        print(f"should have returned {i+1}, instead gave {returned_index}")
        break

returned_index = main.find_time_to_closest_approach_number([0,0,0,100],[510,600,0,50])
if returned_index != -1:
    print(f"find_time_to_closest_approach_number function is not working correctly")
    print(f"should have returned -1, instead gave {returned_index}")
