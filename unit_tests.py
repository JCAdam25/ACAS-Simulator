import math
import numpy as np
import main

## Test create_initial_conditions ######################################################################################
end = False
for rho in range(10000,61000,10000):
    for theta in range(0, 4):
        own_state, int_state, own_path, int_path, path_commands, path_distances = main.create_initial_conditions(rho, (theta*math.pi)/2, math.pi, 200, 200)

        if int_state != [rho*math.sin(-((theta*math.pi)/2)),rho*math.cos(-((theta*math.pi)/2)),math.pi,200]:
            print("create_initial_conditions is not working")
            print(f"int_state is {int_state} but should be {[rho*math.sin(-((theta*math.pi)/2)),rho*math.cos(-((theta*math.pi)/2)),math.pi,200]}")
            end = True
            break
        elif own_state != [0.0,0.0,0.0,200]:
            print("create_initial_conditions is not working")
            print(f"int_state is {int_state} but should be {[0.0,0.0,0.0,200]}")
            end = True
            break
        elif own_path != [[0.0],[0.0]]:
            print("create_initial_conditions is not working")
            print(f"own_path is {own_path} but should be {[[0.0],[0.0]]}")
            end = True
            break
        elif int_path != [[rho*math.sin(-((theta*math.pi)/2))],[rho*math.cos(-((theta*math.pi)/2))]]:
            print("create_initial_conditions is not working")
            print(f"int_path is {int_path} but should be {[[rho*math.sin(-((theta*math.pi)/2))],[rho*math.cos(-((theta*math.pi)/2))]]}")
            end = True
            break
        elif path_commands != ["COC"]:
            print("create_initial_conditions is not working")
            print("did not correctly creat path_commands array")
            end = True
            break
        elif path_distances != [rho]:
            print("create_initial_conditions is not working")
            print("did not correctly creat path_distances array")
            end = True
            break
    if end == True:
        break

## Test find_psi #######################################################################################################
own_state = [0,0,0,0]
int_state = [0,0,0,0]
end = False

for i in range(0,9):
    for j in range (0, 9):
        own_state[2] = i/4*math.pi
        int_state[2] = j/4*math.pi
        psi = main.find_psi(own_state, int_state)
        print(f"own = {i/4}, int = {j/4}, psi = {psi/math.pi}")
    print()
## Test find_theta #####################################################################################################
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

## Test find_rho #######################################################################################################
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

## Test find_time_to_LOS_number ########################################################################################

TIME_TO_CLOSEST_APPROACH = [0,1,5,10,20,40,60,80,100]

for i in range(0, len(TIME_TO_CLOSEST_APPROACH)):
    own_state = [0,0,0, 100]
    int_state = [0,0,0, 50]
    int_state[1] = 499 + ((TIME_TO_CLOSEST_APPROACH[i] + 0.5) * (own_state[3]-int_state[3]))
    returned_index = main.find_time_to_LOS_number(own_state, int_state)
    if returned_index != i+1:
        print(f"find_time_to_LOS_number function is not working correctly")
        print(f"should have returned {i+1}, instead gave {returned_index}")
        break

## Test run_network ####################################################################################################
own_state = [0.0, 0.0, 0.0, 250]
int_state = [0.0, 20000.0, math.pi, 250]

next_command = None

next_command = main.run_network(0, own_state, int_state)

if next_command == None or next_command > 4:
    print(f"run_network not working correctly, did not return a valid command")

## Test step ###########################################################################################################
own_state = [0.0, 0.0, 0.0, 250]
int_state = [0.0, 20000.0, math.pi, 250]
own_path = [[0.0],[0.0]]
int_path = [[0.0],[20000.0]]
path_commands = ["COC"]
path_distances = [20000.0]
min_distance = 20000.0
own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances = main.step(0, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances)
if own_path[0][len(own_path[0])-1] != 0 or own_path[1][len(own_path[1])-1] != 500.0:
    print(f"step is not working correctly in mapping Ownship's path")
    print(f"should be at (0.0,500.0) but is at ({own_path[0][len(own_path[0])-1]},{own_path[1][len(own_path[1])-1]})")
elif int_path[0][len(int_path[0])-1] != -6.123233995736766e-14 or int_path[1][len(int_path[1])-1] != 19500.0:
    print(f"step is not working correctly in mapping Intruder's path")
    print(f"should be at (-6.123233995736766e-14,19500.0) but is at ({int_path[0][len(int_path[0])-1]},{int_path[1][len(int_path[1])-1]})")
elif len(own_path[0]) != 21 or len(own_path[1]) != 21:
    print(f"step is not working correctly in mapping Ownship's path")
    print(f"should have 21 steps but has {len(own_path[0])} for x and {len(own_path[1])} for y")
elif len(int_path[0]) != 21 or len(int_path[1]) != 21:
    print(f"step is not working correctly in mapping Intruder's path")
    print(f"should have 21 steps but has {len(int_path[0])} for x and {len(int_path[1])} for y")
elif own_state[0] != own_path[0][len(own_path[0])-1] or own_state[1] != own_path[1][len(own_path[1])-1]:
    print(f"step is not keeping Ownship's state and path in line")
    printf("state is at ({own_state[0]},{own_state[1]}) and path is at ({own_path[0][len(own_path[0])-1]},{own_path[1][len(own_path[1])-1]})")
elif int_state[0] != int_path[0][len(int_path[0])-1] or int_state[1] != int_path[1][len(int_path[1])-1]:
    print(f"step is not keeping Intruder's state and path in line")
    print(f"state is at ({int_state[0]},{int_state[1]}) and path is at ({int_path[0][len(int_path[0])-1]},{int_path[1][len(int_path[1])-1]})")
elif len(path_commands) != 21 or path_commands[len(path_commands)-1] != "COC":
    print(f"step is not correctly updating path_commands")
    print(f"has {len(path_commands)} steps but should have 21")
    print(f'last command is {path_commands[len(path_commands)]} but should be "COC"')
elif len(path_distances) != 21 or path_distances[len(path_distances)-1] != 19000.0:
    print(f"step is not correctly updating path_distances")
    print(f"has {len(path_distances)} steps but should have 21")
    print(f"last distance is {path_distances[len(path_distances)]} but should be 19000.0")
elif min_distance != 19000.0:
    print(f"step is not correctly updating min_distance")
    print(f"is {min_distance} but should be 19000.0")
