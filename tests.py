import math
import numpy as np
import simulator as sim
import animator as ani

def test_create_initial_conditions():
    '''
    Checks if:
        int_state and own_state have the correct values
        int_path and own_path have the correct coordinates
        path_commands and path_distances are initialised correctly
    ''' 
    for rho in range(10000,61000,10000):
        for theta in range(4):
            own_state, int_state, own_path, int_path, path_commands, path_distances = sim.create_initial_conditions(rho, (theta*math.pi)/2, math.pi, 200, 200)

            if int_state != [rho*math.sin(-((theta*math.pi)/2)),rho*math.cos(-((theta*math.pi)/2)),math.pi,200]:
                print(f"create_initial_conditions is not working")
                print(f"int_state is {int_state} but should be {[rho*math.sin(-((theta*math.pi)/2)),rho*math.cos(-((theta*math.pi)/2)),math.pi,200]}")
                return 1
            elif own_state != [0.0,0.0,0.0,200]:
                print(f"create_initial_conditions is not working")
                print(f"int_state is {int_state} but should be {[0.0,0.0,0.0,200]}")
                return 
            elif own_path != [[0.0],[0.0]]:
                print(f"create_initial_conditions is not working")
                print(f"own_path is {own_path} but should be {[[0.0],[0.0]]}")
                return 1
            elif int_path != [[rho*math.sin(-((theta*math.pi)/2))],[rho*math.cos(-((theta*math.pi)/2))]]:
                print(f"create_initial_conditions is not working")
                print(f"int_path is {int_path} but should be {[[rho*math.sin(-((theta*math.pi)/2))],[rho*math.cos(-((theta*math.pi)/2))]]}")
                return 1
            elif path_commands != ["COC"]:
                print(f"create_initial_conditions is not working")
                print(f"did not correctly creat path_commands array")
                return 1
            elif path_distances != [rho]:
                print(f"create_initial_conditions is not working")
                print(f"did not correctly creat path_distances array")
                return 1
    return 0

def test_find_psi():
    '''
    Checks if it returns correct angle for ownship and intruder in all possible pairs of quadrants
    '''
    own_state = [0,0,0,0]
    int_state = [0,0,0,0]
    correct_values = [0.0,1/4*math.pi,1/2*math.pi,3/4*math.pi,math.pi,-3/4*math.pi, -1/2*math.pi,-1/4*math.pi]

    for i in range(8):
        for j in range (8):
            own_state[2] = i/4*math.pi
            int_state[2] = j/4*math.pi
            psi = sim.find_psi(own_state, int_state)
            if j - i < 0:
                if psi != correct_values[(j-i+8)%8]:
                    print(f"find_psi is not working correctly")
                    print(f"inputs of own heading={own_state[2]}, int heading={int_state[2]}")
                    print(f"returned {psi} but should have returned {correct_values[(i+j)%8]}")
                    return 1
            else:
                if psi != correct_values[(j-i)%8]:
                    print(f"find_psi is not working correctly")
                    print(f"inputs of own heading={own_state[2]}, int heading={int_state[2]}")
                    print(f"returned {psi} but should have returned {correct_values[(i+j)%8]}")
                    return 1

    return 0

def test_find_rho():
    '''
    Checks all possible combinations of postive and negative x and y coordinates
    '''
    own_coords_loop = [[0,1,1,1,0,-1,-1,-1],[1,1,0,-1,-1,-1,0,1]]
    int_coords_loop = [[0,3,5,3,0,-3,-5,-3],[5,4,0,-4,-5,-4,0,4]]
    own_state = [0,0,0,0]
    int_state = [0,0,0,0]
    
    for i in range(len(own_coords_loop[0])):
        for j in range(len(own_coords_loop[0])):
            own_state[0] = own_coords_loop[0][i]
            own_state[1] = own_coords_loop[1][i]
            int_state[0] = own_state[0] + int_coords_loop[0][j]
            int_state[1] = own_state[1] + int_coords_loop[1][j]
            rho = sim.find_rho(own_state, int_state)
            if rho != 5:
                print(f"find_rho is not working correctly")
                print(f"inputs of own state={own_state}, int state={int_state}")
                print(f"returned {rho}, should have returned 5.0")
                return 1

    return 0

def test_find_theta():
    '''
    Checks all posible combinations of quadrants of coordinates and heading
    '''
    coords_loop = [[0,-1,-1,-1,0,1,1,1],[1,1,0,-1,-1,-1,0,1]]

    for i in range(8):
        for j in range(8):
            for k in range(8):
                own_state = [coords_loop[0][j], coords_loop[1][j], (i/4)*math.pi, 100]
                int_state = [coords_loop[0][j]+coords_loop[0][k], coords_loop[1][j]+coords_loop[1][k], 0, 100]
                correct_theta = ((k-i)/4)*math.pi
                while correct_theta <= -math.pi:
                    correct_theta += 2*math.pi
                while correct_theta > math.pi:
                    correct_theta -= 2*math.pi
                theta = sim.find_theta(own_state, int_state)
                if theta < correct_theta - 0.00001 or theta > correct_theta + 0.00001: #error neccessary due to rounding errors in computation
                    print(f"find_theta is not working correctly")
                    print(f"inputs of own state={own_state}, int state={int_state}")
                    print(f"returned {theta}, but should have been {correct_theta}")
                    return 1

    return 0

def test_find_time_to_LOS_number():
    '''
    Checks that each index is correctly returned
    '''
    for i in range(len(sim.LOS_TIMES)):
        own_state = [0,0,0, 100]
        int_state = [0,0,0, 50]
        int_state[1] = 499 + ((sim.LOS_TIMES[i] + 0.5) * (own_state[3]-int_state[3]))
        returned_index = sim.find_time_to_LOS_number(own_state, int_state)
        if returned_index != i+1:
            print(f"find_time_to_LOS_number function is not working correctly")
            print(f"should have returned {i+1}, instead gave {returned_index}")
            return 1

    return 0

def test_run_network():
    '''
    Checks that a network can be loaded and run, and that it returns a valid value
    '''
    own_state = [0.0, 0.0, 0.0, 250]
    int_state = [0.0, 20000.0, math.pi, 250]
    next_command = None

    next_command = sim.run_network(0, own_state, int_state)
    if next_command == None or next_command > 4:
        print(f"run_network not working correctly, did not return a valid command")
        return 1

    return 0


def test_step():
    '''
    Checks that test step correctly applies 20 steps along the aircrafts headings
    '''
    own_state = [0.0, 0.0, 0.0, 250]
    int_state = [0.0, 20000.0, math.pi, 250]
    own_path = [[0.0],[0.0]]
    int_path = [[0.0],[20000.0]]
    path_commands = ["COC"]
    path_distances = [20000.0]
    min_distance = 20000.0
    
    own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances = sim.step(0, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances)
    if own_path[0][len(own_path[0])-1] != 0 or own_path[1][len(own_path[1])-1] != 500.0:
        print(f"step is not working correctly in mapping Ownship's path")
        print(f"should be at (0.0,500.0) but is at ({own_path[0][len(own_path[0])-1]},{own_path[1][len(own_path[1])-1]})")
        return 1
    elif int_path[0][len(int_path[0])-1] != -6.123233995736766e-14 or int_path[1][len(int_path[1])-1] != 19500.0:
        print(f"step is not working correctly in mapping Intruder's path")
        print(f"should be at (-6.123233995736766e-14,19500.0) but is at ({int_path[0][len(int_path[0])-1]},{int_path[1][len(int_path[1])-1]})")
        return 1
    elif len(own_path[0]) != 21 or len(own_path[1]) != 21:
        print(f"step is not working correctly in mapping Ownship's path")
        print(f"should have 21 steps but has {len(own_path[0])} for x and {len(own_path[1])} for y")
        return 1
    elif len(int_path[0]) != 21 or len(int_path[1]) != 21:
        print(f"step is not working correctly in mapping Intruder's path")
        print(f"should have 21 steps but has {len(int_path[0])} for x and {len(int_path[1])} for y")
        return 1
    elif own_state[0] != own_path[0][len(own_path[0])-1] or own_state[1] != own_path[1][len(own_path[1])-1]:
        print(f"step is not keeping Ownship's state and path in line")
        printf("state is at ({own_state[0]},{own_state[1]}) and path is at ({own_path[0][len(own_path[0])-1]},{own_path[1][len(own_path[1])-1]})")
        return 1
    elif int_state[0] != int_path[0][len(int_path[0])-1] or int_state[1] != int_path[1][len(int_path[1])-1]:
        print(f"step is not keeping Intruder's state and path in line")
        print(f"state is at ({int_state[0]},{int_state[1]}) and path is at ({int_path[0][len(int_path[0])-1]},{int_path[1][len(int_path[1])-1]})")
        return 1
    elif len(path_commands) != 21 or path_commands[len(path_commands)-1] != "COC":
        print(f"step is not correctly updating path_commands")
        print(f"has {len(path_commands)} steps but should have 21")
        print(f'last command is {path_commands[len(path_commands)]} but should be "COC"')
        return 1
    elif len(path_distances) != 21 or path_distances[len(path_distances)-1] != 19000.0:
        print(f"step is not correctly updating path_distances")
        print(f"has {len(path_distances)} steps but should have 21")
        print(f"last distance is {path_distances[len(path_distances)]} but should be 19000.0")
        return 1
    elif min_distance != 19000.0:
        print(f"step is not correctly updating min_distance")
        print(f"is {min_distance} but should be 19000.0")
        return 1

    return 0

def system_test(draw=False):
    '''
    Tests the simulation system and prints out the returns
    Set to True to draw simulation as well, however drawing must be closed after viewing to print out results
    '''
    own_path, int_path, path_commands, path_distances = sim.run_simulation(10000, 1/4*math.pi, 3/2*math.pi, 250, 250, draw)
    own_path_string = f"Ownship: ({own_path[0][0]},{own_path[1][0]})"
    int_path_string = f"Intruder: ({int_path[0][0]},{int_path[1][0]})"
    for i in range(1, len(own_path[0])):
        own_path_string = own_path_string + f"->({own_path[0][i]},{own_path[1][i]})"
        int_path_string = int_path_string + f"->({int_path[0][i]},{int_path[1][i]})"
    print(own_path_string)
    print(int_path_string)
    print(path_distances)
    print(path_commands)
    
def run_all_unit_tests():
    failed = 0
    failed = test_create_initial_conditions()
    if failed == 0:
        failed = test_find_psi()
    if failed == 0:
        failed = test_find_rho()
    if failed == 0:
        failed = test_find_theta()
    if failed == 0:
        failed = test_find_time_to_LOS_number()
    if failed == 0:
        failed = test_run_network()
    if failed == 0:
        failed = test_step()

run_all_unit_tests()
system_test()
