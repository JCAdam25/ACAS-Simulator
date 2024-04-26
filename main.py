import math
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TIME_STEP = 0.1
##own_state = [0.0,0.0,0.0,0.0] #[x_coord, y_coord, heading, velocity]
##int_state = [0.0,0.0,0.0,0.0] #[x_coord, y_coord, heading, velocity]

##own_path = [[],[]] #[[x1,x2,...,xn],[y1,y2,...,yn]]
##int_path = [[],[]] #[[x1,x2,...,xn],[y1,y2,...,yn]]

COMMANDS_ACTION = [0.0, 1.5, -1.5, 3.0, -3.0]
COMMANDS_NAME = ["COC", "WL", "WR", "SL", "SR"]
COMMAND_COLOUR = ['red', 'orange', 'purple', 'yellow', 'blue']
CLOSEST_APPROACH_UPPER_LIMITS = [0.5,3,7.5,15,30,50,70,90,100]

def step(current_command, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances):
    '''
    Moves both aircraft along their pathz in 0.1 seconds jumps for total of 2 seconds travel
    For each step additionally updates Ownship's heading based on current command, records command being followed, and records difference between aircraft
    '''
    for i in range(0, int(2/TIME_STEP)):
        own_state[2] += math.radians(COMMANDS_ACTION[current_command]*TIME_STEP) # update Ownship's heading based on command, then update's location 
        own_state[0] += (own_state[3]*TIME_STEP)*math.sin(-own_state[2])
        own_state[1] += (own_state[3]*TIME_STEP)*math.cos(-own_state[2])

        int_state[0] += (int_state[3]*TIME_STEP)*math.sin(-int_state[2]) # updates Intruder's location
        int_state[1] += (int_state[3]*TIME_STEP)*math.cos(-int_state[2])

        own_path[0].append(own_state[0]) #adds next step to Ownship's and Intruder's path, list of previous commands, and list of previous distances
        own_path[1].append(own_state[1])
        int_path[0].append(int_state[0])
        int_path[1].append(int_state[1])
        path_commands.append(COMMANDS_NAME[current_command])
        path_distances.append(int(find_rho(own_state, int_state)))

        if path_distances[len(path_distances)-1] < min_distance: #cheks if current seperation is lower than the current minimum for the encounter and updates if appropriate
            min_distance = path_distances[len(path_distances)-1]

    return own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances

def find_theta(own_state, int_state):
    '''
    Finds the angle from the ownship to intruder
    '''
    dy = int_state[1] - own_state[1]
    dx = int_state[0] - own_state[0]
    theta = (3/2 * math.pi) + np.arctan2(dy, dx) - own_state[2]
    theta = theta % (2*math.pi)
    return theta

def find_psi(own_state, int_state):
    '''
    finds the heading of the intruder based on the heading of ownship
    '''
    psi = own_state[2] - int_state[2]
    return psi

def find_rho(own_state, int_state):
    '''
    find the distance from the ownship to intruder
    '''
    own_coords = [own_state[0], own_state[1]]
    int_coords = [int_state[0], int_state[1]]
    rho = math.dist(own_coords, int_coords)
    return rho

def find_time_to_LOS_number(own_state, int_state):
    '''
    Predicts the number of steps taken to get within NMAC zone if both aircraft continue on current heading
    Uses number of steps to find the time taken, then returns the network number that this time relates to
    '''
    steps_taken = 0
    distance = find_rho(own_state, int_state)
    own_x = own_state[0] #create seperate variables to store x and y so that state's aren't changed when passed by reference
    own_y = own_state[1]
    int_x = int_state[0]
    int_y = int_state[1]
    while distance > 500:
        steps_taken += 1
        own_x = own_x + ((own_state[3] * math.sin(-own_state[2])) * TIME_STEP)
        own_y = own_y + ((own_state[3] * math.cos(-own_state[2])) * TIME_STEP)
        int_x = int_x + ((int_state[3] * math.sin(-int_state[2])) * TIME_STEP)
        int_y = int_y + ((int_state[3] * math.cos(-int_state[2])) * TIME_STEP)
        if math.dist([own_x,own_y],[int_x,int_y]) > distance: #checks that the aircraft haven't already passed point of closest approach without breaking NMAC
            return -1
        distance = math.dist([own_x,own_y],[int_x,int_y])

    time_to_closest_approach = steps_taken*TIME_STEP
    for i in range(0, len(CLOSEST_APPROACH_UPPER_LIMITS)):  #finds correct network number based on time to loss of seperation
        if time_to_closest_approach <= CLOSEST_APPROACH_UPPER_LIMITS[i]:
            return i+1
        elif i == 8:
            return i+1
        

def load_network(last_cmd, time_to_CA):
    onnx_filename = f"ACASXU_files/ACASXU_run2a_{last_cmd + 1}_{time_to_CA + 1}_batch_2000.onnx"

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    session = ort.InferenceSession(onnx_filename)

    # warm up the network
    i = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    i.shape = (1, 1, 1, 5)
    session.run(None, {'input': i})

    return session, range_for_scaling, means_for_scaling

def load_networks():
    nets = []

    for last_cmd in range(5):
        for time_to_CA in range(9):
            nets.append(load_network(last_cmd, time_to_CA))

    return nets

def run_network(nets, last_command, own_state, int_state):
    time_to_closest_approach = find_time_to_LOS_number(own_state, int_state)
    rho = find_rho(own_state, int_state)
    theta = find_theta(own_state, int_state)
    psi = find_psi(own_state, int_state)

    inputs = [rho, theta, psi, own_state[3], int_state[3]]
    net_array = nets[((last_command)*9)+time_to_closest_approach]

    for i in range(5):
        inputs[i] = (inputs[i] - net_array[2][i]) / net_array[1][i]

    in_array = np.array(inputs, dtype=np.float32)
    in_array.shape = (1, 1, 1, 5)
    outputs = net_array[0].run(None, {'input': in_array})

    return np.argmax(outputs[0][0])

def create_initial_conditions(rho, theta, psi, v_own, v_int):
    own_state = [0.0,0.0,0.0,v_own] #[x_coord, y_coord, heading, velocity]
    int_state = [rho*math.sin(-theta),rho*math.cos(-theta),psi,v_int] #[x_coord, y_coord, heading, velocity]

    own_path = [[own_state[0]], [own_state[1]]]
    int_path = [[int_state[0]], [int_state[1]]]

    path_commands = ["COC"]
    path_distances = [rho]

    return own_state, int_state, own_path, int_path, path_commands, path_distances

def run_simulation(rho, theta, psi, v_own, v_int, visualise=True):
    own_state, int_state, own_path, int_path, path_commands, path_distances = create_initial_conditions(rho, theta, psi, v_own, v_int)
    nets = load_networks()
    min_distance = rho
    initial_net = find_time_to_LOS_number(own_state, int_state)
    current_command = run_network(nets, 0, own_state, int_state)
    while ((current_command != 0 and rho < 60760) or path_distances[len(path_distances)-1] < path_distances[len(path_distances)-2]):
        current_command = run_network(nets, current_command, own_state, int_state)
        own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances = step(current_command, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances)
        rho = find_rho(own_state, int_state)

    if visualise == True:
        print(f"The aircraft reached a minimum distance of {min_distance}")
        if min_distance <= 500:
            print(f"The ACAS-Xu system failed to prevent the intruder entering the NMAC zone")
        animate(own_path, int_path, path_commands, path_distances)
    else:
        if min_distance <= 500:
            return 1
        else:
            return 0

def animate(own_path, int_path, path_commands, path_distances):
    fig, ax = plt.subplots()
    line_segments_own = []
    line_segments_int = []
    text = ax.text(0.1, 0.95,
                   'matplotlib',
                   transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='center')

    def init():
        axes_min = min(min(own_path[0]), min(own_path[1]), min(int_path[0]), min(int_path[1]))
        axes_max = max(max(own_path[0]), max(own_path[1]), max(int_path[0]), max(int_path[1]))
        text.set_text(f'')
        edge = (axes_max-axes_min)*0.05
        ax.set_xlim(axes_min - edge, axes_max + edge)
        ax.set_ylim(axes_min - edge, axes_max + edge)
        return []

    def update(frame):
        if frame == 0:
            for i in range(len(COMMANDS_NAME)):
                if path_commands[frame] == COMMANDS_NAME[i]:
                    line_segments_own.append(ax.plot([], [], color=COMMAND_COLOUR[i], lw=2)[0])
            line_segments_int.append(ax.plot([], [], color='green', lw=2)[0])
        else:
            for i in range(len(line_segments_int), frame+1):
                for j in range(len(COMMANDS_NAME)):
                    if path_commands[i] == COMMANDS_NAME[j]:
                        line_segments_own.append(ax.plot([own_path[0][i - 1], own_path[0][i]],
                                                         [own_path[1][i - 1], own_path[1][i]],
                                                         color=COMMAND_COLOUR[j], lw=2)[0])
                line_segments_int.append(ax.plot([int_path[0][i - 1], int_path[0][i]],
                                                 [int_path[1][i - 1], int_path[1][i]],
                                                 color='green', lw=2)[0])
        text.set_text(f'Seperation between aircraft: {path_distances[frame]}\nEncounter time: {frame*TIME_STEP}')
        return line_segments_own + line_segments_int + [text]

    ani = FuncAnimation(fig, update, frames=range(0,len(own_path[0]),5), init_func=init, blit=True, interval=100, repeat=False)
    ax.plot([],[], color='green', label='Intruder')
    for i in range(len(COMMANDS_NAME)):
        ax.plot([],[], color=COMMAND_COLOUR[i], label=COMMANDS_NAME[i])
    plt.legend()
    plt.show()
