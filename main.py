import math
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TIME_STEP = 0.1 # time covered in each step forward
COMMANDS_ACTION = [math.radians(0.0), math.radians(1.5), math.radians(-1.5), math.radians(3.0), math.radians(-3.0)] # degrees to be turned left per command, stored in radian form
COMMANDS_NAME = ["COC", "WL", "WR", "SL", "SR"]
COMMAND_COLOUR = ['red', 'orange', 'purple', 'yellow', 'blue']
LOS_TIMES = [0,1,5,10,20,40,60,80,100] # minimum time to Loss of Separation per index
MEANS_FOR_SCALING = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
RANGES_FOR_SCALING = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

def create_initial_conditions(rho, theta, psi, v_own, v_int):
    '''
    Uses the inputed conditions to generate the aircraft's states and other requried arrays
    '''
    own_state = [0.0,0.0,0.0,v_own] #[x_coord, y_coord, heading, velocity]
    int_state = [rho*math.sin(-theta),rho*math.cos(-theta),psi,v_int] #[x_coord, y_coord, heading, velocity]

    own_path = [[own_state[0]], [own_state[1]]] #[[x_coord_1,...],[y_coord_1,...]]
    int_path = [[int_state[0]], [int_state[1]]] #[[x_coord_1,...],[y_coord_1,...]]

    path_commands = ["COC"] #[current command for this step]
    path_distances = [rho] #[current seperation at this step]

    return own_state, int_state, own_path, int_path, path_commands, path_distances

def find_psi(own_state, int_state):
    '''
    finds the heading of the intruder based on the heading of ownship
    '''
    psi = int_state[2] - own_state[2]
    while psi > math.pi: # ensure that psi stays between -pi < heading <= pi
        psi -= 2*math.pi
    while psi <= -math.pi:
        psi += 2*math.pi
    return psi

def find_rho(own_state, int_state):
    '''
    find the distance from the ownship to intruder
    '''
    own_coords = [own_state[0], own_state[1]]
    int_coords = [int_state[0], int_state[1]]
    rho = math.dist(own_coords, int_coords)
    return rho

def find_theta(own_state, int_state):
    '''
    Finds the angle from the ownship to intruder
    '''
    dy = int_state[1] - own_state[1]
    dx = int_state[0] - own_state[0]
    theta = (np.arctan2(dy, dx) - np.arctan2(1, 0)) - own_state[2]
    while theta <= -math.pi:
        theta += 2*math.pi
    while theta > math.pi:
        theta -= 2*math.pi
    return theta

def find_time_to_LOS_number(own_state, int_state):
    '''
    Predicts the number of steps taken to get within NMAC zone if both aircraft continue on current heading
    Uses number of steps to find the time taken, then returns the network number that this time relates to
    '''
    steps_taken = 0
    distance = find_rho(own_state, int_state)
    own_x = own_state[0] # create seperate variables to store x and y so that state's aren't changed when passed by reference
    own_y = own_state[1]
    int_x = int_state[0]
    int_y = int_state[1]
    while distance > 500:
        steps_taken += 1
        own_x = own_x + ((own_state[3] * math.sin(-own_state[2])) * TIME_STEP)
        own_y = own_y + ((own_state[3] * math.cos(-own_state[2])) * TIME_STEP)
        int_x = int_x + ((int_state[3] * math.sin(-int_state[2])) * TIME_STEP)
        int_y = int_y + ((int_state[3] * math.cos(-int_state[2])) * TIME_STEP)
        
        if math.dist([own_x,own_y],[int_x,int_y]) > distance: # checks that the aircraft hasn't already passed point of closest approach without breaking NMAC
            return len(LOS_TIMES)
        distance = math.dist([own_x,own_y],[int_x,int_y])

    time_to_LoS = steps_taken*TIME_STEP
    for i in range(1, len(LOS_TIMES)):  # finds correct network number based on time to loss of seperation
        if time_to_LoS < LOS_TIMES[i]:
            return i

    return len(LOS_TIMES) # returns max value if no other has been given

def run_network(last_command, own_state, int_state):
    '''
    Uses states to generate the correct inputs, and select the correct network
    Runs the network, and uses the returned weightings to return the next command
    '''
    time_to_LOS = find_time_to_LOS_number(own_state, int_state)
    network = ort.InferenceSession(f"ACASXU_files/ACASXU_run2a_{last_command + 1}_{time_to_LOS}_batch_2000.onnx")
    rho = find_rho(own_state, int_state)
    theta = find_theta(own_state, int_state)
    psi = find_psi(own_state, int_state)

    conditions = [rho, theta, psi, own_state[3], int_state[3]]
    print(f"conditions: {conditions}")
    for i in range(5):
        conditions[i] = (conditions[i] - MEANS_FOR_SCALING[i]) / RANGES_FOR_SCALING[i]
    net_input = np.array(conditions, dtype=np.float32)
    net_input.shape = (1,1,1,5)

    weightings = network.run(None, {'input':net_input})
    print(f"alt command: {np.argmin(weightings)}")
    return np.argmin(weightings[0][0])


def run_simulation(rho, theta, psi, v_own, v_int, visualise=True, extended_sim=False):
    '''
    Main function for running a simulation
    visualise=False can be used to not draw the encounter
    '''
    own_state, int_state, own_path, int_path, path_commands, path_distances = create_initial_conditions(rho, theta, psi, v_own, v_int)
    min_distance = rho
    current_command = 1
    first_step = True
    passed = False
    rho = rho
    while rho < 1000 or passed == False:
        if first_step == True:
            current_command = 0
            first_step = False
        current_command = run_network(current_command, own_state, int_state)
        own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances = step(current_command, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances)
        if find_rho(own_state, int_state) > rho:
            passed = True
        rho = find_rho(own_state, int_state)

    if extended_sim == True:
        passed = False
        if len(path_distances) == 1:
            own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances = step(0, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances)
        while rho < 1000 or passed == False:
            own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances = step(0, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances)
            if find_rho(own_state, int_state) > rho:
                passed = True
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

def step(current_command, own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances):
    '''
    Moves both aircraft along their paths in 0.1 seconds jumps for total of 2 seconds travel
    For each step additionally updates Ownship's heading based on current command, records command being followed, and records difference between aircraft
    '''
    for i in range(0, int(2/TIME_STEP)):
        own_state[2] += COMMANDS_ACTION[current_command]*TIME_STEP # update Ownship's heading based on command, then update's location
        if own_state[2] >= 2*math.pi: # ensure that own_state stays between 0 < heading < 2pi
            own_state[2] -= 2*math.pi
        elif own_state[2] < 0:
            own_state[2] += 2*math.pi

        own_state[0] += (own_state[3]*TIME_STEP)*math.sin(-own_state[2])
        own_state[1] += (own_state[3]*TIME_STEP)*math.cos(-own_state[2])

        int_state[0] += (int_state[3]*TIME_STEP)*math.sin(-int_state[2]) # updates Intruder's location
        int_state[1] += (int_state[3]*TIME_STEP)*math.cos(-int_state[2])

        own_path[0].append(own_state[0]) # adds next step to Ownship's and Intruder's path, list of previous commands, and list of previous distances
        own_path[1].append(own_state[1])
        int_path[0].append(int_state[0])
        int_path[1].append(int_state[1])
        path_commands.append(COMMANDS_NAME[current_command])
        path_distances.append(int(find_rho(own_state, int_state)))

        if path_distances[len(path_distances)-1] < min_distance: # checks if current seperation is lower than the current minimum for the encounter and updates if appropriate
            min_distance = path_distances[len(path_distances)-1]
    print(f"own path: ({own_path[0][len(own_path[0])-1]},{own_path[1][len(own_path[1])-1]})")
    print(f"int path: ({int_path[0][len(int_path[0])-1]},{int_path[1][len(int_path[1])-1]})")
    return own_state, own_path, int_state, int_path, min_distance, path_commands, path_distances

def animate(own_path, int_path, path_commands, path_distances):
    '''
    Draws an animated plot of an encounter, rendering both aircrafts flight paths
    Colours Ownship's path depending on the command being followed
    Writes out separation at each step
    '''

    def init():
        '''
        Initialises the plot
        '''
##        x_min = min(min(own_path[0]), min(int_path[0]))
##        x_max = max(max(own_path[0]), max(int_path[0]))
##        y_min = min(min(own_path[1]), min(int_path[1]))
##        y_max = max(max(own_path[1]), max(int_path[1]))
##        text.set_text(f'')
##        x_edge = (x_max-x_min)*0.05
##        y_edge = (y_max-y_min)*0.05
##        ax.set_xlim(x_min - x_edge, x_max + x_edge)
##        ax.set_ylim(y_min - y_edge, y_max + y_edge)

        axis_min = min(min(own_path[0]), min(int_path[0]), min(own_path[1]), min(int_path[1]))
        axis_max = max(max(own_path[0]), max(int_path[0]), max(own_path[1]), max(int_path[1]))
        text.set_text(f'')
        edge = (axis_max-axis_min)*0.05
        ax.set_xlim(axis_min - edge, axis_max + edge)
        ax.set_ylim(axis_min - edge, axis_max + edge)
        return []

    def update(frame):
        '''
        Creates each new frame for the next step along the journey
        '''
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

    fig, ax = plt.subplots()
    line_segments_own = []
    line_segments_int = []
    text = ax.text(0.1, 0.95,
                   'matplotlib',
                   transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='center')

    ani = FuncAnimation(fig, update, frames=range(0,len(own_path[0]),5), init_func=init, blit=True, interval=100, repeat=False)
    ax.plot([],[], color='green', label='Intruder')
    for i in range(len(COMMANDS_NAME)):
        ax.plot([],[], color=COMMAND_COLOUR[i], label=COMMANDS_NAME[i])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    init_rho = -1
    while init_rho < 10000 or init_rho > 60000:
        init_rho = float(input("Distance between the Ownship and Intruder(ft): "))
    init_theta = -1
    while init_theta < 0 or init_rho >= 2*math.pi:
        init_theta = float(input("Angle to Intruder with regards to Ownship heading(rad): "))
    init_psi = -1
    while init_psi < 0 or init_psi >= 2*math.pi:
        init_psi = float(input("Heading of Intruder with regards to Ownship heading(rad): "))
    init_v_own = -1
    while init_v_own < 100 or init_v_own > 1200:
        init_v_own = float(input("Speed of Ownship(ft/s): "))
    init_v_int = -1
    while init_v_int < 100 or init_v_int > 1200:
        init_v_int = float(input("Speed of Intruder(ft/s): "))

    extend_sim = bool(input("Extend sim till aircraft are past(True/False"))

    run_simulation(init_rho, init_theta, init_psi, init_v_own, init_v_int, True, extend_sim)
