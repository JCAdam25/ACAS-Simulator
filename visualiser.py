import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import simulator as sim

COMMANDS_COLOURS = ['red', 'orange', 'purple', 'yellow', 'blue']

def animate(own_path, int_path, path_commands, frames_per_second = 2):
    '''
    Draws an animated plot of an encounter, rendering both aircrafts flight paths
    Colours Ownship's path depending on the command being followed
    Writes out separation at each step
    '''
    Interval = int(1000/frames_per_second)
    def init():
        '''
        Initialises the plot
        '''
        axis_min = min(min(own_path[0]), min(int_path[0]), min(own_path[1]), min(int_path[1]))
        axis_max = max(max(own_path[0]), max(int_path[0]), max(own_path[1]), max(int_path[1]))
        text.set_text(f'')
        edge = (axis_max-axis_min)*0.05
        ax.set_xlim(axis_min-edge, axis_max+edge)
        ax.set_ylim(axis_min-edge, axis_max+edge)
        return []

    def update(frame):
        '''
        Creates each new frame for the next step along the journey
        '''
        if frame == 0:
            for i in range(len(sim.COMMANDS_NAMES)):
                if path_commands[frame] == sim.COMMANDS_NAMES[i]:
                    line_segments_own.append(ax.plot([], [], color=COMMANDS_COLOURS[i], lw=2)[0])
            line_segments_int.append(ax.plot([], [], color='green', lw=2)[0])
        else:
            for i in range(len(line_segments_int), frame+1):
                for j in range(len(sim.COMMANDS_NAMES)):
                    if path_commands[i] == sim.COMMANDS_NAMES[j]:
                        line_segments_own.append(ax.plot([own_path[0][i - 1], own_path[0][i]],
                                                         [own_path[1][i - 1], own_path[1][i]],
                                                         color=COMMANDS_COLOURS[j], lw=2)[0])
                line_segments_int.append(ax.plot([int_path[0][i - 1], int_path[0][i]],
                                                 [int_path[1][i - 1], int_path[1][i]],
                                                 color='green', lw=2)[0])
        separation = int(math.dist([own_path[0][frame],own_path[1][frame]], [int_path[0][frame], int_path[1][frame]]))
        text.set_text(f'Separation between aircraft: {separation}\nEncounter time: {frame*sim.TIME_STEP}')
        return line_segments_own + line_segments_int + [text]

    fig, ax = plt.subplots()
    line_segments_own = []
    line_segments_int = []
    text = ax.text(0.1, 0.95,
                   'matplotlib',
                   transform=ax.transAxes,
                   horizontalalignment='left',
                   verticalalignment='center')

    ani = FuncAnimation(fig, update, frames=range(0,len(own_path[0]),5), init_func=init, blit=True, interval=Interval, repeat=False)
    ax.plot([],[], color='green', label='Intruder')
    for i in range(len(sim.COMMANDS_NAMES)):
        ax.plot([],[], color=COMMANDS_COLOURS[i], label=sim.COMMANDS_NAMES[i])
    plt.title('A figure showing the flight paths of the Intruder and Ownship')
    plt.xlabel('Distance east from start point of the Ownship (ft)')
    plt.ylabel('Distance north from start point of the Ownship (ft)')
    plt.legend()
    plt.show()
