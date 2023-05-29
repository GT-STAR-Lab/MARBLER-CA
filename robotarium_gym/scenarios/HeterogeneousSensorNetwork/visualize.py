from rps.utilities.misc import *
from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self, args):
        self.agent_marker_size_m = .15
        self.line_width = 3
        self.CM = plt.cm.get_cmap('bone')
        self.show_figure = True
    
    def initialize_markers(self, robotarium, agents):
        agent_marker_size = determine_marker_size(robotarium, self.agent_marker_size_m)
        self.robot_markers = [ robotarium.axes.scatter( \
                agents.agent_poses[0,ii], agents.agent_poses[1,ii],
                s=agent_marker_size, marker='o', facecolors='none',\
                edgecolors = (self.CM(agents.agents[ii].radius)), linewidth=self.line_width )\
                for ii in range(agents.num_robots) ]
    
    def update_markers(self, robotarium, agents ):
        for i in range(agents.agent_poses.shape[1]):
            self.robot_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            # Next two lines updates the marker sizes if the figure window size is changed.
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, self.agent_marker_size_m)])