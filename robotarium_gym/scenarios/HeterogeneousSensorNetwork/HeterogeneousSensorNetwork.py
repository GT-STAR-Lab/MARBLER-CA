import numpy as np
from gym import spaces
import copy
import yaml
import os

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.HeterogeneousSensorNetwork.visualize import *
from robotarium_gym.scenarios.base import BaseEnv
from rps.utilities.graph import *

from robotarium_gym.utilities.misc import is_close
import numpy as np

class Agent:
    '''
    This is a helper class for PredatorCapturePrey
    Keeps track of information for each agent and creates functions needed by each agent
    This could optionally all be done in PredatorCapturePrey
    '''

    def __init__(self, index, radius, action_id_to_word, args):
        self.index = index
        self.radius = radius
        self.action_id2w = action_id_to_word
        self.args = args
    
    def generate_goal(self, goal_pose, action, args):
        '''
        Sets the agent's goal to step_dist in the direction of choice
        Bounds the agent by args.LEFT, args.RIGHT, args.UP and args.DOWN
        '''

        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - args.step_dist, args.LEFT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + args.step_dist, args.RIGHT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = max( goal_pose[1] - args.step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = min( goal_pose[1] + args.step_dist, args.DOWN)
        else:
             goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
             goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        return goal_pose
                


class HeterogeneousSensorNetwork(BaseEnv):
    def __init__(self, args):
        # Settings
        self.args = args

        module_dir = os.path.dirname(__file__)
        # with open(f'{module_dir}/predefined_agents.yaml', 'r') as stream:
        #     self.predefined_agents = yaml.safe_load(stream)
        
        if self.args.seed != -1:
            np.random.seed(self.args.seed)

        if(args.hard_coded_coalition):
            self.args.resample = False
            with open(f'{module_dir}/grid_search_coalitions.yaml', 'r') as stream:
                self.predefined_coalition = yaml.safe_load(stream)
        
        else:
            with open(f'{module_dir}/predefined_coalitions.yaml', 'r') as stream:
                self.predefined_coalition = yaml.safe_load(stream)

        self.num_robots = args.n_agents
        self.agent_poses = None # robotarium convention poses
        
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}

        #Initializes the agents
        self.agents = []

        self.agents = self.load_agents()

        if self.args.capability_aware:
            self.agent_obs_dim = 3
        else:
            self.agent_obs_dim = 2

        #initializes the actions and observation spaces
        actions = []
        observations = []      
        for i in range(self.num_robots):
            actions.append(spaces.Discrete(5))
            #The observations are bounded by the 
            observations.append(spaces.Box(low=-1.5, high=1.5, shape=(self.agent_obs_dim,), dtype=np.float32))        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))

        self.visualizer = Visualize( self.args )
        self.env = roboEnv(self, args)
        self.adj_matrix = 1-np.identity(self.num_robots, dtype=int)
             

    def _generate_step_goal_positions(self, actions):
        '''
        User implemented
        Calculates the goal locations for the current agent poses and actions
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:,i] = agent.generate_goal(goal[:,i], actions[i], self.args)
        return goal

    def load_agents(self):
        '''Loades the pre-defined agents / coalitions
        '''
        t = "train"
        if self.args.test:
            t = "test"
        agents = []
        
        # sample a new coalition
        if(self.args.manual_coalition_selection):
            coalition_idx = self.args.coalition_selection
        else:
            coalition_idx = np.random.randint(self.args.n_coalitions)
            
        # coalition_idx = self.args.coalition_idx
        s = str(self.num_robots) + "_agents"
        agents = self.predefined_coalition[t]["coalitions"][s][coalition_idx]
        
        index = 0
        for idx, agent in agents.items():
            agents.append(Agent(index, agent["radius"], self.action_id2w, self.args))
            index += 1
        return agents
    
    def reset(self):
        '''
        Resets the simulation
        '''
        if self.args.resample:
            # #Initializes the agents
            self.agents = self.load_agents()

        #Generate the agent locations based on the config
        width = self.args.RIGHT - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        #Agents can spawn anywhere in the Robotarium between UP, DOWN, LEFT and RIGHT for this scenario
        self.agent_poses = generate_initial_conditions(self.num_robots, spacing=self.args.start_dist, width=width, height=height)
        #Adjusts the poses based on the config
        self.agent_poses[0] += (1.5 + self.args.LEFT)/2
        self.agent_poses[0] -= (1.5 - self.args.RIGHT)/2
        self.agent_poses[1] -= (1+self.args.UP)/2
        self.agent_poses[1] += (1-self.args.DOWN)/2

        self.episode_steps = 0
        
        self.state_space = self._generate_state_space()
        self.env.reset()
        return [[0]*self.agent_obs_dim] * self.num_robots
        
    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info
        '''
        terminated = False
        self.episode_steps += 1

        # call the environment step function and get the updated state
        return_message = self.env.step(actions_)
        
        self._update_tracking_and_locations(actions_)
        updated_state = self._generate_state_space()
        
        # get the observation and reward from the updated state
        obs     = self.get_observations(updated_state)
        rewards = self.get_rewards(updated_state)

        # penalize for collisions, record in info
        violation_occurred = 0
        if self.args.penalize_violations:
            if self.args.end_ep_on_violation and return_message != '':
                violation_occurred += 1
                # print("violation: ", return_message)
                rewards += self.args.violation_penalty
                terminated=True
            elif not self.args.end_ep_on_violation:
                violation_occurred = return_message
                rewards +=  np.log(return_message+1) * self.args.violation_penalty #Taking the log because this can get out of control otherwise
        
        # terminate if needed
        if self.episode_steps > self.args.max_episode_steps:
            terminated = True    

        info = {
                "violation_occurred": violation_occurred, # not a true count, just binary for if ANY violation occurred
                } 
        
        return obs, [rewards]*self.num_robots, [terminated]*self.num_robots, info

    def get_observations(self, state_space):
        observations = [] #Each agent's individual observation
        neighbors = [] #Stores the neighbors of each agent if delta > -1
        for a in self.agents:
            if self.args.capability_aware:
                if self.args.dual_channel:
                    observations.append([[*self.agent_poses[:, a.index ][:2]], [a.radius]])
                else:
                    observations.append([*self.agent_poses[:, a.index ][:2], a.radius])
            else:
                observations.append([*self.agent_poses[:, a.index ][:2]])
            if self.args.delta > -1:
                neighbors.append(delta_disk_neighbors(self.agent_poses,a.index,self.args.delta))

    def get_rewards(self, state_space):
        # Fully shared reward, this is a collaborative environment.
        reward = 0

        for a1 in self.agents:
            for a2 in self.agents:
                dist = np.linalg.norm(self.agent_poses[:2, a1.index]) - np.linalg.norm(self.agent_poses[:2, a2.index])
                reward += abs(dist - (a1.radius + a2.radius)) * self.args.dist_reward_multiplier

        reward += min([np.linalg.norm(self.agent_poses[:2, a.index] - [0, 0]) for a in self.agents]) * self.args.dist_reward_multiplier
        return reward
    
    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space
