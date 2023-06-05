import numpy as np
from gym import spaces
import copy
import yaml
import os
from copy import deepcopy
import math

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

    def __init__(self, index, id, radius, action_id_to_word, args):
        self.index = index
        self.id = id
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
            with open(f'{module_dir}/{args.coalition_file}', 'r') as stream:
                self.predefined_coalition = yaml.safe_load(stream)

        self.num_robots = args.n_agents
        self.agent_poses = None # robotarium convention poses
        self.episode_number = 0

        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}

        #Initializes the agents
        self.agents = []

        if self.args.load_from_predefined_coalitions:
            # #Initializes the agents
            self.agents = self.load_agents_from_predefined_coalitions()
        elif(self.args.load_from_predefined_agents):
            self.agents = self.load_new_coalition_from_predefined_agents()
            print("Loading from Predefined Agents")
        else:
            self.agents = self.load_agents_from_trait_distribution()


        if self.args.capability_aware:
            self.agent_obs_dim = 3
        elif self.args.agent_id: # agent ids are one hot encoded
            self.agent_obs_dim = 2 + self.num_robots * self.args.n_coalitions
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
    
    def load_agents_from_trait_distribution(self):
        """Loades agents/coaltions from by sampling each agent individually from trait distribution
        """
        agents = []
        func_args = copy.deepcopy(self.args.traits['radius'])
        del func_args['distribution']

        index = 0
        for idx in range(self.num_robots):
            radius_val = float(getattr(np.random, self.args.traits["radius"]['distribution'])(**func_args))
            default_id = ['0'] * (self.num_robots * self.args.n_coalitions)
            agents.append(Agent(index, default_id, radius_val, self.action_id2w, self.args))
            index += 1
    
        return agents

    def load_agents_from_predefined_coalitions(self):
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
        coalition = self.predefined_coalition[t]["coalitions"][s][coalition_idx]
        
        index = 0
        for idx, agent in coalition.items():
            agents.append(Agent(index, agent["id"], agent["radius"], self.action_id2w, self.args))
            index += 1
        return agents

    def load_new_coalition_from_predefined_agents(self):
        '''Loades the pre-defined agents, and draws a random coalition from them
        '''
        t = "train"
        if self.args.test:
            t = "test"
        
        agent_pool = []
        agents = []
            
        # coalition_idx = self.args.coalition_idx
        s = str(self.num_robots) + "_agents"
        for coalition_idx in range(self.args.n_coalitions):
            coalition = self.predefined_coalition[t]["coalitions"][s][coalition_idx]
            
            index = 0
            for idx, agent in coalition.items():
                agent_pool.append(Agent(index, agent["id"], agent["radius"], self.action_id2w, self.args))
                index += 1

        # sample a coalitions
        for idx in range(self.num_robots):
            agents.append(random.choice(agent_pool))
        return(agents)
    

    def reset(self):
        '''
        Resets the simulation
        '''
        self.episode_number += 1
        if self.args.resample and (self.episode_number % self.args.resample_frequency == 0):
            if self.args.load_from_predefined_coalitions:
                self.agents = self.load_agents_from_predefined_coalitions()
            elif(self.args.load_from_predefined_agents):
                self.agents = self.load_new_coalition_from_predefined_agents()
            else:
                self.agents = self.load_agents_from_trait_distribution()

        # shuffles the order of agents
        if self.args.shuffle_agent_order:
            self.agents = self.shuffle_agents(self.agents)
        
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
        
        # get the observation and reward from the updated state
        obs     = self.get_observations()
        rewards, edges, total_overlap = self.get_rewards()

        # penalize for collisions, record in info
        violation_occurred = 0
        if self.args.penalize_violations:
            if self.args.end_ep_on_violation and return_message != '':
                violation_occurred += 1
                rewards += self.args.violation_penalty
                # terminated=True
            elif not self.args.end_ep_on_violation:
                violation_occurred += 1
                # rewards += self.args.violation_penalty
                violation_occurred = return_message
                rewards +=  np.log(return_message+1) * self.args.violation_penalty #Taking the log because this can get out of control otherwise
        
        # terminate if needed
        if self.episode_steps > self.args.max_episode_steps:
            terminated = True    

        info = {
                "violation_occurred": violation_occurred, # not a true count, just binary for if ANY violation occurred
                "connectivity": edges,
                "total_overlap": total_overlap
                } 
        
        return obs, [rewards]*self.num_robots, [terminated]*self.num_robots, info

    def get_observations(self):
        observations = [] #Each agent's individual observation
        neighbors = [] #Stores the neighbors of each agent if delta > -1

        def one_hot_encode(number, num_classes):
            encoded = [0] * num_classes
            encoded[number] = 1
            return encoded

        for a in self.agents:
            if self.args.capability_aware:
                if self.args.dual_channel:
                    observations.append([[*self.agent_poses[:, a.index ][:2]], [a.radius]])
                else:
                    observations.append([*self.agent_poses[:, a.index ][:2], a.radius])
            elif self.args.agent_id:    # append agent id
                agent_id = [int(bit) for bit in a.id]
                observations.append([*self.agent_poses[:, a.index ][:2], *agent_id])
            else:
                observations.append([*self.agent_poses[:, a.index ][:2]])
            if self.args.delta > -1:
                neighbors.append(delta_disk_neighbors(self.agent_poses,a.index,self.args.delta))

        return(observations)
    
    def get_rewards(self):
        # Fully shared reward, this is a collaborative environment.
        reward = 0
        center_reward = []
        edges = 0
        total_overlap = 0 if self.args.calculate_total_overlap else -1

        #The agents goal is to get their radii to touch
        for i, a1 in enumerate(self.agents):
            # reward agent if they are more towards the center.
            x1, y1 = self.agent_poses[:2, a1.index]
            center_reward.append(np.sqrt(np.sum(np.square(self.agent_poses[:2, a1.index] - np.array([0, 0]))))) # push agents towards center
            for j, a2 in enumerate(self.agents[i+1:],i+1): # don't duplicate
                x2, y2 = self.agent_poses[:2, a2.index]
                dist = np.sqrt(np.sum(np.square(self.agent_poses[:2, a1.index] - self.agent_poses[:2, a2.index])))
                # dist = np.linalg.norm(self.agent_poses[:2, a1.index]) - np.linalg.norm(self.agent_poses[:2, a2.index])
                difference = dist - (a1.radius + a2.radius)
                # reward += -1*abs(difference)

                if(self.args.calculate_total_overlap):
                    r1 = a1.radius; r2 = a2.radius
                    overlap = 0.0

                    if dist < r1 + r2:
                        if dist <= abs(r1 - r2):
                            # One circle is fully inside the other
                            overlap = math.pi * min(r1, r2)**2
                        else:
                            # Partial overlap
                            theta1 = math.acos((r1**2 + dist**2 - r2**2) / (2 * r1 * dist))
                            theta2 = math.acos((r2**2 + dist**2 - r1**2) / (2 * r2 * dist))
                            overlap = theta1 * r1**2 + theta2 * r2**2 - 0.5 * r1**2 * math.sin(2 * theta1) - 0.5 * r2**2 * math.sin(2 * theta2)
                    
                    total_overlap += overlap

                #incur more penatly if the agents boundaries are not touching
                if(difference < 0): # agents are touching
                    edges += 1
                    reward += -0.9 * abs(difference) + 0.05
                else:
                    reward += -1.1 * abs(difference) - 0.05 
                
                # reward += abs(dist - (a1.radius + a2.radius)) * self.args.dist_reward_multiplier
        
        #This is to center the agents in the middle of the field
        # reward += min([np.linalg.norm(self.agent_poses[:2, a.index] - [0, 0]) for a in self.agents]) * self.args.dist_reward_multiplier
        reward += -1*min(center_reward) * self.args.dist_reward_multiplier
        return reward, edges, total_overlap

    def shuffle_agents(self, agents):
        """
        Shuffle the order of agents
        """
        agents_ = deepcopy(agents)
        random.shuffle(agents_)
        return(agents_)
    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space
