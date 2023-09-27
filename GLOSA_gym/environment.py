import configparser
import os
import traci
from gym import Env
from gym.spaces import Box
import numpy as np
import random
import xml.etree.ElementTree as ET
from GLOSA_gym.state import VehicleObserver
from GLOSA_gym.settings import configure_sumo
from scipy.stats import norm
import configparser
from typing import Tuple, List
from GLOSA_gym.utils import import_sumo_tools


class SumoGlosaEnv(Env):
    def __init__(self, config: configparser.ConfigParser, sumo_path: str, evaluate: bool = False, evaluator=None,
                 gui: bool = False, ad: int = 0):
        super().__init__()
        """
        :param config: configparser.ConfigParser containing the config.ini
        :param sumo_path: path to the sumo cfg file
        :param evaluate: bool, if True, the environment is used for evaluation
        :param evaluator: Evaluator object, only used if evaluate is True
        :param gui: bool, if True, the sumo gui is used
        :param ad: int, used for evaluation, the ad to the base time that should be evaluated
        """
        self.sumo_path = sumo_path
        self.evaluate = evaluate
        self.evaluator = evaluator
        self.base = config.getint('Simulation', 'base_starttime')
        self.seed = 0
        self.config = config
        random.seed = self.seed
        self.sumo_cfg = config.get('Simulation', 'sumo_cfg')
        if not evaluate:
            self.ad = random.randint(1, 3600)
        else:
            self.ad = ad
        self.generate_ego()
        print('finished generating ego route')
        self.observer = VehicleObserver(config)
        print('finished creating VehicleObserver object')
        self.num_states = self.observer.get_state_dim()
        print('finished getting state dim')
        if gui:
            self.gui = True
        else:
            self.gui = config.getboolean('Simulation', 'gui')
        self._sumo_cmd = configure_sumo(self.gui, self.sumo_path, self.sumo_cfg, self.ad + self.base - 180)
        traci.start(self._sumo_cmd)
        self.state = [0] * self.num_states
        self.observation_space = Box(low=np.array([0 for _ in range(self.num_states)]),
                                     high=np.array([np.inf for _ in range(self.num_states)]))
        self.reward = 0
        self.action_space = Box(low=np.array([-1 for _ in range(1)]), high=np.array([1 for _ in range(1)]))
        self.steps = config.getint('GLOSA_general', 'steps')
        if config.get('GLOSA_general', 'glosa_agent') == 'rl':
            self.agent = 'rl'
        else:
            self.agent = 'classic'
        self.acc = config.getfloat('Simulation', 'ego_acc')
        self.last_update = 0

    def step(self, action: int, pre_eval=False) -> Tuple[List[float], float, bool, dict]:
        self.observer.current_action = action
        self.observer.action_deque.append(action)
        rl_action = action
        while True:
            done = False
            if 'ego' in traci.vehicle.getIDList():

                # set the deceleration of the ego vehicle according to the configuation
                traci.vehicle.setDecel('ego', self.acc)
                if self.observer.starttime is None:
                    self.observer.starttime = traci.simulation.getTime()

                # get the action from the agent
                if self.agent == 'rl':
                    action = map_values(action, new_max=self.config.getfloat('Simulation', 'max_speed'),
                                        new_min=self.config.getfloat('Simulation', 'min_speed'))

                    # include the human response time to the action
                    if execute(self.last_update, function='cdf'):
                        traci.vehicle.setSpeed('ego', action)
                        self.last_update = 0
                    else:
                        self.last_update += 1
                elif self.agent == 'classic':
                    if action == None:
                        pass
                    else:
                        if pre_eval:
                            action = map_values(action, new_max=self.config.getfloat('Simulation', 'max_speed'),
                                                new_min=self.config.getfloat('Simulation', 'min_speed'))
                        traci.vehicle.setSpeed('ego', action)
                next_tls = traci.vehicle.getNextTLS('ego')[0][0]

                # execute the action for the number of steps defined in the config
                for step in range(self.steps):
                    if not 'ego' in traci.vehicle.getIDList() or len(traci.vehicle.getNextTLS('ego')) == 0:
                        done = True
                        self.state = self.observer.return_state()
                        break
                    self.observer.update_tls()
                    self.observer.rewards.append(self.observer.get_reward())
                    self.observer.states.append(self.observer.get_state())
                    if self.gui:
                        traci.gui.trackVehicle('View #0', 'ego')
                        traci.gui.setZoom('View #0', 1000)
                    traci.simulation.step()
                    if self.evaluate:
                        self.evaluator.get_info()
                    if not 'ego' in traci.vehicle.getIDList() or len(traci.vehicle.getNextTLS('ego')) == 0:
                        done = True
                        self.observer.rewards.append(self.observer.get_reward())
                        break
                    if next_tls != traci.vehicle.getNextTLS('ego')[0][0]:
                        print('next tls changed')
                        break
                self.state = self.observer.return_state()
                self.reward = self.observer.return_reward()
                self.observer.old_action = rl_action
                break
            else:
                traci.simulation.step()
        if done:
            traci.close()
        return self.state, self.reward, done, {}

    def reset(self) -> List[float]:
        self.generate_ego()
        self._sumo_cmd = configure_sumo(self.gui, self.sumo_path, 'config_a9-thi.sumocfg', self.ad + self.base - 180)
        try:
            print('open')
            traci.start(self._sumo_cmd)
        except:
            print('reset')
            traci.close()
            print('open')
            traci.start(self._sumo_cmd)

        self.state = [0] * self.num_states
        self.observer.starttime = None
        return self.state

    def generate_ego(self) -> None:
        '''
        generates the ego vehicle with the specified route and departure time
        '''
        if not self.evaluate:
            self.ad = random.randint(1, 3600)
        print(f'generating ego at {self.base + self.ad}')

        tree = ET.parse(os.path.join(self.sumo_path, 'test.rou.xml'))
        root = tree.getroot()
        # create the new vehicle element
        vehicle = ET.Element("vehicle", id="ego", type="standard_car", depart=str(self.ad + self.base),
                             departLane="free",
                             departSpeed="max", color="0,1,0", )
        route = ET.Element("route",
                           edges=self.config.get('Simulation', 'ego_route'))
        vehicle.append(route)
        # find the "routes" element and append the new vehicle element to it
        for i, v in enumerate(root):
            if 'depart' in v.keys():
                if float(v.attrib['depart']) > (self.base + self.ad):
                    break

        root.insert(i, vehicle)
        # write the modified XML to a new file
        tree.write(os.path.join(self.sumo_path, "add_ego.rou.xml"))
        print('added ego')


def map_values(original_value, new_min=6.94, new_max=13.89):
    original_min = -1
    original_max = 1
    mapped_value = ((original_value - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min
    return mapped_value


def execute(x, function=None) -> bool:
    """
    This function tries to imite the human response time by executing an action with a certain probability depending on the
    last time an action was executed.
    :param x: time since last update
    :param function: function to determine if action should be executed
    :return: True if execution of action should be performed else False
    """
    rand = random.random()

    # always execute action
    if function is None:
        return True

    # cumulated density function according to https://ieeexplore.ieee.org/document/9829229
    elif function == 'cdf':
        mu = 4.8
        sigma = 2.5
        threshold = norm.cdf(x, mu, sigma)

    # linear increase of probability to execute action
    elif function == 'linear':
        if x <= 5:
            slope = (0.75 - 0.2) / 5
            threshold = 0.2 + slope * x
        else:
            threshold = 0.75
    else:
        raise ValueError('execution function not defined')

    if rand <= threshold:
        return True
    else:
        return False
