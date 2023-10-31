try:
    from GLOSA_gym.settings import configure_sumo
    from GLOSA_gym.glosa import get_tl_info, get_preceding_vehciles
except:
    from settings import configure_sumo
    from glosa import get_tl_info, get_preceding_vehciles
import os
import configparser
import wandb
import numpy as np
import traci
from collections import deque
from typing import List, Tuple

class VehicleObserver:
    """
    Class to get the state and rewards from the environment/ego vehicle
    """
    def __init__(self, config: configparser.ConfigParser, vehcile_id: str = 'ego'):
        self.config = config
        self.vehicle_id = vehcile_id
        self.sumo_net = os.path.join('sumo_sim', 'config_a9-thi.sumocfg')
        self.rewards = list() # stores the reward values in one environment step
        self.states = list() # stores the state values in one environment step

        self.phase_on_arrival = None
        self.current_speed = None
        self.time_to_tl = None
        self.up_phases_index = None
        self.tl_index = None

        self.state_distance= self.config.getboolean('RL-State', 'distance')
        self.state_phase_arrival = self.config.getboolean('RL-State', 'phase_arrival')
        self.state_next_green_red_switch = self.config.getboolean('RL-State', 'next_green_red_switch')
        self.state_cars_on_lane = self.config.getboolean('RL-State', 'cars_on_lane')
        self.state_speed = self.config.getboolean('RL-State', 'speed')

        self.state_depth = self.config.getint('RL-State', 'depth')

        self.old_action = None
        self.current_action = 0

        self.reward_waiting = list()
        self.reward_keep_speed = list()
        self.keep_action = list()

        self.next_tls = None
        self.update_in_tls = True
        self.action_deque = deque([1 for _ in range(5)], maxlen=5)
    def return_reward(self) -> float:
        """
        aggregation of the rewards during after one environment step is completed
        """
        #print(f'return_rewards:{self.rewards}')
        reward = sum(self.rewards)
        self.rewards = list()
        return reward

    def log_episode_multimodal(self):
        waiting = sum(self.reward_waiting)
        wandb.log({'waiting_reward': waiting})
        keep_speed = sum(self.reward_keep_speed)
        wandb.log({'keep_speed_reward': keep_speed})
        keep_action = sum(self.keep_action)
        wandb.log({'keep_action_reward': keep_action})
        self.reward_keep_speed = list()
        self.reward_waiting = list()
        self.keep_action = list()

    def return_state(self) -> List[float]:
        """
        aggregation of the rewards during after one environment step is completed
        """
        state = self.states[-1]
        return state

    def get_reward(self) -> float:
        """
        get the current reward of the timestep
        """
        reward = 0
        if True:
            if self.config.getboolean('RL-Reward', 'waiting'):
                if traci.vehicle.getSpeed(self.vehicle_id) < 3:
                    reward += - self.config.getfloat('RL-Reward', 'waiting_factor')
                    self.reward_waiting.append(- self.config.getfloat('RL-Reward', 'waiting_factor'))
            if self.config.getboolean('RL-Reward', 'keep_speed'):
                if traci.vehicle.getSpeed(self.vehicle_id) > -100:
                    reward += -(14 - traci.vehicle.getSpeed(self.vehicle_id)) * self.config.getfloat('RL-Reward', 'keep_speed_factor')
                    self.reward_keep_speed.append(-(14 - traci.vehicle.getSpeed(self.vehicle_id)) * self.config.getfloat('RL-Reward', 'keep_speed_factor'))
                else:
                    print('error in last')
            if self.config.getboolean('RL-Reward', 'co2'):
                reward += -traci.vehicle.getCO2Emission(self.vehicle_id) * self.config.getfloat('RL-Reward', 'co2_factor')
            if self.config.getboolean('RL-Reward', 'keep_action'):
                if len(self.states) > 1:
                    if self.states[-1][-1] == 0:
                        reward += -(abs(self.old_action - self.current_action))* self.config.getfloat('RL-Reward', 'keep_action_factor')
                        self.keep_action.append(-(abs(self.old_action - self.current_action))* self.config.getfloat('RL-Reward', 'keep_action_factor'))
                    else:
                        print('----------no old action reward')
        return float(reward)

    def get_state(self) -> List[float]:
        """
        get the current state of the timestep
        """
        state = list()
        if self.config['GLOSA_general']['glosa_agent'] == 'rl':
            for d in range(self.state_depth):
                if len(traci.vehicle.getNextTLS('ego')) -1 < d:
                    if self.state_distance:
                        state.append(-1)
                    if self.state_phase_arrival:
                        state.append(-1)
                    if self.state_next_green_red_switch:
                        state.append(-1)
                    if self.state_cars_on_lane:
                        state.append(-1)
                    if self.state_speed:
                        state.append(-1)
                    for _ in range(5):
                        state.append(-1)
                    #state.append(-1)
                    state.append(-1)
                else:
                    if self.state_distance:
                        state.append(self.get_distance(d))
                    if self.state_phase_arrival:
                        state.append(self.get_phase_on_arrival(d))
                    if self.state_next_green_red_switch:
                        state.append(self.get_next_green_red_switch())
                    if self.state_cars_on_lane:
                        state.append(len(get_preceding_vehciles(d,'ego')))
                    if self.state_speed:
                        state.append(traci.vehicle.getSpeed('ego'))
                    for i in self.action_deque:
                        state.append(float(i))
                    if self.update_in_tls:
                        state.append(1)
                    else:
                        state.append(0)
            if self.config.getboolean('RL-State', 'leader_distance'):
                leader = traci.vehicle.getLeader(self.vehicle_id, 500)
                if leader is None:
                    state.append(-1)
                elif leader[1] < -100:
                    print('skipping leader information')
                    state.append(-1)
                else:
                    state.append(leader[1])
            if self.config.getboolean('RL-State', 'leader_speed'):
                leader = traci.vehicle.getLeader(self.vehicle_id, 500)
                if leader is None:
                    state.append(-1)
                else:
                    state.append(traci.vehicle.getSpeed(leader[0]))


        elif self.config['GLOSA_general']['glosa_agent'] == 'classic':
            up_phase_index, pre_phase_index = self.get_phases_index()
            state.append(up_phase_index)
            state.append(pre_phase_index)
        return state

    def get_state_dim(self) -> int:
        '''
        start the simulation to get the state dim
        '''
        sumo_cmd = configure_sumo(False, os.path.join('sumo_sim'), 'config_a9-thi.sumocfg', 39600)
        traci.start(sumo_cmd)
        while True:
            traci.simulation.step()
            ids = traci.vehicle.getIDList()
            if 'ego' in ids:
                #traci.gui.trackVehicle('View #0', 'ego')
                #traci.gui.setZoom('View #0', 1000)
                traci.simulation.step()
                break
        state = self.get_state()
        traci.close()
        print('state dim: ', len(state) )
        return len(state)

    def get_distance(self, depth) -> float:
        tl_distance = round(traci.vehicle.getNextTLS('ego')[depth][2], 2)
        self.tl_distance = tl_distance
        return tl_distance
    def get_phase_on_arrival(self, depth) -> int:
        up_phases_index, pre_phases_index, self.tl_index = get_tl_info(next = depth)
        self.up_phases_index = up_phases_index = [x.lower() for x in up_phases_index]
        tl_distance = self.tl_distance
        current_state = traci.vehicle.getNextTLS('ego')[depth][3].lower()
        # Identifying arrival time of the ego vehicle based on its current speed
        current_speed = round(traci.vehicle.getSpeed('ego'), 2)
        time_to_tl = int(tl_distance/current_speed)
        phase_on_arrival = up_phases_index[time_to_tl]
        self.phase_on_arrival = phase_on_arrival
        self.current_speed = current_speed
        self.time_to_tl = time_to_tl
        if phase_on_arrival == "g":
            return 1
        else:
            return 0

    def get_next_green_red_switch(self) -> int:
        if self.phase_on_arrival == "g":
            for i in range(self.time_to_tl, len(self.up_phases_index)):
                if self.up_phases_index[i] != 'g':
                    return i - self.time_to_tl
        else:
            for i in range(self.time_to_tl, len(self.up_phases_index)):
                if self.up_phases_index[i] == 'g':
                    return i - self.time_to_tl
        raise ValueError("no next relevant phase found, "
                         "consider choosing a higher value"
                         " of up_phases in get_tl_info")

    def get_blocking_cars(self) -> int:
        next_tls = traci.vehicle.getNextTLS('ego')[0][0]
        return traci.trafficlight.getBlockingVehicles(next_tls, self.tl_index)

    def get_phases_index(self) -> Tuple[List[str], List[str]]:
        up_phases_index, pre_phases_index, self.tl_index = get_tl_info()
        return up_phases_index, pre_phases_index

    def update_tls(self) -> None:
        next_tls = traci.vehicle.getNextTLS('ego')[0][0]
        if next_tls != self.next_tls:
            self.update_in_tls = True
        else:
            self.update_in_tls = False
        self.next_tls = traci.vehicle.getNextTLS('ego')[0][0]

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    observer = VehicleObserver(config)
    num_states = observer.get_state_dim()
