import traci
import numpy as np
import configparser
from typing import Tuple, List

config = configparser.ConfigParser()
config.read('config.ini')


class GLOSA_agent:
    def __init__(self):

        self.id = 'ego'
        self.algo_Queue = str(config.get('Classic-Configs', 'algo_queue'))
        self.delay = config.getfloat('Classic-Configs', 'delay')
        self.activation_distance = config.getint('Classic-Configs', 'activation_distance')
        self.max_speed = config.getfloat('Simulation', 'max_speed')
        self.min_speed = config.getfloat('Simulation', 'min_speed')
        self.steps = config.getint('GLOSA_general', 'steps')

    def predict(self, obs):
        """
        Superordinate function to obtain the speed advisory
        :param obs: Phase observation
        :return: Speed advisory
        """

        print("obs:", obs)
        if obs[0] == 0:
            return self.max_speed, None
        else:
            up_phases_index = obs[0]
            up_phases_index = [x.lower() for x in up_phases_index]
            #print(self.algo_Queue)
            speed_advisory = self.max_speed

            if self.id in traci.vehicle.getIDList():
                tl_distance_init = round(traci.vehicle.getNextTLS(self.id)[0][2],
                                         2)  # Distance of the ego vehicle to the traffic light

                # **************************************************
                # Distinguishing between algorithms "basic" and "waiting"
                if (self.algo_Queue == "WAITING"):
                    tl_distance = calculate_waitingDistance(tl_distance_init, up_phases_index, self.delay, depth=0)
                else:
                    tl_distance = tl_distance_init

                print("TL distance: ", tl_distance)

                # **************************************************
                # Identifying arrival time of the ego vehicle based on its current speed
                current_speed = round(traci.vehicle.getSpeed(self.id), 2)
                print("Current speed:", current_speed)

                if current_speed >= 1:
                    time_to_tl = int(tl_distance / current_speed)
                    print("TTL:", time_to_tl)

                    # Identifying switch time, arrival time and reaching state
                    current_state, reaching_state, next_switch, relevant_switchtime, red_duration = calculate_relevantSwitchTime(
                        up_phases_index, time_to_tl, depth=0, id="ego")

                    if time_to_tl == next_switch:
                        time_to_tl += 1

                    if (tl_distance <= self.activation_distance) & (time_to_tl <= 1000) & (
                            time_to_tl > 0):  # Speed advisory at specific activation distance
                        print("Current State:", current_state)
                        print("Reaching State:", reaching_state)

                        # **************************************************
                        # Calculating speed advisory
                        speed_advisory = calculate_speedAdvisory(up_phases_index, reaching_state, current_speed,
                                                                 tl_distance, relevant_switchtime, red_duration,
                                                                 self.max_speed, self.min_speed)

            print("Speed Advisory final:", speed_advisory)

            return speed_advisory, None


def get_tl_info(id='ego', up_duration=5000, prev_duration=100, next=0) -> Tuple[List[str], List[str], int]:
    """
    Get the traffic light information of the next traffic light and specifically the index the ego vehicle is approaching
    :param id: id of the ego vehicle
    :param up_duration: number of steps to look ahead
    :param prev_duration: number of steps to look back
    :param next: index of the traffic light link the ego vehicle is approaching
    :return: list of the phases the ego vehicle is approaching, list of the phases the ego vehicle is leaving,
    index of the traffic light the ego vehicle is approaching
    """

    if id in traci.vehicle.getIDList():
        tl = traci.vehicle.getNextTLS('ego')[next][0]
        tl_index = traci.vehicle.getNextTLS('ego')[next][1]
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)
        program = traci.trafficlight.getProgram(tl)
        next_phases = list()
        for l in logic:
            if l.programID == program:
                current_logic = l
                break
        next_switch = traci.trafficlight.getNextSwitch(tl) - traci.simulation.getTime()
        current_phase = traci.trafficlight.getPhase(tl)
        up_phases = []

        for t in range(int(next_switch)):
            up_phases.append(current_logic.phases[current_phase].state)
        to_append = up_duration - next_switch
        num_phases = len(current_logic.phases)
        n_phase = current_phase + 1
        if n_phase >= num_phases:
            n_phase = 0
        while to_append != 0:
            for t in range(int(current_logic.phases[n_phase].minDur)):
                current_logic.phases[n_phase].state
                up_phases.append(current_logic.phases[n_phase].state)
                to_append -= 1
                if to_append == 0:
                    break
            n_phase += 1
            if n_phase >= num_phases:
                n_phase = 0
        up_phases_index = list()
        for phase in up_phases:
            up_phases_index.append(phase[tl_index])
        # previous phases
        pre_phases = []

        for t in range(int(current_logic.phases[current_phase].minDur - int(next_switch))):
            pre_phases.append(current_logic.phases[current_phase].state)
        n_phase = current_phase - 1
        if n_phase < 0:
            n_phase = num_phases - 1
        to_append = prev_duration - (current_logic.phases[current_phase].minDur - int(next_switch))
        if to_append < 0:
            for _ in range(prev_duration):
                pre_phases.append(current_logic.phases[n_phase].state)
        else:
            while to_append != 0:
                for t in range(int(current_logic.phases[n_phase].minDur)):
                    current_logic.phases[n_phase].state
                    pre_phases.append(current_logic.phases[n_phase].state)
                    to_append -= 1
                    if to_append == 0:
                        break
                n_phase -= 1
                if n_phase < 0:
                    n_phase = num_phases - 1
        pre_phases_index = list()
        for phase in pre_phases:
            pre_phases_index.append(phase[tl_index])
        return up_phases_index, pre_phases_index, tl_index


def get_preceding_vehciles(depth, ego='ego'):
    """
    Determining  preceding vehicles at the traffic light with depth ahead
    Preceding vehicles are those that will arrive at the traffic light before the ego vehicle and in the
    same lane as the ego vehicle
    :param depth: Considering the traffic light at depth ahead (depth = 0 indicates next traffic light)
    :param ego: ID of the vehicle whose preceding vehicles are determined
    :return: Preceding vehicles
    """

    tl_distance_init = round(traci.vehicle.getNextTLS(ego)[depth][2], 2)  # Distance of the ego vehicle to the traffic light
    tl_index = traci.vehicle.getNextTLS(ego)[depth][1]
    tl = traci.vehicle.getNextTLS(ego)[depth][0]
    precedingVehicles = []
    vehicleIDlist = traci.vehicle.getIDList()
    # print("vehicleIDlist: ", vehicleIDlist)

    if len(vehicleIDlist) != 0:
        for x in vehicleIDlist:
            try:
                tl_x = traci.vehicle.getNextTLS(x)[depth][0]
                tl_ind_x = traci.vehicle.getNextTLS(x)[depth][1]
                tl_distance_x = round(traci.vehicle.getNextTLS(x)[depth][2], 2)
            except:
                tl_x = None
                tl_ind_x = None
                tl_distance_x = 0

            if (tl_x == tl) & (tl_ind_x == tl_index) & (tl_distance_x <= tl_distance_init) & (x != "ego"):
                precedingVehicles.append((tl_distance_x, x))
    precedingVehicles.sort()

    return precedingVehicles


def calculate_waitingDistance(tl_distance_init, up_phases_index, delay, depth):
    """
    Calculating waiting distance, in other words queue length, based on preceding vehicles at the traffic light
    with depth ahead
    :param tl_distance_init: Distance of the ego vehicle to the next traffic light
    :param up_phases_index: Next phases of the traffic light
    :param delay: Delay factor in order to compensate queue dissipation time
    :param depth: Considering the traffic light at depth ahead (depth = 0 indicates next traffic light)
    :return: Updated distance to the traffic light after subtracting the queue length (in meters)
    """

    state_list = []
    waiting_distance = 0

    precedingVehicles = get_preceding_vehciles(depth=depth, ego='ego')
    number_vehicles = len(precedingVehicles)
    new_precedingVehicles = map(lambda y: y[1], precedingVehicles)
    new_precedingVehicles = list(new_precedingVehicles)
    # print("new_precedingVehicles: ", new_precedingVehicles)
    # print("number_vehicles: ", number_vehicles)

    if number_vehicles > 0:

        for v in new_precedingVehicles:
            tl_distance_v = round(traci.vehicle.getNextTLS(v)[depth][2], 2)
            current_speed_v = round(traci.vehicle.getSpeed(v), 2)

            # Considering only those vehicles that will arrive at red or yellow phase
            if current_speed_v > 0.1:
                time_to_tl_v = int(tl_distance_v / current_speed_v)
                reaching_state_v = up_phases_index[time_to_tl_v - 1]
                if reaching_state_v == "y":
                    state_list.append("r")
                else:
                    state_list.append(reaching_state_v)

            else:
                state_list.append("r")

        if "r" in state_list:
            ind_state = state_list.index("r")
            waiting_distance = 1 + (number_vehicles - ind_state) * (
                        delay + traci.vehicle.getLength("ego") + traci.vehicle.getMinGap("ego"))

    if tl_distance_init > waiting_distance:
        tl_distance = round(tl_distance_init - waiting_distance, 2)
    else:
        tl_distance = 0.1

    return tl_distance


def calculate_relevantSwitchTime(up_phases_index, time_to_tl, depth, id="ego"):
    """
    Calculating relevant switch time from yellow or red to green or reverse, which will be used for calculating
    the speed advisory
    :param up_phases_index: Next phases of the traffic light
    :param time_to_tl: Time needed for the ego vehicle to reach the traffic light
    :param depth: Considering the traffic light at depth ahead (depth = 0 indicates next traffic light)
    :param id: ID of the vehicle for which the relevant switch time is determined
    :return: Current state of the traffic light, state at arrival of the ego vehicle, next switch of the
    ego vehicle (in seconds), relevant switch time for the ego vehicle (in seconds), duration of red phase (in seconds)
    """

    current_state = traci.vehicle.getNextTLS(id)[depth][3].lower()
    reaching_state = up_phases_index[time_to_tl - 1]
    print("Ampel:", traci.vehicle.getNextTLS(id)[depth][0])

    # Identifying next switch time for ego link
    if current_state == "g":
        switch_state = "y"
    elif current_state == "r":
        switch_state = "g"
    elif current_state == "y":
        switch_state = "r"

    next_switch = up_phases_index.index(switch_state)

    # Identifying relevant switchtime for next raffic light
    if reaching_state == "g":
        next_state = "y"
    elif reaching_state == "r":
        next_state = "g"
    elif reaching_state == "y":
        next_state = "r"

    new_list = up_phases_index[time_to_tl:]
    index = new_list.index(next_state)
    relevant_switchtime = index + time_to_tl + 1

    # Identifying Red Phase Duration
    red_list = up_phases_index[next_switch:]
    ind_r1 = red_list.index("r")
    red_list_2 = red_list[ind_r1:]
    ind_r2 = red_list_2.index("g")
    red_list_3 = red_list_2[:ind_r2]
    red_duration = len(red_list_3)

    # print(red_duration)
    # print("Relevant switchtime:", relevant_switchtime)

    return current_state, reaching_state, next_switch, relevant_switchtime, red_duration


def calculate_speedAdvisory(up_phases_index, reaching_state, current_speed, tl_distance, relevant_switchtime,
                            red_duration, max_speed, min_speed):
    """
    Calculating the speed advisory
    :param up_phases_index: Next phases of the traffic light
    :param reaching_state: State at arrival of the ego vehicle
    :param current_speed: Current speed of the ego vehicle
    :param tl_distance: Distance of the ego vehicle to the traffic light (For queue algorithm,
    queue length is already subtracted)
    :param relevant_switchtime: Relevant switch time for the ego vehicle (in seconds), for which
    the speed advisory is calculated
    :param red_duration: Duration of red phase (in seconds)
    :param max_speed: Maximum allowed speed for the road segment
    :param min_speed: Minimum allowed speed for the road segment (half of max speed)
    :return: Speed advisory for the ego vehicle
    """

    # Calculating optimal speed
    advisory_speed = max_speed

    match reaching_state:
        case "g":
            # Check whether higher speeds (near max_speed) are also possible
            if current_speed < max_speed:
                for i in np.arange(current_speed, max_speed, 0.01):
                    state = up_phases_index[int(tl_distance / i) - 1]

                    if state == "g":
                        advisory_speed = round(max(min_speed, i), 2)
            else:
                advisory_speed = max_speed

        case "r":
            # Check whether it is possible to accelerate in order to reach the previous green phase
            x = relevant_switchtime - red_duration - 5
            if x > 0:
                prevgreen_speed = ((2 * tl_distance) / x) - current_speed
                print("prevgreen_speed", prevgreen_speed)
                if (prevgreen_speed <= max_speed) & (prevgreen_speed > 0):
                    for i in np.arange(prevgreen_speed, max_speed, 0.01):
                        state = up_phases_index[int(tl_distance / i) - 1]
                        if state == "g":
                            advisory_speed = round(max(min_speed, i), 2)
                else:
                    # Speed for next green time
                    ttg_speed = round(((2 * tl_distance) / relevant_switchtime) - current_speed, 2)
                    print("ttg_speed:", ttg_speed)
                    if ttg_speed <= max_speed:
                        advisory_speed = max(ttg_speed, min_speed)
                    else:
                        advisory_speed = min_speed
            else:
                ttg_speed = round(((2 * tl_distance) / relevant_switchtime) - current_speed, 2)
                print("ttg_speed:", ttg_speed)
                if ttg_speed <= max_speed:
                    advisory_speed = max(ttg_speed, min_speed)
                else:
                    advisory_speed = min_speed

        case "y":
            # Check whether it is possible to accelerate in order to reach the previous green phase
            try:
                prevgreen_speed = ((2 * tl_distance) / (relevant_switchtime - 5)) - current_speed
                print("prevgreen_speed", prevgreen_speed)
                if (prevgreen_speed <= max_speed) & (prevgreen_speed > 0):
                    for i in np.arange(prevgreen_speed, max_speed, 0.01):
                        state = up_phases_index[int(tl_distance / i) - 1]
                        if state == "g":
                            advisory_speed = round(max(min_speed, i), 2)
                elif prevgreen_speed > max_speed:
                    advisory_speed = min_speed
            except:
                advisory_speed = min_speed

    return advisory_speed
