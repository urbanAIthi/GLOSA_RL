import pandas as pd
import numpy as np
import wandb
import shutil
import traci
import os
import plotly.graph_objs as go
import cv2
import configparser
from typing import Tuple

from typing import List
from GLOSA_gym.glosa import GLOSA_agent
from utils.helpers import algorithm_classes, create_zip
from GLOSA_gym.environment import SumoGlosaEnv
from GLOSA_gym.glosa import get_tl_info


class Evaluator:
    def __init__(self, path: str, config: configparser.ConfigParser, gui: bool = False):
        self.path = path
        if gui:
            self.gui = True
        else:
            self.gui = config.getboolean('Simulation', 'gui')

        self.speeds = {}
        self.acceleration = {}
        self.distance = {}
        self.co2 = {}
        self.fuel = {}
        self.energy = {}

        self.states = {}
        self.actions = {}
        self.dones = {}
        self.rewards = {}

        self.tl_info = {}

    def get_info(self):
        '''
        Upate the information from the simulation
        '''
        time = traci.simulation.getTime()
        self.speeds[time] = traci.vehicle.getSpeed('ego')
        self.acceleration[time] = traci.vehicle.getAcceleration('ego')
        self.distance[time] = traci.vehicle.getDistance('ego')
        self.co2[time] = traci.vehicle.getCO2Emission('ego')
        self.fuel[time] = traci.vehicle.getFuelConsumption('ego')
        self.energy[time] = traci.vehicle.getElectricityConsumption('ego')

        if self.gui:
            self._save_screenshot(time)

        if len(self.tl_info) == 0:
            upcoming_tls = traci.vehicle.getNextTLS('ego')
            for n, tl in enumerate(upcoming_tls):
                tl_phases, _, _ = get_tl_info(up_duration=500, prev_duration=0, next=n)
                self.tl_info[tl[0]] = {'phases': tl_phases}
        else:
            upcoming_tls_list = [traci.vehicle.getNextTLS('ego')[i][0] for i in
                                 range(len(traci.vehicle.getNextTLS('ego')))]
            for tl in self.tl_info.keys():
                if tl not in upcoming_tls_list:
                    if len(self.tl_info[tl]) < 2:
                        self.tl_info[tl]['distance'] = traci.vehicle.getDistance('ego')

    def add_rl_info(self, action: float, obs: List[float], reward: float, dones: bool):
        '''
        Add the information from the RL agent
        '''
        try:
            time = traci.simulation.getTime()
            self.time = time
        except:
            time = self.time + 1
        self.states[time] = obs
        try:
            self.actions[time] = action.item()
        except:
            self.actions[time] = action
        self.rewards[time] = reward
        self.dones[time] = dones

    def evaluate_infos(self) -> Tuple[float, float, float, float, float, float]:
        '''
        Summarize, evaluate and create plots for the current scenario
        '''
        self._create_tlchart(self.tl_info)
        self._create_linechart(self.speeds, 'time', 'speed', 'speed_over_time')
        self._create_linechart(self.actions, 'time', 'action', 'action_over_time')
        self._create_linechart(self.distance, 'time', 'distance', 'distance_over_time')
        self._create_linechart(self.fuel, 'time', 'fuel', 'fuel_over_time')
        self._create_linechart(self.acceleration, 'time', 'acceleration', 'acceleration_over_time')
        self._create_linechart(self.co2, 'time', 'co2', 'co2_over_time')
        self._create_linechart(self.energy, 'time', 'energy', 'energy_over_time')
        summary_dict = {'waiting_time': np.sum(np.array(list(self.speeds.values())) < 0.5),
                        'cum_co2': np.sum(list(self.co2.values())),
                        'time_on_site': list(self.speeds.keys())[-1] - list(self.speeds.keys())[0]}
        # self._save_dict_to_csv(summary_dict, os.path.join(self.path, 'summary.csv'))
        eval_metrics = {}
        for t in self.speeds:
            eval_metrics[t] = {'speed': self.speeds[t],
                               'acceleration': self.acceleration[t],
                               'fuel': self.fuel[t],
                               'distance': self.distance[t],
                               'co2': self.co2[t]}

        df_eval_metrics = pd.DataFrame.from_dict(eval_metrics).T
        df_eval_metrics.to_csv(os.path.join(self.path, 'eval_metrics.csv'))
        eval_rl = {}
        for t in self.states:
            eval_rl[t] = {'states': self.states[t],
                          'rewards': self.rewards[t],
                          'dones': self.dones[t],
                          'actions': self.actions[t]}
        df_eval_rl = pd.DataFrame.from_dict(eval_metrics).T
        df_eval_rl.to_csv(os.path.join(self.path, 'eval_rl.csv'))

        if self.gui:
            self._save_video()
        time_on_site = list(self.speeds.keys())[-1] - list(self.speeds.keys())[0]
        co2_emission = np.sum(list(self.co2.values()))
        fuel_consumption = np.sum(list(self.fuel.values()))
        waiting_time = np.sum(np.array(list(self.speeds.values())) < 0.5)
        reward_sum = np.sum(list(self.rewards.values()))
        energy_consumption = np.sum(list(self.energy.values()))
        return reward_sum, time_on_site, co2_emission, fuel_consumption, waiting_time, energy_consumption

    def _create_linechart(self, d, x_label, y_label, title):
        # Create a trace for the line chart
        trace = go.Scatter(
            x=list(d.keys()),
            y=list(d.values()),
            mode='lines'
        )

        # Create a layout for the chart
        layout = go.Layout(title='Speeds over Time',
                           xaxis=dict(title=x_label),
                           yaxis=dict(title=y_label)
                           )

        # Create a Figure object and add the trace and layout
        fig = go.Figure(data=[trace], layout=layout)
        fig.write_html(os.path.join(self.path, f'{title}.html'))

    def _save_screenshot(self, time):
        os.makedirs(os.path.join(self.path, 'screenshots'), exist_ok=True)
        traci.gui.trackVehicle('View #0', 'ego')
        traci.gui.setZoom('View #0', 1000)
        traci.gui.screenshot('View #0', os.path.join(self.path, 'screenshots', f'{int(time)}.png'))

    def _create_tlchart(self, tl_info):
        layout = go.Layout(
            xaxis=dict(title='time(s)'),
            yaxis=dict(title='distance(m)')
        )
        fig = go.Figure(layout=layout)
        for tl in tl_info:
            xs = np.linspace(0, len(tl_info[tl]['phases']), len(tl_info[tl]['phases']))
            ys = tl_info[tl]['distance']
            df = pd.DataFrame({'x': xs, 'y': ys, 'color': tl_info[tl]['phases']})

            fig.add_scattergl(x=xs, y=df.y.where(df.color == 'r'), line={'color': 'red', 'width': 3}, showlegend=False)
            fig.add_scattergl(x=xs, y=df.y.where(df.color == 'Y'), line={'color': 'yellow', 'width': 3},
                              showlegend=False)
            fig.add_scattergl(x=xs, y=df.y.where(df.color == 'G'), line={'color': 'green', 'width': 3},
                              showlegend=False)
            fig.add_scattergl(x=xs, y=df.y.where(df.color == 'g'), line={'color': 'green', 'width': 3},
                              showlegend=False)
        start_value = list(self.distance.keys())[0]
        hover_dict = {k - start_value: v * 3.6 for k, v in self.speeds.items()}
        fig.add_scatter(x=[i - start_value for i in list(self.distance.keys())], y=list(self.distance.values()),
                        mode='lines', hovertext=list(hover_dict.values()), showlegend=False,
                        line={'color': 'black', 'width': 3})
        fig.write_html(os.path.join(self.path, 'tl_info.html'))
        self._save_dict_to_csv(tl_info, os.path.join(self.path, 'tlinfo.csv'))

    def _save_video(self):
        image_folder = os.path.join(self.path, 'screenshots')  # path to the folder containing the images
        video_name = os.path.join(self.path, 'replay.avi')  # name of the output video file

        images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
        if len(images) > 0:
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter(video_name, 0, 30, (width, height))
            for image in images:
                video.write(cv2.imread(os.path.join(image_folder, image)))

            cv2.destroyAllWindows()
            video.release()

    def _save_dict_to_csv(self, d, path):
        pd.DataFrame.from_dict(d).to_csv(path)


def evaluate(config: configparser.ConfigParser, path: str, sumo_path: str, load: bool = True,
             gui: bool = False) -> float:
    '''
    Function to evaluate the performance of the GLOSA agent (either classic or RL) also in comparison to no GLOSA
    :param config: ConfigParser object containing the configuration of the evaluation
    :param path: Path to the dir of the run
    :param sumo_path: Path to the sumo executable
    :param load: Boolean indicating whether the agent should be loaded from a file or not
    :param gui: Boolean indicating whether the simulation should be run in gui mode or not
    :return: The average reward of the agent for the different test scenarios
    '''

    # Initialize the storage lists for the different metrics to keep track of several test scenarios
    reward_sums = list()
    time_on_sites = list()
    co2_emissions = list()
    fuel_consumptions = list()
    waiting_times = list()
    energy_consumptions = list()

    # Evaluate the performance for the different test scenarios
    org_path = path
    for ad in eval(config.get('GLOSA_general', 'evaluations')):
        path = os.path.join(org_path, str(ad))
        os.mkdir(path)

        # Initialize the evaluator, environment
        evaluator = Evaluator(path, config, gui=gui)
        env = SumoGlosaEnv(config, sumo_path, evaluate=True, evaluator=evaluator, gui=gui, ad=ad)

        # Initialize the classic or the RL agent
        if config.get('GLOSA_general', 'glosa_agent') == 'classic':
            agent = GLOSA_agent()
        elif config.get('GLOSA_general', 'glosa_agent') == 'rl':
            if load:
                # To test trained agent
                print(
                    f'file trained_agent exists: {os.path.exists(os.path.join(os.path.abspath(os.path.join(org_path, os.pardir)), "best_model.zip"))}')
                agent = algorithm_classes[config.get('RL-Training', 'rl_agent')].load(
                    os.path.join(os.path.abspath(os.path.join(org_path, os.pardir)), 'best_model'), env=env)
            else:
                raise ValueError('Train agent before testing')

        # Run the evaluation
        obs = env.reset()
        while True:

            # Set the action to 1 to test the normal baseline (this will allow the vehicle to drive with full speed)
            if not load:
                action = np.array(1, dtype=np.float32)  # To test normal baseline
                pre_eval = True

            # Predict the action of the GLOSA agent
            else:
                action, _states = agent.predict(obs)
                pre_eval = False
            obs, rewards, dones, info = env.step(action, pre_eval=pre_eval)
            evaluator.add_rl_info(action, obs, rewards, dones)
            if dones:
                break

        # Get the results of the current test scenario and store them in the storage lists
        reward_sum, time_on_site, co2_emission, fuel_consumption, waiting_time, energy_consumption = evaluator.evaluate_infos()
        reward_sums.append(reward_sum)
        time_on_sites.append(time_on_site)
        co2_emissions.append(co2_emission)
        fuel_consumptions.append(fuel_consumption)
        waiting_times.append(waiting_time)
        energy_consumptions.append(energy_consumption)

    # Log the mean results to wandb
    wandb.log({"reward_sum": np.mean(reward_sum), "time_on_site": np.mean(time_on_sites),
               "co2_emission": np.mean(co2_emissions), "fuel_consumption": np.mean(fuel_consumptions),
               "waiting_time": np.mean(waiting_times), "energy_consumption": np.mean(energy_consumptions)})
    return np.mean(reward_sum).item()


if __name__ == "__main__":
    wandb.init(project="glosa_anna", name="iaa_eval")
    path = os.path.join('evals', 'iaa_rl_glosa')
    sumo_path = 'sumo_sim'
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'config.ini'))
    if os.path.exists(os.path.join(path, 'pre_eval')):
        shutil.rmtree(os.path.join(path, 'pre_eval'))
    os.mkdir(os.path.join(path, 'pre_eval'))
    reward_sum = evaluate(config, os.path.join(path, 'pre_eval'), sumo_path, load=False, gui=True)
    create_zip(os.path.join(path, 'pre_eval'),
               os.path.join(path, 'pre_eval.zip'))
    if os.path.exists(os.path.join(path, 'eval')):
        shutil.rmtree(os.path.join(path, 'eval'))
    os.mkdir(os.path.join(path, 'eval'))
    reward_sum = evaluate(config, os.path.join(path, 'eval'), sumo_path, gui=True)
    create_zip(os.path.join(path, 'eval'),
               os.path.join(path, 'eval.zip'))
    wandb.save(os.path.join(path, 'eval.zip'))
    wandb.save(os.path.join(path, 'pre_eval.zip'))
