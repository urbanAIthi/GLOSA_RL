import traci
import os
import configparser
from GLOSA_gym.environment import SumoGlosaEnv
from GLOSA_gym.glosa import GLOSA_agent
from evaluate import evaluate
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback
import time
import datetime
from utils.helpers import create_zip, check_display
import shutil
import yaml

config = configparser.ConfigParser()
config.read("config.ini")
if not check_display():  # check if display is available
    config.set('Simulation', 'gui', 'False')
    with open('../config.ini', 'w') as f:
        config.write(f)
config.read('config.ini')


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = SumoGlosaEnv()  # gym.make(env_id)
        return env

    return _init


def train():
    with wandb.init(
            project="glosa_sweep",
            #config=sweep_config,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
            mode="online"
    ):
        wandb_c = wandb.config
        #General Settings
        if 'agent_type' in wandb_c:
            config.set('GLOSA_general', 'glosa_agent', str(wandb_c.agent_type))
        if 'steps' in wandb_c:
            config.set('GLOSA_general', 'steps', str(wandb_c.steps))

        #Classic Settings
        if 'algo_queue' in wandb_c:
            config.set('Classic-Configs', 'algo_queue', str(wandb_c.algo_queue))
        if 'algo_singlemulti' in wandb_c:
            config.set('Classic-Configs', 'algo_singlemulti', str(wandb_c.algo_singlemulti))
        if 'depth' in wandb_c:
            config.set('Classic-Configs', 'depth', str(wandb_c.depth))
        if 'delay' in wandb_c:
            config.set('Classic-Configs', 'delay', str(wandb_c.delay))


        #RL Settings
        if 'reward_waiting_factor' in wandb_c:
            config.set('RL-Reward', 'waiting_factor', str(wandb_c.reward_waiting_factor))
        if 'reward_keep_speed_factor' in wandb_c:
            config.set('RL-Reward', 'keep_speed_factor', str(wandb_c.reward_keep_speed_factor))
        if 'reward_co2_factor' in wandb_c:
            config.set('RL-Reward', 'co2_factor', str(wandb_c.reward_co2_factor))
        if 'reward_traveltime_factor' in wandb_c:
            config.set('RL-Reward', 'traveltime_factor', str(wandb_c.reward_traveltime_factor))
        if 'reward_keep_action_factor' in wandb_c:
            config.set('RL-Reward', 'keep_action_factor', str(wandb_c.reward_keep_action_factor))
        if 'reward_keep_action_factor' in wandb_c:
            config.set('RL-Reward', 'keep_action_factor', str(wandb_c.reward_keep_action_factor))
        if 'state_depth' in wandb_c:
            config.set('RL-State', 'depth', str(wandb_c.state_depth))
        if 'state_phase_arrival' in wandb_c:
            config.set('RL-State', 'phase_arrival', str(wandb_c.state_phase_arrival))
        if 'state_next_green_red_switch' in wandb_c:
            config.set('RL-State', 'next_green_red_switch', str(wandb_c.state_next_green_red_switch))
        if 'state_distance' in wandb_c:
            config.set('RL-State', 'distance', str(wandb_c.state_distance))
        if 'state_leader' in wandb_c:
            config.set('RL-State', 'leader', str(wandb_c.state_leader))
        if 'state_leader_speed' in wandb_c:
            config.set('RL-State', 'leader_speed', str(wandb_c.state_leader_speed))
        if 'state_cars_on_lane' in wandb_c:
            config.set('RL-State', 'cars_on_lane', str(wandb_c.state_cars_on_lane))

        #Save config
        with open('../config.ini', 'w') as f:
            config.write(f)
        wandb.save('../config.ini')
        config.read('config.ini')
        # config_dicts = dict(config._sections)
        # config_dicts = [v for v in config_dicts.values() if isinstance(v, dict)]
        # wandb_config = {}
        # for config_dict in config_dicts:
        #    wandb_config.update(config_dict)

        #f"_step{config.get('GLOSA_general', 'steps') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
        # f"_dep{config.get('Classic-Configs', 'depth') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
        # f"_del{config.get('Classic-Configs', 'delay') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \

        experiment_name = f"{config.get('GLOSA_general', 'glosa_agent')}" \
                          f"{config.get('RL-Training', 'rl_agent') if config.get('GLOSA_general', 'glosa_agent') == 'rl' else ''}" \
                          f"_{config.get('Classic-Configs', 'algo_queue') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
                          f"_{config.get('Classic-Configs', 'algo_singlemulti') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
                          f"_del{config.get('Classic-Configs', 'delay') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
                          f"_{datetime.datetime.fromtimestamp(int(time.time())).strftime('%m-%d-%H-%M-%S')}"
        os.makedirs(os.path.join('../runs', experiment_name, 'pre_eval'))
        sumo_path = os.path.join('../runs', experiment_name, 'sumo')
        os.makedirs(sumo_path)
        for filename in os.listdir('../sumo_sim'):
            src_file = os.path.join('../sumo_sim', filename)
            dst_file = os.path.join(sumo_path, filename)
            shutil.copy(src_file, dst_file)
        reward_sum = evaluate(config, os.path.join('../runs', experiment_name, 'pre_eval'), sumo_path,
                              load=False)  # Create Baseline Evaluation
        create_zip(os.path.join('../runs', experiment_name, 'pre_eval'),
                   os.path.join('../runs', experiment_name, 'pre_eval.zip'))
        if config.get('GLOSA_general', 'glosa_agent') == 'classic':
            env = SumoGlosaEnv(config, sumo_path)
            agent = GLOSA_agent()
            try:
                traci.close()
            except:
                pass
            os.makedirs(os.path.join('../runs', experiment_name, 'eval'))
            evaluate(config, os.path.join('../runs', experiment_name, 'eval'), sumo_path)
            create_zip(os.path.join('../runs', experiment_name, 'eval'),
                       os.path.join('../runs', experiment_name, 'eval.zip'))

        elif config.get('GLOSA_general', 'glosa_agent') == 'rl':
            if config.getint('RL-Training', 'num_parallel') == 1:
                env = SumoGlosaEnv(config, sumo_path)
                env = Monitor(env, os.path.join('../runs', experiment_name))
            else:
                env = SubprocVecEnv([make_env(i) for i in range(config.getint('RL-Training', 'num_parallel'))])
                env = VecMonitor(env, os.path.join('../runs', experiment_name))
            algorithm_classes = {
                "PPO": PPO,
                "SAC": SAC,
                "A2C": A2C
            }
            if config.get('RL-Training', 'rl_agent') in algorithm_classes:
                agent = algorithm_classes[config.get('RL-Training', 'rl_agent')]("MlpPolicy", env, verbose=1,
                                                                                 tensorboard_log=f"runs/{experiment_name}")
            else:
                raise ValueError("Unknown algorithm: {}".format(config.get('GLOSA_general', 'glosa_agent')))
            agent.learn(total_timesteps=config.getint('RL-Training', 'num_steps'), callback=WandbCallback(
                gradient_save_freq=100,
                model_save_freq=100,
                model_save_path=f"runs/{experiment_name}"
            ))
            agent.save(os.path.join('../runs', experiment_name, 'trained_agent'))
            try:
                traci.close()
            except:
                pass
            os.makedirs(os.path.join('../runs', experiment_name, 'eval'))
            reward_sum = evaluate(config, os.path.join('../runs', experiment_name, 'eval'), sumo_path)
            create_zip(os.path.join('../runs', experiment_name, 'eval'),
                       os.path.join('../runs', experiment_name, 'eval.zip'))

        else:
            raise NotImplementedError
        wandb.save(os.path.join('../runs', experiment_name, 'monitor.csv'))
        wandb.save(os.path.join('../runs', experiment_name, 'trained_agent.zip'))
        wandb.save(os.path.join('../runs', experiment_name, 'pre_eval.zip'))
        wandb.save(os.path.join('../runs', experiment_name, 'eval.zip'))
        wandb.save(os.path.join('../config.ini'))


if __name__ == '__main__':
    yaml_path = os.path.join('../yaml_sweeps', 'rl_full.yaml')
    # Load the sweep configuration from the YAML file
    with open(yaml_path, 'r') as file:
        sweep_config = yaml.safe_load(file)
    #sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="glosa_sweep")
    wandb.agent(sweep_id, function=train)
