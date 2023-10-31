import traci
import os
import configparser
from GLOSA_gym.environment import SumoGlosaEnv
from GLOSA_gym.glosa import GLOSA_agent
from evaluate import evaluate
from stable_baselines3.common.monitor import Monitor
import wandb
import time
import datetime
import shutil
from utils.helpers import create_zip, check_display, SaveOnBestTrainingRewardCallback, algorithm_classes

config = configparser.ConfigParser()
config.read("config.ini")
if not check_display(): #check if display is available
    config.set('Simulation', 'gui', 'False')
    with open('config.ini', 'w') as f:
        config.write(f)
config.read('config.ini')


if __name__ == '__main__':
    # declare the experiment name
    experiment_name = f"{config.get('GLOSA_general', 'glosa_agent')}" \
                      f"{config.get('RL-Training', 'rl_agent') if config.get('GLOSA_general', 'glosa_agent') == 'rl' else ''}" \
                      f"_{config.get('Classic-Configs', 'algo_queue') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
                      f"_step{config.get('GLOSA_general', 'steps') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
                      f"_del{config.get('Classic-Configs', 'delay') if config.get('GLOSA_general', 'glosa_agent') == 'classic' else ''}" \
                      f"_{datetime.datetime.fromtimestamp(int(time.time())).strftime('%m-%d-%H-%M')}"
    
    assert config.get('GLOSA_general', 'glosa_agent') in ['classic', 'rl'], "Unknown GLOSA agent"

    # store the config values for wandb
    wandb_config = {k: v for section in config._sections.values() if isinstance(section, dict) for k, v in section.items()}

    # initialize wandb for experiment tracking
    wandb.init(
        name=experiment_name,
        project=config.get('wandb', 'project'),
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        mode = config.get('wandb', 'mode')
    )

    # create the folder structure for the experiment
    os.makedirs(os.path.join('runs', experiment_name, 'pre_eval'))
    sumo_path = os.path.join('runs', experiment_name, 'sumo')
    os.makedirs(sumo_path)
    for filename in os.listdir('sumo_sim'):
        if not filename.startswith('.'):
            src_file = os.path.join('sumo_sim', filename)
            dst_file = os.path.join(sumo_path, filename)
            shutil.copy(src_file, dst_file)

    # create baseline evaluation without GLOSA and save the results
    reward_sum = evaluate(config, os.path.join('runs', experiment_name, 'pre_eval'), sumo_path, load=False)
    create_zip(os.path.join('runs', experiment_name, 'pre_eval'), os.path.join('runs', experiment_name, 'pre_eval.zip'))

    # evaluate the current classic GLOSA setting of train and evaluate the RL setting
    if config.get('GLOSA_general', 'glosa_agent') == 'classic':
        env = SumoGlosaEnv(config, sumo_path)
        agent = GLOSA_agent()
        try:
            traci.close()
        except:
            pass
        os.makedirs(os.path.join('runs', experiment_name, 'eval'))
        evaluate(config, os.path.join('runs', experiment_name, 'eval'), sumo_path)
        create_zip(os.path.join('runs', experiment_name, 'eval'),
                   os.path.join('runs', experiment_name, 'eval.zip'))

    elif config.get('GLOSA_general', 'glosa_agent') == 'rl':

        # initialize the environment and set up monitor for the training
        env = SumoGlosaEnv(config, sumo_path)
        env = Monitor(env, os.path.join('runs', experiment_name))

        # initialize the RL agent with the defined algorithm
        if config.get('RL-Training', 'rl_agent') in algorithm_classes:
            #policy_kwargs = dict(net_arch=dict(pi=[64,256,256,64], qf=[64,256,256,64]))
            agent = algorithm_classes[config.get('RL-Training', 'rl_agent')]("MlpPolicy", env, verbose=1,
                                                                             tensorboard_log=f"runs/{experiment_name}")
            print(agent.policy)
        else:
            raise ValueError("Unknown algorithm: {}".format(config.get('GLOSA_general', 'glosa_agent')))

        # create callback to save the best model during training
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=os.path.join('runs', experiment_name))
        if os.path.exists(os.path.join('runs', experiment_name, 'best_model')):
            shutil.rmtree((os.path.join('runs', experiment_name, 'best_model')))

        # train the agent and save the final model
        agent.learn(total_timesteps=config.getint('RL-Training', 'num_steps'), callback=callback)
        agent.save(os.path.join('runs', experiment_name, 'trained_agent'))
        try:
            traci.close()
        except:
            pass
        if os.path.exists(os.path.join('runs', experiment_name, 'best_model')):
            shutil.rmtree(os.path.join('runs', experiment_name, 'best_model'))

        # load the best model and evaluate it. Save the results
        os.makedirs(os.path.join('runs', experiment_name, 'eval'))
        reward_sum = evaluate(config, os.path.join('runs', experiment_name, 'eval'), sumo_path)
        create_zip(os.path.join('runs', experiment_name, 'eval'),
                   os.path.join('runs', experiment_name, 'eval.zip'))

    else:
        raise ValueError("Unknown GLOSA agent: {}".format(config.get('GLOSA_general', 'glosa_agent')))

    # store the results in wandb
    wandb.save(os.path.join('runs', experiment_name, 'monitor.csv'))
    wandb.save(os.path.join('runs', experiment_name, 'trained_agent.zip'))
    wandb.save(os.path.join('runs', experiment_name, 'pre_eval.zip'))
    wandb.save(os.path.join('runs', experiment_name, 'eval.zip'))
    wandb.save(os.path.join('runs', experiment_name, 'best_model.zip'))
    wandb.save(os.path.join('config.ini'))
