import datetime
import os
import sys
from enum import Enum

import pandas as pd


def create_intersection_path(models_path_name):
    """
    Create a new intersection model path with an incremental integer, also considering previously created model paths.
    """
    models_path = models_path_name.split('/', 1)[1]
    train_model_path = os.path.join(os.getcwd(), f'models/{models_path}', '')
    os.makedirs(os.path.dirname(train_model_path), exist_ok=True)

    intersection_model_relative_path = f'intersection/{models_path}'
    return create_incremental_path(intersection_model_relative_path)


def create_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    return create_incremental_path(models_path_name)


def create_incremental_path(relative_path):
    path = os.path.join(os.getcwd(), relative_path, '')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    new_version = 1
    dir_content = os.listdir(path)
    if dir_content:
        previous_versions = [int(name.split('_')[-1]) for name in dir_content if not name.startswith('.')]
        new_version = max(previous_versions) + 1

    new_path = os.path.join(path, 'model_' + str(new_version), '')
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    return new_path, new_version


def create_test_path(test_model_path_name):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), test_model_path_name, '')

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')


def add_master_data(path, config, scores, training_time, wait, queue):
    master_df = pd.read_excel(MASTER_DATA_FILE)
    path = path[0:-1]
    name = os.path.split(path)[1]
    master_df = pd.concat([master_df, pd.DataFrame([{
        'run_name': name,
        'datetime': datetime.datetime.now(),
        'agent_type': config['agent_type'],
        'model': config['model'],
        'total_episodes': config['total_episodes'],
        'generation_process': config['agent_type'],
        'num_states': config['num_states'],
        'cars_generated': config['n_cars_generated'],
        'num_actions': config['num_actions'],
        'state_representation': config['state_representation'],
        'action_representation': config['action_definition'],
        'final_reward': scores[-1],
        'training_time': training_time[-1],
        'final_waiting_time': wait,
        'final_length': queue
    }])], ignore_index=True)

    master_df.to_excel(MASTER_DATA_FILE, index=False)


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def opposite(self):
        return Direction((self.value + 2) % 4)

    def relative(self, next_direction):
        relative_directions = [RelativeDirection.LEFT, RelativeDirection.STRAIGHT, RelativeDirection.RIGHT]
        relative_direction_index = ((self.value - next_direction.value) % 4) - 1
        return relative_directions[relative_direction_index]


class RelativeDirection(Enum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Logger:
    def __init__(self):
        self.log_file_path = None

    def set_log_file(self, log_file_path):
        self.log_file_path = log_file_path
        print('Logging output will be saved to:', log_file_path)
        open(log_file_path, 'a').close()

    def log(self, *args, **kwargs):
        print(*args, **kwargs)

        if self.log_file_path is not None:
            with open(self.log_file_path, 'a') as file:
                print(*args, **kwargs, file=file)


logger = Logger()


def import_sumo_tools():
    """
    Import Python modules from the $SUMO_HOME/tools directory.
    """
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
