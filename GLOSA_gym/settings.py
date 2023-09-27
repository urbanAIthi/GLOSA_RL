import os
import sys
from enum import Enum, EnumMeta, auto
from sumolib import checkBinary

import configparser
import argparse


class ActionDefinition(Enum):
    PHASE = auto()
    CYCLE = auto()


class TrainingStrategy(Enum):
    NONSTRATEGIC = auto()
    CONCURRENT = auto()
    CENTRALIZED = auto()


class RewardDefinition(Enum):
    WAITING = auto()
    WAITING_FAST = auto()
    FUEL = auto()


class StateRepresentation(Enum):
    VOLUME_LANE = auto()
    VOLUME_LANE_FAST = auto()
    WAITING_T = auto()
    STAULANGE = auto()


class AgentType(Enum):
    DQN = auto()
    MADQN = auto()
    PPO = auto()
    MAPPO = auto()
    RAINBOW_DQN = auto()
    WOLP = auto()

    def is_multi(self):
        return self == self.MADQN or self == self.MAPPO


_settings = {
        'simulation': {
            'gui': bool,
            'total_episodes': int,
            'max_steps': int,
            'n_cars_generated': int,
            'generation_process': str,
            'green_duration': int,
            'yellow_duration': int,
            'red_duration': int,
            'num_intersections': int,
            'intersection_length': int,
            'cycle_time': int,
            'min_green_duration': int
        },

        'model': {
            # Common agent settings
            'batch_size': int,

            # DQN agents settings
            'hidden_dim': list,
            'target_update': int,
            'learning_rate': float,

            # PPO agents settings
            'critic_dim': list,
            'actor_dim': list,
            'policy_learning_rate': float,
            'value_learning_rate': float,

            # WOLP/DDPG agent settings
            'weight_decay': float,
            'warmup': int,
            'actor_init_w': float,
            'critic_init_w': float,
        },

        'memory': {
            'memory_size_min': int,
            'memory_size_max': int,
        },

        'strategy': {
            'eps_start': float,
            'eps_end': float,
            'eps_decay': float,
            'eps_policy': int,
        },

        'agent': {
            # Common agent settings
            'agent_type': AgentType,
            'model': str,
            'is_train': bool,
            'state_representation': StateRepresentation,
            'action_definition': ActionDefinition,
            'reward_definition': RewardDefinition,
            'num_actions': int,
            'num_states': int,
            'gamma': float,

            # DQN agents settings
            'tau': float,

            # PPO agents settings
            'gae_lambda': float,
            'policy_clip': float,
            'n_epochs': int,
            'learning_interval': int,

            # MAPPO agent settings
            'training_strategy': TrainingStrategy,
            'actor_parameter_sharing': bool,
            'critic_parameter_sharing': bool,

            # Single agent settings
            'fixed_action_space': bool,

            # Multi agent settings
            'single_state_space': bool,
            'local_reward_signal': bool,

            # WOLP/DDPG agent settings
            'ou_theta': float,
            'ou_mu': float,
            'ou_sigma': float,
        },

        'dir': {
            'models_path_name': str,
            'test_model_path_name': str,
            'sumocfg_file_name': str,
        }
    }


def _import_configuration_data():
    """Read the appropriate config file (for training or testing)."config"""
    config_file = os.path.join('GLOSA_gym', 'training_settings_old.ini')
    options = _get_cli_options()
    config_data = _parse_config_file(config_file, options, is_train_config=True)

    if not config_data['is_train']:
        test_config_file = os.path.join(config_data['test_model_path_name'], config_file)
        config_data = _parse_config_file(test_config_file, options, is_train_config=False)
        _overwrite_config_with_options(config_data, options)

    return config_data


def _parse_config_file(config_file, options, is_train_config):
    content = configparser.ConfigParser()
    content.read(config_file)
    config_data = {}

    for category, category_settings in _settings.items():
        for setting, setting_type in category_settings.items():
            if setting_type == bool:
                value = content[category].getboolean(setting)
            elif setting_type == int:
                value = content[category].getint(setting)
            elif setting_type == float:
                value = content[category].getfloat(setting)
            elif setting_type == str:
                value = content[category].get(setting)
            elif setting_type == list:
                value = [int(v) for v in content[category].get(setting).split(',')]
            elif type(setting_type) == EnumMeta:
                value_str = content[category].get(setting)

                try:
                    value = setting_type[value_str.upper()]
                except KeyError:
                    sys.exit(
                        f'Invalid value "{value_str}" for setting "{setting}". '
                        f'Valid values are: {[name.lower() for name in setting_type.__members__]}'
                    )
            else:
                sys.exit(f'Invalid type "{setting_type}" for setting "{setting}"')

            config_data[setting] = value

    _overwrite_config_with_options(config_data, options)

    # Handle the multi-agent and single agent cases
    if config_data['agent_type'].is_multi():
        config_data['single_agent'] = False
        config_data['fixed_action_space'] = False
    else:
        config_data['single_agent'] = True
        config_data['single_state_space'] = False
        config_data['local_reward_signal'] = False

    # Change settings for test configuration
    if not is_train_config:
        config_data['is_train'] = False

    return config_data


def _create_enum_converter(enum_class):
    def convert_to_enum(arg: str):
        return enum_class[arg.upper()]

    return convert_to_enum


def _get_cli_options():
    arg_parser = argparse.ArgumentParser()

    for category, category_settings in _settings.items():
        for setting, setting_type in category_settings.items():
            cli_option = setting.replace('_', '-')

            if setting_type == bool:
                action = 'store_true'
                arg_parser.add_argument(f'--{cli_option}', action=action, default=None, dest=setting)
                arg_parser.add_argument(f'--not-{cli_option}', action='store_false', default=None, dest=setting)
            elif setting_type == list:
                arg_parser.add_argument(f'--{cli_option}', action=_IntListAction, default=None, dest=setting)
            else:
                action = 'store'
                arg_type = _create_enum_converter(setting_type) if type(setting_type) == EnumMeta else setting_type
                arg_parser.add_argument(f'--{cli_option}', action=action, type=arg_type, default=None, dest=setting)

    arg_parser.add_argument('--test', action='store_false', dest='is_train')
    return arg_parser.parse_args()


class _IntListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        value = [int(v) for v in values.split(',')]
        setattr(namespace, self.dest, value)


def _overwrite_config_with_options(config_data, options):
    for option_name, value in options.__dict__.items():
        if value is not None:
            config_data[option_name] = value


class Configuration(dict):
    def __init__(self, config_data):
        super().__init__(config_data)

    def overwrite(self, config_changes):
        for setting, value in config_changes.items():
            self[setting] = value

    def __getattr__(self, item):
        return self[item]

def configure_sumo(gui, model_path, sumocfg_file_name, begin_time, max_steps=3600):
    """
    Configure various parameters of SUMO.
    """
    # Setting the cmd mode or the visual mode
    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    # Setting the cmd command to run sumo at simulation time
    model_path = os.path.join(model_path, sumocfg_file_name)
    sumo_cmd = [
        sumo_binary, "-c", model_path, "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps), "--xml-validation", "never", "--start", "--quit-on-end", "--begin", str(begin_time), "-W"
    ]

    return sumo_cmd

#config = Configuration(_import_configuration_data())
