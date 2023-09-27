import wandb
import pprint
def get_sweep_config():
    sweep_config = {
        'method': 'grid'
        }
    metric = {
        'name': 'reward_sum',
        'goal': 'maximize'
        }

    sweep_config['metric'] = metric
    parameters_dict = {
        'delay': {
            'values': [5, 7, 10, 15, 20, 25]},
        'steps': {
            'values': [1, 2, 3, 5, 10, 20]},
        'algo_queue': {
            'values': ["BASIC", "WAITING"]},
        }

            #'state_cars_on_lane': { 'values' : [False, True]},
            #'reward_keep_speed_factor': {'values': [1]},
            #'reward_co2_factor': {'values': [1]},
            #'reward_traveltime_factor': {'values': [1]},
            #'reward_keep_action_factor': {'values': [5, 1, 0.5]},
            #'state_leader': {'values': [True, False]},
            #'state_leader_speed': {'values': [True, False]},
        #}
        # 'n_steps': {
        #     'values': [8, 16, 32, 64, 128, 256]
        #     },
        # 'gamma': {
        #       'values': [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
        #     },
        # 'n_epochs': {
        #     'values': [1, 5, 10, 20]
        # },
        # 'gae_lambda': {
        #     'values': [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
        # },
        # }


    sweep_config['parameters'] = parameters_dict
    return sweep_config

if __name__ == '__main__':
    #wandb.login()
    pprint.pprint(get_sweep_config())
