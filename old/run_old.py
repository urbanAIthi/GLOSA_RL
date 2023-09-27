

def make_env(rank, config, sumo_path, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init(config, sumo_path):
        env = SumoGlosaEnv(config, sumo_path)
        return env

    """
    env = SubprocVecEnv([make_env(i, config, sumo_path) for i in range(config.getint('RL-Training', 'num_parallel'))])
    env = VecMonitor(env, os.path.join('runs', experiment_name))
    """

    return _init