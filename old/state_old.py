class VehicleManager:
    def __init__(self, traffic_lights: dict):
        self.junction_observers = dict.fromkeys(traffic_lights)
        for junction_name in traffic_lights:
            self.junction_observers[junction_name] = JunctionObserver(
                config['sumocfg_file_name'], junction_name, config['state_representation'],
                'sum', config['reward_definition'], 'sum'
            )

    def receive_states(self) -> None:
        for junction_observer in self.junction_observers.values():
            junction_observer.append_states()

    def receive_rewards(self) -> None:
        for junction_observer in self.junction_observers.values():
            junction_observer.receive_reward()

    def get_state_sizes(self) -> dict:
        state_sizes = dict.fromkeys(self.junction_observers)

        for junction_name, junction_observer in self.junction_observers.items():
            state_sizes[junction_name] = junction_observer.get_state_dimension() + junction_observer.num_programs

        return state_sizes

    def return_states(self) -> dict:
        action_states = dict.fromkeys(self.junction_observers)

        for junction_name, junction_observer in self.junction_observers.items():
            if junction_observer.action == 1:
                aggregated_states = junction_observer.aggregate_states()
                one_hot = np.zeros(junction_observer.num_programs)

                if len(one_hot) > 0:
                    one_hot[junction_observer.program] = 1
                    aggregated_states_one_hot = [*aggregated_states, *one_hot]
                    action_states[junction_name] = aggregated_states_one_hot
                else:
                    aggregated_states.append(traci.trafficlight.getPhase(junction_name))
                    action_states[junction_name] = aggregated_states

                self.junction_observers[junction_name].clear_states()

            elif junction_observer.action == 0:
                action_states[junction_name] = None

        return action_states

    def compute_rewards(self) -> dict:
        action_rewards = dict.fromkeys(self.junction_observers)

        for junction_name, junction_observer in self.junction_observers.items():
            junction_action = junction_observer.action
            if junction_action == 1:
                aggregated_rewards = junction_observer.aggregate_reward()
                action_rewards[junction_name] = aggregated_rewards
                junction_observer.clear_rewards()
            elif junction_action == 0:
                action_rewards[junction_name] = None

        return action_rewards