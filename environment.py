from traffic_light import TrafficLight
from gymnasium import Env, spaces
from xml.etree import ElementTree
import numpy as np
import sumolib
import traci
from traci.exceptions import TraCIException
from traci import tc
import logging
import sys
import os
from pathlib import Path


VEHICLE_LENGTH = 5
MIN_GAP = 1
MAX_SPEED = 13.89  # Vehicles with speed factors greater than 1 may exceed speed limit. We cap them at the speed limit
MIN_SPEED = 0.0
MAX_ACCEL = 2.6
MIN_ACCEL = -4.5  # Deceleration during emergency braking will be as low as -9.0, which will be clipped at -4.5

VEHICLE_DATA_TO_SUBSCRIBE = [
    tc.VAR_SPEED,
    tc.VAR_ACCELERATION,
    tc.VAR_ACCUMULATED_WAITING_TIME,
    tc.VAR_WAITING_TIME,
    tc.VAR_LANEPOSITION,
    tc.VAR_TIMELOSS,
    tc.VAR_CO2EMISSION,
    tc.VAR_COEMISSION,
    tc.VAR_HCEMISSION,
    tc.VAR_NOXEMISSION,
    tc.VAR_PMXEMISSION,
    tc.VAR_FUELCONSUMPTION,
]


class Environment(Env):

    def __init__(
            self,
            config_file: str,
            traci_label: str = "default",
            timestep_length: int = 10,
            max_simulation_steps: int = 3600,
            comm_range: float = 200,
            speed_thresh: float | int = 0.1,
            yellow_length: int = 3,
            numerical_features: str = "num_vehicles",
            reward_scale: int | float = 1,
            lane_reverse: bool = False,
            pad_value: float | int = np.nan,
            log_dir: str = None,
            print_interval: int = None,
            tripinfo: bool = False,
            log_emissions: bool = False,
            gui: bool = False,
    ) -> None:

        self.config_file = config_file
        self.traci_label = traci_label
        self.timestep_length = timestep_length
        self.MAX_SIMULATION_STEPS = max_simulation_steps
        self.comm_range = comm_range
        self.speed_thresh = speed_thresh
        self.yellow_length = yellow_length
        self.numerical_features = numerical_features.split(",") if numerical_features.strip() else []
        self.reward_scale = reward_scale
        self.lane_reverse = lane_reverse
        self.pad_value = pad_value
        self.log_dir = log_dir
        self.print_interval = print_interval
        self.tripinfo = tripinfo
        self.log_emissions = log_emissions
        self.gui = gui

        net_file = ElementTree.parse(self.config_file).getroot().find('input/net-file').get('value')
        parent_dir = Path(config_file).parent.absolute()
        net_file = parent_dir.joinpath(net_file).as_posix()
        net = sumolib.net.readNet(net_file, withPrograms=True)
        self.sumolib_tl_list = net.getTrafficLights()
        self.num_signals = len(self.sumolib_tl_list)
        self.max_num_lanes = max([
            sum([len(edge.getLanes()) for edge in tl.getEdges()])
            for tl in self.sumolib_tl_list
        ])
        self.max_num_phases = max([
            len([phase.state for phase in tl.getPrograms()["0"].getPhases() if "y" not in phase.state])
            for tl in self.sumolib_tl_list
        ])

        # Configure observation space
        max_lane_num_vehicles = int((self.comm_range + MIN_GAP) / (VEHICLE_LENGTH + MIN_GAP)) + 2  # +2 for headroom
        sequential_obs_shape = (self.num_signals, self.max_num_lanes, max_lane_num_vehicles, 3)
        numerical_obs_shape = (self.num_signals, self.max_num_lanes * len(self.numerical_features))
        high = np.zeros(sequential_obs_shape)
        low = np.zeros(sequential_obs_shape)
        high[:, :, :, 0] = comm_range
        high[:, :, :, 1] = MAX_SPEED
        high[:, :, :, 2] = MAX_ACCEL
        low[:, :, :, 2] = MIN_ACCEL
        obs_space_dict = {
            "sequential_obs": spaces.Box(low=low, high=high, shape=sequential_obs_shape),
            "phase": spaces.Box(low=0, high=1, shape=(self.num_signals, self.max_num_phases)),
        }
        if self.numerical_features:
            obs_space_dict["numerical_obs"] = spaces.Box(low=0, high=max_lane_num_vehicles, shape=numerical_obs_shape)
        self.observation_space = spaces.Dict(obs_space_dict)
        self.observation = {}

        # Configure action space.
        action_space_shape = []
        for tl in self.sumolib_tl_list:
            action_space_shape.append(
                len([phase.state for phase in tl.getPrograms()["0"].getPhases() if "y" not in phase.state])
            )
        self.action_space = spaces.MultiDiscrete(action_space_shape)

        self.tl_list = []
        self.cumulative_reward = 0
        self.simulation_time = 0
        self.timestep = 0
        self.total_timestep = 0
        self.episode = 0
        self.arrived = 0
        self.current_vehicles = []
        self.subscribed_vehicle_ids = set()
        self.waiting_time = dict()  # {vehicle_id: accumulated_waiting_time}
        self.time_loss = dict()  # {vehicle_id: time_loss_since_departure}
        if self.log_emissions:  # Each value is a dictionary of {vehicle_id: total_emission}.
            self.emissions = {"CO2": dict(), "CO": dict(), "HC": dict(), "NOx": dict(), "PMx": dict(), "fuel": dict()}

        # Configure logger.
        logging_format = "[%(asctime)s][%(name)s] %(message)s"
        date_format = "%Y/%m/%d %H:%M:%S"
        formatter = logging.Formatter(logging_format, datefmt=date_format)
        console_handler = logging.StreamHandler(sys.stdout)  # used to print to the console
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger = logging.getLogger("env_" + self.traci_label)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

            # file handler for general info
            log_file_path = os.path.join(self.log_dir, f"log_env.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.info("Logging to file:" + log_file_path)

            # file handler for reward logging
            self.result_logger = logging.getLogger("result_logger")
            self.result_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(
                logging.FileHandler(os.path.join(self.log_dir, f"results_{self.traci_label}.csv"), mode="w")
            )
            header_row = "episode,timestep,total_timestep,reward,waiting_time,time_loss,num_arrived"
            if self.log_emissions:
                header_row += ",CO2,CO,HC,NOx,PMx,fuel"
            self.result_logger.info(header_row)

        self.logger.info(f"Fog environment initialized.")
        self.is_running = False

    def _init_observation(self) -> dict:
        obs = {
            "sequential_obs": np.full(self.observation_space["sequential_obs"].shape, self.pad_value, dtype=np.float32),
            "phase": np.zeros(self.observation_space["phase"].shape),
        }
        if self.numerical_features:
            obs["numerical_obs"] = np.zeros(self.observation_space["numerical_obs"].shape, dtype=np.float32)
        return obs

    def reset(self, **kwargs) -> tuple[dict, dict]:
        """
        Reset the environment and return the initial observation.
        """
        self.logger.info(
            f"Episode: {self.episode}, "
            f"Total timesteps: {self.timestep}, "
            f"cumulative reward: {self.cumulative_reward:>9.4f}, "
            f"avg waiting time: {self.get_avg_waiting_time()}, "
            f"avg time loss: {self.get_avg_time_loss()}, "
            f"num arrived: {self.arrived}"
        )
        self.logger.info(f"Resetting environment...")

        # Restart SUMO simulation.
        self._stop_simulation()
        self._start_simulation()
        self.episode += 1

        # Create a list of traffic light objects.
        self.tl_list = [
            TrafficLight(
                obj=tl,
                comm_range=self.comm_range,
                pad_num_lanes=self.max_num_lanes,
                pad_num_phases=self.max_num_phases,
                speed_thresh=self.speed_thresh,
                numerical_features=self.numerical_features
            ) for tl in self.sumolib_tl_list
        ]

        # self.observation = self._init_observation()
        self.cumulative_reward = 0
        self.simulation_time = 0
        self.timestep = 0
        self.arrived = 0
        self.subscribed_vehicle_ids = set()
        self.waiting_time = dict()
        self.time_loss = dict()
        if self.log_emissions:
            self.emissions = {"CO2": dict(), "CO": dict(), "HC": dict(), "NOx": dict(), "PMx": dict(), "fuel": dict()}

        self.logger.info(f"Environment has been reset.")
        self.observation = self.get_observation()
        return self.observation, {}

    def step(self, action: np.ndarray) -> tuple[dict, int, bool, bool, dict]:
        """
        :param action: np.ndarray, binary vector with n_traffic_light elements.
            action[i] = 0 means to keep current phase of the ith traffic light, 1 means switching to the next phase.
        :return: new observation, step reward, done or not, info.
        """

        reward = 0

        # Apply transition (yellow) phase and calculate reward.
        for i, next_phase_idx in enumerate(action):
            tl = self.tl_list[i]
            current_phase_idx = tl.get_current_phase_idx()
            transition_phase_idx = tl.transition_phase[current_phase_idx][next_phase_idx]
            tl.set_phase(transition_phase_idx)
        for _ in range(self.yellow_length):
            self.simulation_step()
            reward -= self.get_total_num_waiting_vehicles() / self.reward_scale

        # apply next phase
        for i, next_phase_idx in enumerate(action):
            tl = self.tl_list[i]
            tl.set_phase(next_phase_idx)
        for _ in range(self.timestep_length):
            self.simulation_step()
            reward -= self.get_total_num_waiting_vehicles() / self.reward_scale

        self.timestep += 1
        self.total_timestep += 1

        # Get new observation.
        self.observation = self.get_observation()

        avg_waiting_time = self.get_avg_waiting_time()
        avg_time_loss = self.get_avg_time_loss()
        num_arrived = self.get_num_arrived()

        self.cumulative_reward += reward

        if self.print_interval is not None and self.timestep % self.print_interval == 0:
            self.logger.info(
                f"Step {self.timestep:>4}, action: {action}, reward: {int(reward):>4}, "
                f"cumulative reward: {int(self.cumulative_reward):>7}, "
                f"average waiting time: {avg_waiting_time:>6.2f}, "
                f"average time loss: {avg_time_loss:>6.2f}, "
                f"num_arrived: {num_arrived:>4}"
            )

        # Determine whether this episode should end
        terminated = traci.vehicle.getSubscriptionResults("")[tc.ID_COUNT] == 0 and self.timestep > 20
        truncated = self.simulation_time >= self.MAX_SIMULATION_STEPS

        if (truncated or terminated) and hasattr(self, "result_logger"):
            info = (f"{self.episode},{self.timestep},{self.total_timestep},{self.cumulative_reward},{avg_waiting_time},"
                    f"{avg_time_loss},{num_arrived}")
            if self.log_emissions:
                avg_emissions = self.get_avg_emissions()
                info += (f",{avg_emissions['CO2']},{avg_emissions['CO']},{avg_emissions['HC']},{avg_emissions['NOx']},"
                         f"{avg_emissions['PMx']},{avg_emissions['fuel']}")
            self.result_logger.info(info)

        info = {
            "timestep": self.timestep,
            "total_timestep": self.total_timestep,
            "cumulative_reward": self.cumulative_reward,
            "avg_waiting_time": avg_waiting_time,
            "avg_time_loss": avg_time_loss,
            "num_arrived": num_arrived
        }

        return self.observation, reward, terminated, truncated, info

    def simulation_step(self) -> None:
        """
        Update environment
        """
        traci.simulationStep()
        self.simulation_time += 1
        self.arrived += traci.simulation.getSubscriptionResults()[tc.VAR_ARRIVED_VEHICLES_NUMBER]
        self.current_vehicles = traci.vehicle.getSubscriptionResults("")[tc.TRACI_ID_LIST]

        for vehicle_id in self.current_vehicles:
            if vehicle_id not in self.subscribed_vehicle_ids:
                traci.vehicle.subscribe(vehicle_id, VEHICLE_DATA_TO_SUBSCRIBE)
                traci.vehicle.subscribeLeader(vehicle_id)
                self.subscribed_vehicle_ids.add(vehicle_id)

        for tl in self.tl_list:
            tl.update_lane_vehicle_id_list()

        self._update_waiting_time()
        if self.log_emissions:
            self._update_emissions()

    def close(self) -> None:
        """
        Terminate the simulation and close the environment.
        """
        self._stop_simulation()
        self.logger.info(
            f"Environment closed. "
            f"Current episode time steps: {self.timestep}, "
            f"Total time steps: {self.total_timestep}, "
            f"cumulative reward: {self.cumulative_reward}, "
            f"avg waiting time: {self.get_avg_waiting_time()}\n"
        )
        self.total_timestep = 0
        if len(self.logger.handlers) > 1:
            self.logger.handlers[1].close()
            self.result_logger.handlers[0].close()

    def _start_simulation(self) -> None:
        sumo_binary = sumolib.checkBinary("sumo-gui") if self.gui else sumolib.checkBinary("sumo")
        cmd = [sumo_binary, "-c", self.config_file, "--no-step-log", "True", "--no-warnings", "True"]
        if self.log_dir:
            if self.tripinfo:
                cmd += ["--tripinfo-output", os.path.join(self.log_dir, f"tripinfo_ep{self.episode}.xml"),
                        "--tripinfo-output.write-unfinished"]
            if self.log_emissions:
                cmd += ["--device.emissions.probability", "1.0"]

        try:  # make sure simulation with the same label doesn't exist
            traci.switch(self.traci_label)
            sys.exit(f"SUMO instance with label {self.traci_label} already exists.")
        except TraCIException:
            traci.start(cmd, label=self.traci_label)
            self.is_running = True
            traci.vehicle.subscribe("", [tc.TRACI_ID_LIST, tc.ID_COUNT])
            traci.simulation.subscribe([tc.VAR_ARRIVED_VEHICLES_NUMBER])

    def _stop_simulation(self) -> None:
        if self.is_running:
            traci.close()
            self.is_running = False

    def get_observation(self) -> dict:
        """
        Gather features from traffic lights and aggregate them as the observation passed to the agent
        """
        obs = self._init_observation()
        for tl_idx, tl in enumerate(self.tl_list):
            tl_obs = tl.get_dict_obs()
            for lane_idx, lane_sequential_obs in enumerate(tl_obs["tl_sequential_obs"]):
                if len(lane_sequential_obs) > 0:
                    if self.lane_reverse:
                        lane_sequential_obs.reverse()
                    obs["sequential_obs"][tl_idx, lane_idx, :len(lane_sequential_obs), :] = lane_sequential_obs
            if self.numerical_features:
                obs["numerical_obs"][tl_idx] = tl_obs["tl_numerical_obs"]
            obs["phase"][tl_idx] = tl_obs["phase"]
        obs["sequential_obs"][:, :, :, 0] = self.min_max(obs["sequential_obs"][:, :, :, 0], 0, self.comm_range)
        obs["sequential_obs"][:, :, :, 1] = self.min_max(obs["sequential_obs"][:, :, :, 1], 0, MAX_SPEED)
        obs["sequential_obs"][:, :, :, 2] = self.min_max(obs["sequential_obs"][:, :, :, 2], MIN_ACCEL, MAX_ACCEL)
        if self.numerical_features:
            obs["numerical_obs"] /= self.observation_space["numerical_obs"].high
        return obs

    def min_max(self, array, orig_low, orig_high, target_low=-1, target_high=1):
        """
        Min-max normalize the input array from [orig_low, orig_high] to the [target_low, target_high] range.
        """
        return (array.clip(orig_low, orig_high) - orig_low) / (orig_high - orig_low) * (target_high - target_low) + target_low

    def get_total_num_waiting_vehicles(self) -> int:
        num_waiting = 0
        for tl in self.tl_list:
            num_waiting += tl.get_num_waiting_vehicles()
        return num_waiting

    def get_step(self) -> int:
        """
        Return the current timestep.
        """
        return self.timestep

    def _update_waiting_time(self) -> None:
        """
        Update the accumulated waiting time and time loss for every vehicle.
        """
        for vehicle_id in self.current_vehicles:
            self.time_loss[vehicle_id] = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_TIMELOSS]
            if self.speed_thresh == 0.1:
                self.waiting_time[vehicle_id] = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_ACCUMULATED_WAITING_TIME]
            else:
                if traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_SPEED] < self.speed_thresh:
                    self.waiting_time[vehicle_id] = self.waiting_time.get(vehicle_id, 0) + 1

    def _update_emissions(self) -> None:
        """
        Update emissions (CO2, CO, HC, NOx, PMx, fuel) for every vehicle
        """
        for vehicle_id in self.current_vehicles:
            co2 = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_CO2EMISSION]
            co = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_COEMISSION]
            hc = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_HCEMISSION]
            nox = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_NOXEMISSION]
            pmx = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_PMXEMISSION]
            fuel = traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_FUELCONSUMPTION]
            self.emissions["CO2"][vehicle_id] = self.emissions["CO2"].get(vehicle_id, 0) + co2
            self.emissions["CO"][vehicle_id] = self.emissions["CO"].get(vehicle_id, 0) + co
            self.emissions["HC"][vehicle_id] = self.emissions["HC"].get(vehicle_id, 0) + hc
            self.emissions["NOx"][vehicle_id] = self.emissions["NOx"].get(vehicle_id, 0) + nox
            self.emissions["PMx"][vehicle_id] = self.emissions["PMx"].get(vehicle_id, 0) + pmx
            self.emissions["fuel"][vehicle_id] = self.emissions["fuel"].get(vehicle_id, 0) + fuel

    def get_avg_waiting_time(self) -> float:
        """
        Return the average waiting time of all vehicles from start to the current time step.
        """
        return sum(self.waiting_time.values()) / len(self.waiting_time) if len(self.waiting_time) > 0 else 0

    def get_avg_time_loss(self) -> float:
        """
        Return the average time loss of all vehicles from start to the current time step.
        """
        return sum(self.time_loss.values()) / len(self.time_loss) if len(self.time_loss) > 0 else 0

    def get_avg_emissions(self) -> dict:
        """
        Return a dictionary of each emission type and its average emissions per vehicle.
        """
        avg_emissions = dict()
        for key, table in self.emissions.items():
            avg_emissions[key] = sum(table.values()) / len(table) if len(table) > 0 else 0
        return avg_emissions

    def get_num_arrived(self) -> int:
        """
        Return the number of vehicles that have arrived at their destinations and exited the simulation.
        """
        return self.arrived
