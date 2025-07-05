import sumolib
import traci
from traci import tc
from itertools import chain


MIN_GAP = 1
VEHICLE_LENGTH = 5
MAX_SPEED = 13.89


class TrafficLight:

    def __init__(
            self,
            obj: sumolib.net.TLS,
            pad_num_lanes: int,
            pad_num_phases: int,
            comm_range: float | int = 50,
            speed_thresh: float = 0.1,
            numerical_features: tuple[str] | list[str] = ("num_vehicles",),
            yellow_duration: int = 3,
    ):

        self.function_table = {
            "num_vehicles": self.get_lane_num_vehicles,
            "num_waiting": self.get_lane_num_waiting_vehicles,
            "pressure": self.get_lane_total_pressure,
            "first_distance": self.get_lane_first_vehicle_distance,
            "first_speed": self.get_lane_first_vehicle_speed,
            "first_waiting_time": self.get_lane_first_vehicle_waiting_time,
            "avg_distance": self.get_lane_avg_vehicle_distance,
            "avg_speed": self.get_lane_avg_speed,
        }

        self.id = obj.getID()
        self.pad_num_lanes = pad_num_lanes
        self.pad_num_phases = pad_num_phases
        self.DETECT_RANGE = comm_range
        self.STOP_SPEED_THRESH = speed_thresh
        self.numerical_features = numerical_features
        self.yellow_duration = yellow_duration

        self.MAX_QUEUE = int((self.DETECT_RANGE + MIN_GAP) / (VEHICLE_LENGTH + MIN_GAP)) + 2  # +2 for headroom
        self.edges = sorted(obj.getEdges(), key=lambda x: x.getID())
        self.lane_ids = []
        self.lanes = []
        self.lane_table = dict()
        for edge in self.edges:
            for lane in edge.getLanes():
                lane_id = lane.getID()
                self.lane_ids.append(lane_id)
                self.lanes.append(lane)
                self.lane_table[lane_id] = lane
                traci.lane.subscribe(lane_id, [tc.LAST_STEP_VEHICLE_ID_LIST])

        self.phases = [phase.state for phase in obj.getPrograms()["0"].getPhases() if "y" not in phase.state]
        self.num_phases = len(self.phases)
        self.transition_phase = [[0] * self.num_phases for _ in range(self.num_phases)]
        self.next_phase = {i: {0: i, 1: (i + 1) % len(self.phases)} for i in range(len(self.phases))}
        self._build_phases()
        self.current_phase_idx = 0

        self.lane_vehicles = {lane_id: [] for lane_id in self.lane_ids}

    def _build_phases(self):
        """
        Inspired by RESCO (https://github.com/Pi-Star-Lab/RESCO).
        Build the transitional (yellow) phases and the corresponding mapping. The result is the 2D array called
        transition_phase, where transition_phase[i][j] is the transitional phase index to apply before switching from
        phase[i] to phase[j].
        """
        for i in range(self.num_phases):
            state_i = self.phases[i]
            for j in range(self.num_phases):
                if i == j:
                    self.transition_phase[i][j] = i
                    continue
                state_j = self.phases[j]
                yellow_str = ''
                for a, b in zip(state_i, state_j):
                    if a == "G" or a == "g":
                        if b == "r":
                            yellow_str += "y"
                        elif b == "s":
                            yellow_str += "s"
                        else:
                            yellow_str += a
                    else:
                        yellow_str += a

                self.phases.append(yellow_str)
                self.transition_phase[i][j] = len(self.phases) - 1

    def get_list_obs(self):
        """
        Extract features without normalization. To be forwarded to VecNormalize for normalization.
        """
        waiting = []
        tl_numerical_obs = [[0] * self.pad_num_lanes for _ in range(len(self.numerical_features))]
        lane_idx = 0
        for lane in self.lanes:
            lane_id = lane.getID()
            for i, feature in enumerate(self.numerical_features):
                tl_numerical_obs[i][lane_idx] = self.function_table[feature](lane_id)
            lane_idx += 1
            waiting.append(self.get_lane_num_waiting_vehicles(lane_id))

        current_phase_one_hot = [0] * self.pad_num_phases
        current_phase_one_hot[self.get_current_phase_idx()] = 1

        features = list(chain(*tl_numerical_obs, current_phase_one_hot))

        return features

    def get_dict_obs(self) -> dict:
        """
        :return: extracted high-level feature sent to fog node
        """
        tl_sequential_obs = [[] for _ in range(self.pad_num_lanes)]
        tl_numerical_obs = [[0] * self.pad_num_lanes for _ in range(len(self.numerical_features))]
        for lane_idx, lane in enumerate(self.lanes):
            lane_id = lane.getID()
            tl_sequential_obs[lane_idx] = self.get_lane_sequential_obs(lane_id)
            if self.numerical_features:
                for feature_idx, numerical_feature in enumerate(self.numerical_features):
                    tl_numerical_obs[feature_idx][lane_idx] = self.function_table[numerical_feature](lane_id)
        tl_numerical_obs = list(chain.from_iterable(tl_numerical_obs))

        current_phase_one_hot = [0] * self.pad_num_phases
        current_phase_one_hot[self.get_current_phase_idx()] = 1

        obs = {
            "tl_sequential_obs": tl_sequential_obs,
            "tl_numerical_obs": tl_numerical_obs,
            "phase": current_phase_one_hot,
        }
        return obs

    def get_lane_sequential_obs(self, lane_id: str):
        vehicles = self.lane_vehicles[lane_id]
        lane_length = traci.lane.getLength(lane_id)
        return [
            [
                lane_length - traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_LANEPOSITION],
                traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_SPEED],
                traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_ACCELERATION],
            ]
            for veh_id in vehicles
        ]

    def get_num_waiting_vehicles(self) -> int:
        num_waiting = 0
        for lane_id in self.lane_ids:
            num_waiting += self.get_lane_num_waiting_vehicles(lane_id)
        return num_waiting

    def get_lane_num_waiting_vehicles(self, lane_id: str) -> int:
        """
        Return the number of vehicles that are waiting within the detect range of a given lane. By default, SUMO
        considers a vehicle stopped when its speed is lower than 0.1 m/s. This can be customized by changing the value
        of self.STOP_SPEED_THRESH.
        """
        num_waiting = 0
        for vehicle_id in self.lane_vehicles[lane_id]:
            if traci.vehicle.getSubscriptionResults(vehicle_id)[tc.VAR_SPEED] <= self.STOP_SPEED_THRESH:
                num_waiting += 1
        return num_waiting

    def get_lane_first_vehicle_distance(self, lane_id: str) -> int | float:
        """
        Return the distance in meters between the first vehicle and the stop line for a given lane.
        """
        vehicles = self.lane_vehicles[lane_id]
        if not vehicles:
            return self.DETECT_RANGE
        return traci.lane.getLength(lane_id) - traci.vehicle.getSubscriptionResults(vehicles[-1])[tc.VAR_LANEPOSITION]

    def get_lane_first_vehicle_speed(self, lane_id: str) -> int | float:
        vehicles = self.lane_vehicles[lane_id]
        if not vehicles:
            return MAX_SPEED
        return traci.vehicle.getSubscriptionResults(vehicles[-1])[tc.VAR_SPEED]

    def get_lane_first_vehicle_waiting_time(self, lane_id) -> int | float:
        vehicles = self.lane_vehicles[lane_id]
        if not vehicles:
            return 0
        return traci.vehicle.getSubscriptionResults(vehicles[-1])[tc.VAR_WAITING_TIME]

    def get_lane_avg_vehicle_distance(self, lane_id: str) -> int | float:
        vehicles = self.lane_vehicles[lane_id]
        if len(vehicles) < 2:
            return self.DETECT_RANGE
        distances = []
        for i, veh_id in enumerate(vehicles[::-1]):
            pos = traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_LANEPOSITION]
            distance_to_stopline = traci.lane.getLength(lane_id) - pos
            if distance_to_stopline > self.DETECT_RANGE:
                break
            if i == 0:
                # when waiting behind the stopline, the distance between the vehicle and the line is 1m
                distances.append(distance_to_stopline - 1)
            else:
                distances.append(traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_LEADER][1])
        return sum(distances) / len(distances) if distances else self.DETECT_RANGE

    def get_lane_avg_speed(self, lane_id: str) -> int | float:
        vehicles = self.lane_vehicles[lane_id]  # self.get_lane_vehicle_id_list_in_range(lane_id)
        total_speed = 0
        for veh_id in vehicles:
            total_speed += traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_SPEED]
        return total_speed / len(vehicles) if vehicles else MAX_SPEED

    def get_lane_total_pressure(self, lane_id: str) -> int:
        """
        Return the total pressure of movements involving the given incoming lane. An incoming lane may be connected to
        one or more outgoing lanes, in which case all outgoing lanes are treated aggregatively as one lane
        """
        lane = self.lane_table[lane_id]
        num_incoming_vehicles = self.get_lane_num_vehicles(lane_id)
        num_outgoing_vehicles = 0
        for out_lane in lane.getOutgoingLanes():
            num_outgoing_vehicles += self.get_lane_num_vehicles(out_lane.getID(), reverse=True)
        return num_incoming_vehicles - num_outgoing_vehicles

    def get_lane_num_vehicles(self, lane_id: str, reverse: bool = False) -> int:
        """
        :param lane_id: the ID of the lane of interest
        :param reverse: if True, will start counting from the upstream end of the lane
        :return: number of vehicles that are within the detection range
        """
        if not reverse:
            return len(self.lane_vehicles[lane_id])
        else:
            return len(self.get_lane_vehicle_id_list_in_range(lane_id, reverse=True))

    def get_lane_vehicle_id_list_in_range(self, lane_id: str, reverse: bool = False) -> list[str]:
        """
        :param lane_id: the ID of the lane of interest
        :param reverse: if True, will start counting from the upstream end of the lane
        :return: list of IDs of vehicles that are within the detection range
        """
        try:
            vehicle_ids = traci.lane.getSubscriptionResults(lane_id)[tc.LAST_STEP_VEHICLE_ID_LIST]
        except KeyError:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        if not reverse:
            vehicle_ids = vehicle_ids[::-1]  # vehicle_ids is sorted from upstream to downstream by default
        vehicle_ids_in_range = []
        for veh_id in vehicle_ids:
            pos = traci.vehicle.getSubscriptionResults(veh_id)[tc.VAR_LANEPOSITION]
            distance_to_signal = pos if reverse else traci.lane.getLength(lane_id) - pos
            if distance_to_signal > self.DETECT_RANGE:
                break
            vehicle_ids_in_range.append(veh_id)
        return vehicle_ids_in_range

    def update_lane_vehicle_id_list(self):
        """
        Update the IDs of vehicles within the detection range in each lane.
        """
        for lane in self.lanes:
            lane_id = lane.getID()
            self.lane_vehicles[lane_id] = self.get_lane_vehicle_id_list_in_range(lane_id)

    def get_passed_vehicles(self) -> int:
        """
        Return the number of vehicles that passed the stop line during the last time step.
        """
        num_passed = 0
        for lane_id in self.lane_ids:
            # Right lane not considered
            # if lane_id[-1] == '0':
            #     continue
            # waiting_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)[::-1]
            vehicles = self.get_lane_vehicle_id_list_in_range(lane_id)
            if self.lane_vehicles[lane_id] and vehicles and self.lane_vehicles[lane_id][0] != vehicles[0]:
                num_passed += 1
            self.lane_vehicles[lane_id] = vehicles[:]
        return num_passed

    def get_current_phase_idx(self) -> int:
        """
        Return the index of the current phase. Complete list of phases can be found in .net.xml files.
        """
        return self.current_phase_idx

    def set_phase(self, phase_idx):
        """
        Set the phase to self.phases[phase_idx].
        """
        traci.trafficlight.setRedYellowGreenState(self.id, self.phases[phase_idx])
        self.current_phase_idx = phase_idx

    def get_id(self) -> str:
        """
        Return the ID of the traffic light.
        """
        return self.id

    def get_num_phases(self) -> int:
        """
        Return the total number of phases excluding transition (yellow) phases.
        """
        return self.num_phases
