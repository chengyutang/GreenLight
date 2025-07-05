from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from environment import Environment
from utils import make_env
import os
import sys
import argparse
import time
import logging
import json


if __name__ == "__main__":

    # parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--model-dir", required=True, type=str, help="Path to the model (and normalizer) file")
    parser.add_argument("--sumocfg", required=True, type=str, help="Path to the .sumocfg file")
    parser.add_argument("--max-simulation-steps", type=int, default=3600,
                        help="Maximum number of simulation steps of each episode")
    parser.add_argument("--timestep-length", type=int, default=10,
                        help="Number of simulation steps between two actions")
    parser.add_argument("--yellow-length", type=int, default=3,
                        help="Duration (in seconds) of the yellow (transitional) phase between two regular phases")
    parser.add_argument("--range", type=int, default=200,
                        help="Sensing range of the traffic signal. Vehicles out of range will be ignored")
    parser.add_argument("--label", type=str, default="default", help="TraCI label")
    parser.add_argument("--numerical-features", type=str, default="num_vehicles",
                        help="Comma-separated lane feature(s) used to form the observation (along with the current "
                             "phase and the running elapsed time of current phase). Available options: "
                             "\"num_vehicles\", \"num_waiting\", \"pressure\", \"first_distance\", \"first_speed\", "
                             "\"first_waiting_time\", \"avg_distance\", \"avg_speed \"")
    parser.add_argument("--lane-reverse", type=eval, choices=[True, False], default="True",
                        help="If True, the lane features will be stored in the order of from the entering end to the "
                             "stop line end of the lane, i.e., the same direction as the traffic flow")
    parser.add_argument("--reward-scale", type=float, default=1, help="The value the reward to be divided by")
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run. Default: 1")
    parser.add_argument("--deterministic", action="store_true", help="Whether the agent acts deterministically")
    parser.add_argument("--log-dir", type=str, help="Directory for the log files")
    parser.add_argument("--tripinfo", action="store_true",
                        help="Enabling this will generate the tripinfo files in the logging directory by appending "
                             "\"--tripinfo-output tripinfo_ep[x].xml\" to the sumo command. Requires \"--log-dir\"")
    parser.add_argument("--log-emissions", action="store_true",
                        help="Enabling this will generate the emissions files in the logging directory by appending "
                             "\"--emission-output emissions_ep[x].xml\" to the sumo command. Requires \"--log-dir\"")
    args = parser.parse_args()

    # set up loggers
    formatter = logging.Formatter("[%(asctime)s][%(name)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    logger = logging.getLogger("main_test_" + args.label)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)  # used to print to the console
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if args.log_dir is not None:

        os.makedirs(args.log_dir, exist_ok=True)

        # save configuration as json
        with open(os.path.join(args.log_dir, "test_config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        # file handler of the main logger
        log_file_path = os.path.join(args.log_dir, f"log_test.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file: " + log_file_path)

    # set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # load testing model
    model = PPO.load(os.path.join(args.model_dir, "model"))
    model.set_random_seed(args.seed)

    # setup environment
    env = DummyVecEnv([
        make_env(
            Environment,
            args.sumocfg,
            traci_label=args.label,
            timestep_length=args.timestep_length,
            max_simulation_steps=args.max_simulation_steps,
            comm_range=args.range,
            yellow_length=args.yellow_length,
            numerical_features=args.numerical_features,
            lane_reverse=args.lane_reverse,
            reward_scale=args.reward_scale,
            log_dir=args.log_dir,
            tripinfo=args.tripinfo,
            log_emissions=args.log_emissions,
        )
    ])
    # If previously trained using "--norm-wrapper True", there should be an "env_norm_wrapper" file under --model_dir.
    if "env_norm_wrapper" in os.listdir(args.model_dir):
        env = VecNormalize.load(os.path.join(args.model_dir, "env_norm_wrapper"), env)
        env.training = False
        env.norm_reward = False

    waiting_time_list = []
    time_loss_list = []
    throughput_list = []
    done = False
    obs = env.reset()
    start = time.time()

    for i in range(args.n_episodes):

        logger.info(f"Testing episode {i + 1}/{args.n_episodes}")
        decision_time = []
        info = dict()

        while not done:

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)

        avg_waiting_time = info[0]["avg_waiting_time"]
        avg_time_loss = info[0]["avg_time_loss"]
        cumulative_reward = info[0]["cumulative_reward"]
        throughput = info[0]["num_arrived"]

        # print and log evaluations at the end of each episode
        logger.info(
            f"Total reward: {cumulative_reward}, "
            f"average waiting time: {avg_waiting_time}, "
            f"average time loss: {avg_time_loss}\n"
        )

        waiting_time_list.append(avg_waiting_time)
        time_loss_list.append(avg_time_loss)
        throughput_list.append(throughput)

        done = False

        if args.deterministic:
            break

    env.close()

    testing_time = time.time() - start
    logger.info(f"{'#' * 100}\nTest finished. Time elapsed: {testing_time} s, or {testing_time / 60} min.")
    logger.info(f"Waiting time: {waiting_time_list}.\n Average: {sum(waiting_time_list) / len(waiting_time_list)} s\n.")
    logger.info(f"Time loss: {time_loss_list}.\n Average: {sum(time_loss_list) / len(time_loss_list)} s\n.")
    logger.info(f"Throughput: {throughput_list}.\n Average: {sum(throughput_list) / len(throughput_list)}\n")

    if args.log_dir:
        logger.handlers[1].close()
        # SB3 vecenv automatically resets at the end of an episode, so an extra tripinfo file will be created
        if args.tripinfo and f"tripinfo_ep{args.n_episodes}.xml" in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f"tripinfo_ep{args.n_episodes}.xml"))
