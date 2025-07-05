import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from environment import Environment
from features_extractor import FeaturesExtractor
from utils import get_lr_schedule, make_env
import time
import os
import sys
import logging
import argparse
import json


if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--log-dir", type=str,
                        help="Directory for the log files. If not specified, nothing will be saved. Also required to "
                             "enable the --tripinfo option")

    # environment arguments
    parser.add_argument("--sumocfg", required=True, type=str, help="Path to the .sumocfg file")
    parser.add_argument("--label", type=str, default="default", help="TraCI label")
    parser.add_argument("--max-simulation-steps", type=int, default=3600,
                        help="Maximum number of simulation steps (seconds) of each episode")
    parser.add_argument("--timestep-length", type=int, default=10,
                        help="Number of simulation steps between two actions")
    parser.add_argument("--yellow-length", type=int, default=3,
                        help="Duration (in seconds) of the yellow (transitional) phase between two regular phases")
    parser.add_argument("--range", type=int, default=200,
                        help="Sensing range of the traffic signal. Vehicles out of range will be ignored")
    parser.add_argument("--numerical-features", type=str, default="num_vehicles",
                        help="Comma-separated lane feature(s) used to form the observation (along with the current "
                             "phase and the running elapsed time of current phase). Available options: "
                             "\"num_vehicles\", \"num_waiting\", \"pressure\", \"first_distance\", \"first_speed\", "
                             "\"first_waiting_time\", \"avg_distance\", \"avg_speed \"")
    parser.add_argument("--lane-reverse", type=eval, choices=[True, False], default="False",
                        help="If True, the lane features will be stored in the order of from the entering end to the "
                             "stop line end of the lane, i.e., the same direction as the traffic flow")
    parser.add_argument("--norm-wrapper", type=eval, choices=[True, False], default="False",
                        help="Whether to use the VecNormalized wrapper ")
    parser.add_argument("--reward-scale", type=float, default=1, help="The value the reward to be divided by")
    parser.add_argument("--print-interval", type=int)

    # training arguments
    parser.add_argument("--n-processes", type=int, default=1,
                        help="Number of parallel environments used to train the model")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--n-steps", type=int, default=300, help="Batch size of each update")
    parser.add_argument("--batch-size", type=int, help="Mini-batch size. If not specified, it equals --nsteps")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epoch when optimizing the surrogate loss")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr-decay-start", type=float, default=1.0,
                        help="Select the ratio of remaining progress when the lr decay begins. If 1.0, the lr decay "
                             "will start at the very beginning. If 0.0, the lr will remain constant")
    parser.add_argument("--final-lr", type=float, default=0.0, help="Final learning rate for linear annealing")
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE coefficient")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Clipping range of PPO")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.9, help="Value function coefficient")
    parser.add_argument("--recurrent", type=str, default="rnn", choices=["rnn", "lstm"],
                        help="Choose from \"rnn\" and \"lstm\". Only one layer is supported correctly.")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden dimension of RNN/LSTM")
    parser.add_argument("--embedding-size", type=int, default=64,
                        help="Embedding size of the attention layer.")
    parser.add_argument("--attn-blocks", type=int, default=1,
                        help="Number of self-attention blocks in the features extractor. "
                             "If set to 0, an MLP features extractor will be used instead")
    parser.add_argument("--net-arch", type=eval, default="{'pi': [128, 32], 'vf': [128, 32]}",
                        help="Architectures of the value network and policy network. Should be in the form of "
                             "{'pi': [layer_1_dim, layer_2_dim, ...], 'vf': [layer_1_dim, layer_2_dim, ...]}")
    parser.add_argument("--bias", type=eval, choices=[True, False], default="True",
                        help="Whether to use bias or not in the attention features extractor")
    parser.add_argument("--activation", type=str, default="Tanh",
                        help="Activation function. Must be one of the PyTorch activation names, such as \"Tanh\" or "
                             "\"ReLU\", as it will be passed to the model using getattr(torch.nn, activation).")

    args = parser.parse_args()

    # save final configurations to json file if log is enabled
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    # setup logger
    formatter = logging.Formatter("[%(asctime)s][%(name)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    logger = logging.getLogger("main_train_" + args.label)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)  # used to print to the console
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if args.log_dir is not None:
        log_file_path = os.path.join(args.log_dir, f"log_main.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file: " + log_file_path)

    # set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # crate environment
    env_fn_list = [
        make_env(
            Environment,
            args.sumocfg,
            traci_label=args.label + "_" + str(i),
            timestep_length=args.timestep_length,
            max_simulation_steps=args.max_simulation_steps,
            comm_range=args.range,
            yellow_length=args.yellow_length,
            numerical_features=args.numerical_features,
            lane_reverse=args.lane_reverse,
            reward_scale=args.reward_scale,
            log_dir=args.log_dir,
            print_interval=args.print_interval,
        )
        for i in range(args.n_processes)
    ]
    env = DummyVecEnv(env_fn_list) if args.n_processes == 1 else SubprocVecEnv(env_fn_list, start_method="spawn")
    if args.norm_wrapper:
        env = VecNormalize(env, clip_obs=float("inf"), clip_reward=float("inf"), norm_obs_keys=["numerical_obs"])

    # create model
    policy_kwargs = dict(
        features_extractor_class=FeaturesExtractor,
        features_extractor_kwargs=dict(
            recurrent=args.recurrent,
            recur_hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            n_attn_blocks=args.attn_blocks,
            bias=args.bias,
        ),
        net_arch=args.net_arch,
        activation_fn=getattr(torch.nn, args.activation),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=get_lr_schedule(init_lr=args.lr, start_decay_at=args.lr_decay_start, end_lr=args.lr/10),
        n_steps=args.n_steps,
        batch_size=args.batch_size if args.batch_size is not None else args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=2,
        seed=args.seed,
        device=device,
    )

    logger.info("\nStart training...")
    start = time.time()
    total_timesteps = args.max_simulation_steps // (args.timestep_length + args.yellow_length) * args.n_episodes
    model.learn(total_timesteps=total_timesteps * args.n_processes)
    training_time = time.time() - start

    env.close()
    if args.log_dir is not None:
        model.logger.close()
        model.save(os.path.join(args.log_dir, "model"))
        if args.norm_wrapper:
            env.save(os.path.join(args.log_dir, "env_norm_wrapper"))
        logger.info(f"Training finished. Time elapsed: {training_time / 60} min. Model saved to {args.log_dir}")
        logger.handlers[1].close()
    else:
        logger.info(f"Training finished. Time elapsed: {training_time / 60} min.")
