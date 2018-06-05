#!/usr/bin/env python3

from functools import partial

from baselines import logger
from acktr_disc import learn
from cmd_util import make_atari_env, atari_arg_parser
from baselines.ppo2.policies import CnnPolicy

from vec_frame_stack import VecFrameStack

NUM_ENV = 10
NUM_CPU = 32

def train(env_id, num_timesteps, seed, num_cpu):
    env = VecFrameStack(make_atari_env(env_id, NUM_CPU, seed), NUM_ENV)
    policy_fn = partial(CnnPolicy, one_dim_bias=True)
    learn(policy_fn,
      env,
      seed,
      total_timesteps=int(num_timesteps * 1.1),
      nprocs=num_cpu,
      log_interval=20,
      save_interval=1000
    )
    env.close()

def main():
    args = atari_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_cpu=NUM_CPU)

if __name__ == '__main__':
    main()
