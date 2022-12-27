#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="The render mode (the "
                                                   "human_rand mode can only be used in "
                                                   "tankwar_play.py)",
                    choices=["human", "human_rand", "rgb_array"], 
                    default="rgb_array")
parser.add_argument("-sh", "--starting_hp", type=int,
                    help="The starting HP of the player",
                    choices=range(1, 11), metavar="[1-10]",
                    default=3)
parser.add_argument("-s", "--seed", type=int, 
                    help="The seed for random number generator (if this argument "
                         "is specified in tankwar_play.py or tankwar_test.py, the "
                         "sequence of randomly generated action of enemy is identical "
                         "for the same seed used in two different runs so that for the "
                         "same model and the same seed, tankwar_test.py will yield the "
                         "same result.)",
                    default=None)
parser.add_argument("-d", "--difficulty", type=int, 
                    help="The level of enemy tank path finding AI",
                    choices=(0, 1), metavar="[0, 1]",
                    default=0)
parser.add_argument("-fe", "--full_enemy", action="store_true", 
                    help="Fix the number of enemies to its maximum value")
parser.add_argument("-e", "--episodes", type=int, 
                    help="The number of playing episodes", 
                    default=1000)
parser.add_argument("-traine", "--train_episodes", type=int, 
                    help="The number of training episodes", 
                    default=1000)
parser.add_argument("-teste", "--test_episodes", type=int, 
                    help="The number of testing episodes", 
                    default=100)
parser.add_argument("-ms", "--max_steps", type=int, 
                    help="The maximum number of steps in an episode when FPS=30",
                    default=7200)
parser.add_argument("-fast", "--fast", action="store_true", 
                    help="Use fast mode to train DQN model in around 20 minutes")
parser.add_argument("-fps", "--fps", type=int, 
                    help="Frames per second (Although possible, it would be "
                         "better not to use this argument in tankwar_train.py "
                         "and tankwar_test.py)",
                    choices=(15, 30, 60), metavar="[15, 30, 60]", 
                    default=30)
parser.add_argument("-f", "--file", type=str, 
                    help="The file name of the HDF5 model file (without .h5 suffix)",
                    default=None)
args = parser.parse_args()
print(args)
