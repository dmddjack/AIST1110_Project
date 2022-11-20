import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="The render mode",
                    choices=["human", "human_rand", "rgb_array"], 
                    default="rgb_array")
parser.add_argument("-sh", "--starting_hp", type=int,
                    help="The starting HP of the player",
                    choices=range(1, 11), metavar="[1-10]",
                    default=3)
parser.add_argument("-s", "--seed", type=int, 
                    help="The seed for random number generator",
                    default=None)
parser.add_argument("-e", "--episodes", type=int, 
                    help="The number of episodes", 
                    default=1000)
parser.add_argument("-ms", "--max_steps", type=int, 
                    help="The maximum number of steps in an episode", 
                    default=5400)
parser.add_argument("-fps", "--fps", type=int, 
                    help="The rendering speed in frames per second", 
                    choices=range(1, 61), metavar="[1-60]", 
                    default=None)
parser.add_argument("-f", "--file", type=str, 
                    help="The file name of the Q-table file",
                    default=None)
args = parser.parse_args()
print(args)
