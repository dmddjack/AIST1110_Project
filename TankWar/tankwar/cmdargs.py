import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str, help="The render mode",
                    choices=["human", "human_rand", "rgb_array"], 
                    default="rgb_array")
parser.add_argument("-sh", "--starting_hp", type=int,
                    help="The starting HP of the player",
                    choices=range(1, 11), metavar="[1-10]",
                    default=3)
parser.add_argument("-ww", "--window_width", type=int,
                    help="The width of the pygame window",
                    choices=range(450, 601), metavar="[450-600]",
                    default=600)
parser.add_argument("-wh", "--window_height", type=int,
                    help="The height of the pygame window",
                    choices=range(300, 401), metavar="[300-400]",
                    default=400)
parser.add_argument("-me", "--max_enemies", type=int,
                    help="The maximum number of enemies",
                    choices=range(1, 5), metavar="[1-4]",
                    default=4)
parser.add_argument("-s", "--seed", type=int, 
                    help="The seed for random number generator",
                    default=None)
parser.add_argument("-e", "--episodes", type=int, 
                    help="The number of episodes", 
                    default=1000)
parser.add_argument("-traine", "--train_episodes", type=int, 
                    help="The number of train episodes", 
                    default=1000)
parser.add_argument("-teste", "--test_episodes", type=int, 
                    help="The number of test episodes", 
                    default=100)
parser.add_argument("-ms", "--max_steps", type=int, 
                    help="The maximum number of steps in an episode", 
                    default=3600)
parser.add_argument("-fps", "--fps", type=int, 
                    help="The rendering speed in frames per second", 
                    choices=range(15, 61), metavar="[15-60]", 
                    default=None)
parser.add_argument("-f", "--file", type=str, 
                    help="The file name of the HDF5 model file",
                    default=None)
args = parser.parse_args()
print(args)
