# Command-line arguments
# Available command:
python tankwar_play.py [-m <MODE>] [-sh <STARTING_HP>] [-s <SEED>] [-d <DIFFICULTY>] [-fe] [-e <EPISODES>] [-ms <MAX_STEPS>] [-fps <FPS>]
python tankwar_train.py [-m <human|rgb_array>] [-sh <STARTING_HP>] [-s <SEED>] [-d <DIFFICULTY>] [-fe] [-e <EPISODES>] [-ms <MAX_STEPS>] [-fps <FPS>]
python tankwar_test.py [-m <MODE>] [-sh <STARTING_HP>] [-s <SEED>] [-d <DIFFICULTY>] [-fe] [-e <EPISODES>] [-ms <MAX_STEPS>] [-fps <FPS>]

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
parser.add_argument("-d", "--difficulty", type=int, 
                    help="The difficulty of AI",
                    choices=(0, 1), metavar="[0, 1]",
                    default=0)
parser.add_argument("-fe", "--full_enemy", action="store_true", 
                    help="Use all enemies")
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
                    default=7200)
parser.add_argument("-fast", "--fast", action="store_true", 
                    help="Train the model in around 20 minutes")
parser.add_argument("-fps", "--fps", type=int, 
                    help="The rendering speed in frames per second", 
                    choices=(15, 30, 60), metavar="[15, 30, 60]", 
                    default=30)
parser.add_argument("-f", "--file", type=str, 
                    help="The file name of the HDF5 model file",
                    default=None)
args = parser.parse_args()
print(args)
