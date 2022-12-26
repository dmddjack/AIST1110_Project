# Command-line arguments
# This document partly follows Google developer documentation style guide. For more information, see https://developers.google.com/style/code-syntax.
# Available command:
python tankwar_play.py [-m MODE] [-sh STARTING_HP] [-s SEED] [-d DIFFICULTY] [-fe] [-e EPISODES] [-ms MAX_STEPS] [-fps FPS] [-h]
python tankwar_train.py -s SEED [-m {human | rgb_array}] [-sh STARTING_HP] [-d DIFFICULTY] [-fe] [-traine TRAIN_EPISODES] [-ms MAX_STEPS] [-fast] [-fps FPS] [-h]
python tankwar_test.py -f FILE [-m {human | rgb_array}] [-sh STARTING_HP] [-s SEED] [-d DIFFICULTY] [-fe] [-teste TEST_EPISODES] [-ms MAX_STEPS] [-fps FPS] [-h]

Argument name       Argument name alias    Definition & usage                      Argument data type    Feasible value                Default value

--mode              -m                     The render mode (the human_rand mode    str                   MODE = {human | human_rand    rgb_array
                                           can only be used in tankwar_play.py)                          | rgb_array}
                                           
--starting_hp       -sh                    The starting HP of the player           int                   1 <= STARTING_HP <= 10        3

--seed              -s                     The seed for random number generator    int | NoneType        N/A                           None
                                           (if this argument is specified in 
                                           tankwar_play.py or tankwar_test.py,
                                           every episode of the game will be 
                                           the same)

--difficulty        -d                     The difficulty of enemy tank path       int                   DIFFICULTY = (0 | 1)          0
                                           finding AI

--full_enemy        -fe                    Fix the number of enemies to its        N/A                   N/A                           False
                                           maximum value 

--episodes          -e                     The number of playing episodes          int                   EPISODES > 0                  1000

--train_episodes    -traine                The number of training episodes         int                   TRAIN_EPISODES > 0            1000

--test_episodes     -teste                 The number of testing episodes          int                   TEST_EPISODES > 0             100

--max_steps         -ms                    The maximum number of steps in          int                   MAX_STEPS > 0                 7200
                                           an episode when FPS=30

--fast              -fast                  Use fast mode to train DQN model        N/A                   N/A                           False
                                           in around 20 minutes

--fps               -fps                   The rendering frames per second         int                   FPS = (15 | 30 | 60)          30
                                           (Although possible, it would be 
                                           better not to use this argument in
                                           tankwar_train.py and tankwar_test.py)

--file              -f                     The file name of the HDF5 model file    str |  NoneType       N/A                           None
                                           (If the file name is model.h5, only
                                           model should be inputed)
                                           
--help              -h                     Show the help message and exit          N/A                   N/A                           N/A
