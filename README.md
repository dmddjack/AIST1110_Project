# AIST1110_Project
Project Name: Tank War

## Steps
1. Open your own conda environment
1. Do

        cd gym-tankwar
        
1. If you are using Windows, do

        setup.bat
        
   If you are using MacOS, do
   
        sh setup.sh
        
1. Do

        cd ../tankwar
        
1. Do

        python tankwar_play_v1.py
        
   to expose the game in a non-GUI mode, or
   
        python tankwar_play_v1.py -m human_rand
        
   to visualize the gameplay GUI in a read-only human mode, or
   
        python tankwar_play_v1.py -m human
        
   or
   
        python tankwar_play_v2.py
        
   to play the game
        
1. Press "q" or "Escape" to quit

## TO-DO
- [ ] Play audio
  - [ ] Two shooting sound effects
  - [ ] Two moving sound effects
  - [ ] Explosion sound effects
  - [ ] Background music
- [ ] Design a better algorithm for the enemies' movement
- [ ] Handle situations when different enemies touch one another\
  ***The current handling method is to kill both enemies. It would be better to reverse their directions.***
- [ ] Determine whether to use a normal observation space or a normalized observation space
- [ ] Use Q-table or DQN to train a model

## Ideas
- [ ] Difficulty
- [ ] Block
- [ ] Animations
- [ ] Multiple scenes]
