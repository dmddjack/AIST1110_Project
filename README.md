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
        
   to play the game

## TO-DO
- [x] Handle situations when different enemies touch one another \
- [x] Use DQN to train a model
- [ ] Add sufficient comments in all scripts
- [x] Change variables names in tankwar_train.py
- [x] Delete unnecessary codes in all scripts
- [x] Solve enemy collision bug
- [x] Add -fast mode for 20-min training
- [ ] Do experiment:
    - [x] Time interval
    - [x] Huber or MSE loss
    - [ ] Modify reward
- [ ] Reformat all scripts such that they are (mostly) consistent

## Ideas
- [x] Difficulty
- [ ] Block
- [x] Animations
- [x] Multiple scenes
    - [x] Title scene
    - [x] Ending scene
- [ ] Power ups
