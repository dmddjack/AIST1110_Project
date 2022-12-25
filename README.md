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
        
1. Press "q" or "Escape" to quit (except tankwar_play_v2.py)

## TO-DO
- [x] Handle situations when different enemies touch one another \
  ***The current handling method is to kill both enemies. It would be better to reverse their directions.***
- [x] Use DQN to train a model
- [ ] Add sufficient comments in all scripts
- [ ] Change variables names in tankwar_train.py
- [ ] Delete unnecessary codes in all scripts
- [x] Solve enemy collision bug

## Ideas
- [x] Difficulty
- [ ] Block
- [x] Animations
- [x] Multiple scenes
    - [x] Title scene
    - [x] Ending scene
- [ ] Power ups
