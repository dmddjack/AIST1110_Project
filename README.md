# AIST1110_Project
Project Name: Tank War

## TO-DO
- [x] Show time
- [x] Show score
- [ ] Play audio
  - [ ] Two shooting sound effects
  - [ ] Two moving sound effects
  - [ ] Explosion sound effects
  - [ ] Background music
- [x] Design better algorithm for the enemies' movement
- [x] Handle situations when different enemies touch one another\
  **The current handling method is to kill both enemies. It would be better to reverse their directions.**
- [x] Handle situations when enemies touch the walls
- [x] Handle situations when enemies are overlapped at spawn
- [x] Handle situations when enemies spwan at player's location
- [x] Decide the number of enemies, their speed and their shooting interval at any score under _score2enemy()
- [ ] Determine the maximum number of enemies $n$ and the maximum number of enemies' bullets $e$ -> size of observation space $=2 + 2n + 2e$
- [ ] Add sufficient comments
- [ ] Use Q-table or DQN to train a model

## Ideas
- [x] HP
- [ ] Difficulty
- [ ] Block
- [ ] Animations
- [ ] Multiple scenes
- [x] Display player's bullet reload time
