# AIST1110_Project
###Project Name:
Tank War

###Group Member: 

LAM, Yiu Fung Anson

FONG, Shi Yuk


## Steps
0. `[...]` means optional argument. 
`[...|...]` means optional argument, and choose one of the two arguments only.    
1. Open your own conda environment.
2. Execute the following commands in the Terminal:
```
cd TankWar/gym-tankwar 
conda create --name tankwar python=3.10 -y 
conda activate tankwar 
pip install -e .
```  

3. Execute the following commands in the Terminal:
```
cd ../tankwar
```     
4.  To play the game, execute:
```
python tankwar_play.py -m human [-d DIFFICULTY] [-e EPISODES]
```
   To visualize the gameplay GUI in a read-only human mode, execute:
```
python tankwar_play.py -m human_rand [-d DIFFICULTY] [-e EPISODES]
```
   To expose the game in a non-GUI mode, execute:
```
python tankwar_play.py [-d DIFFICULTY] [-e EPISODES]
```


(`DIFFICULTY`: Game mode. 0 for easy mode, 1 for hard mode. Default: 0)

(`EPISODES`:Number of games to be played. Default: 1000)

(Check `README.txt` or execute:

```
python tankwar_play.py -h
```

to get all information of the command line arguments.)

5. Before training and testing the agent, execute:
```
pip install tensorflow==2.10.0
```
6. To train the agent, execute:
```
python tankwar_train.py -s SEED -d DIFFICULTY [-traine TRAIN_EPISODES | -fast]
```
(`TRAIN_EPISODES`:The number of training episodes. Default: 1000. Suggested value: 300)

(`-fast`: train the model in fast mode. Training finishes in around 20 minutes.)

7. To test the model, execute:
```
python tankwar_test.py -f FILE [-d DIFFICULTY] [-teste TEST_EPISODE]
```
(`SEED`: Seed for the random generator.)
(`TEST_EPISODES`:The number of training episodes. Default: 100.)
(`FILE`: model file name. 4 sample model is given in `/models` folder. Note that `DIFFICULTY` 
should match the specification given by the sample file name.)