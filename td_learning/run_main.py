import os
import sys
import pickle
import warnings
import logging

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from maze_env import Maze
from plot_utils import plot_rewards, plot_length

from RL_brainsample_wrong import rlalgorithm as WrongAlgo 
from RL_brainsample_sarsa import rlalgorithm as SARSAAlgo
from RL_brainsample_qlearning import rlalgorithm as QLearningAlgo
from RL_brainsample_expsarsa import rlalgorithm as ExpectedSARSAAlgo
from RL_brainsample_doubqlearning import rlalgorithm as DoubleQLearningAlgo


DEBUG = 1

def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)

#SIM_SPEED of .1 is nice to view, .001 is fast but visible, SIM_SPEED has not effect if SHOW_RENDER is False
SIM_SPEED = 0.001 #.001

#Which task to run, select just one
USE_TASK = 1 # 1,2,3

#Example Short Fast start parameters for Debugging
# EPISODES = 2000 #100, 500, 1000
EPISODES = 10 #100, 500, 1000
RENDER_EVERY_NTH = 10000 #10, 100, 250 
PRINT_EVERY_NTH = 100 #10, 25, 100
WINDOW = 250 #10, 25

DO_PLOT_REWARDS=True
DO_PLOT_LENGTH=True

# True means RENDER_EVERY_NTH episode only, False means don't render at all
SHOW_RENDER = False 


HYPERPARAM_KWARGS = dict(
    epsilon = 0.1,
    alpha = 0.1,
    gamma = 0.9
)

# TODO(howird): yea this is insane and should be removed, im just tired and want it to run
if __name__ == "__main__" and len(sys.argv) > 1:
    USE_TASK = int(sys.argv[1])
    if len(sys.argv) > 2:
        HYPERPARAM_KWARGS["alpha"] = float(sys.argv[2])


CURR_DATETIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RUNS_DIR = Path("./runs")
OUT_DIR = RUNS_DIR / f"{CURR_DATETIME}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = OUT_DIR / f"experiment.log.txt"

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

logging.info(f"Number of Episodes: {EPISODES}, Render Every Nth: {RENDER_EVERY_NTH}, Print Every Nth: {PRINT_EVERY_NTH}, Window: {WINDOW}")
logging.info(f"Algorithm Hyperparameters: {HYPERPARAM_KWARGS}")

# Example Full Run, you may need to run longer
#SHOW_RENDER=False
#EPISODES=1000
#RENDER_EVERY_NTH=100
#PRINT_EVERY_NTH=20
#WINDOW=100
#DO_PLOT_REWARDS=True
#DO_PLOT_LENGTH=True


def update(env, RL, data):
    global_reward = np.zeros(EPISODES)
    data['global_reward']=global_reward
    ep_length = np.zeros(EPISODES)
    data['ep_length']=ep_length
    if EPISODES >= WINDOW:
        med_rew_window = np.zeros(EPISODES-WINDOW)
        var_rew_window = np.zeros(EPISODES)
    else:
        med_rew_window = []
        var_rew_window = []
    data['med_rew_window'] = med_rew_window
    data['var_rew_window'] = var_rew_window

    for episode in range(EPISODES):  
        t=0
        ''' initial state
            Note: the state is represented as two pairs of 
            coordinates, for the bottom left corner and the 
            top right corner of the agent square.
        '''
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        logging.debug('state(ep:{},t:{})={}'.format(episode, t, state))

        if(SHOW_RENDER and (episode % RENDER_EVERY_NTH)==0):
            logging.info(f'Rendering Now Alg:{RL.display_name} Ep:{episode}/{EPISODES} at speed:{SIM_SPEED}')

        # The main loop of the training on an episode
        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            if(SHOW_RENDER and (episode % RENDER_EVERY_NTH)==0):
                env.render(SIM_SPEED)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            logging.debug('state(ep:{},t:{})={}'.format(episode, t, state))
            logging.debug('reward={:.3f} return_t={:.3f} Mean50={:.3f}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))

            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        logging.info(f"({RL.display_name}) Ep {episode} Length={t} Summed Reward={global_reward[episode]:.3}")

        #save data about length of the episode
        ep_length[episode]=t

        if(episode>=WINDOW):
            med_rew_window[episode-WINDOW] = np.median(global_reward[episode-WINDOW:episode])
            var_rew_window[episode-WINDOW] = np.var(global_reward[episode-WINDOW:episode])
            logging.debug("    Med-{}={:.3f} Var-{}={:.3f}".format(
                    WINDOW,
                    med_rew_window[episode-WINDOW],
                    WINDOW,
                    var_rew_window[episode-WINDOW]))
    logging.info('Algorithm {} completed'.format(RL.display_name))
    env.destroy()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The frame\.append method is deprecated.*")

    # Task Specifications
    # point [0,0] is the top left corner
    # point [x,y] is x columns over and y rows down
    # range of x and y is [0,9]
    # agentXY=[0,0] # Agent start position [column, row]
    # goalXY=[4,4] # Target position, terminal state

    if USE_TASK == 1:
        agentXY=[1,6] # Agent start position
        goalXY=[8,1] # Target position, terminal state
        wall_shape=np.array([[2,6], [2,5], [2,4], [6,1],[6,2],[6,3]])
        pits=np.array([[9,1],[8,3], [0,9]])
    if USE_TASK == 2: # cliff face
        agentXY=[0,2] # Agent start position
        goalXY=[2,6] # Target position, terminal state
        wall_shape=np.array([ [0,3], [0,4], [0,5], [0,6], [0,7], [0,1],[1,1],[2,1],[8,7],[8,5],[8,3],[2,7]])
        pits=np.array([[1,3], [1,4], [1,5], [1,6], [1,7], [2,5],[8,6],[8,4],[8,2]])
    if USE_TASK == 3:
        agentXY=[3,1] # Agent start position
        goalXY=[3,8] # Target position, terminal state
        wall_shape=np.array([[1,2],[1,3],[2,3],[4,3],[7,4],[3,6],[3,7],[2,7]])
        pits=np.array([[2,2],[3,4],[5,2],[0,5],[7,5],[0,6],[8,6],[0,7],[4,7],[2,8]])

    # First Demo Experiment 
    # Each combination of Algorithm and environment parameters constitutes an experiment that we will run for a number episodes, restarting the environment again each episode but keeping the value function learned so far.
    # You can add a new entry for each experiment in the experiments list and then they will all plot side-by-side at the end.

    experiments=[]

    # name1 = "WrongAlg on Task " + str(USE_TASK)
    # env1 = Maze(agentXY,goalXY,wall_shape, pits, name1, SHOW_RENDER)
    # RL1 = WrongAlgo(actions=list(range(env1.n_actions)))
    # data1={}
    # env1.after(10, update(env1, RL1, data1))
    # env1.mainloop()
    # experiments.append((name1, env1,RL1, data1))

    # SARSA
    name2 = "SARSA on Task " + str(USE_TASK)
    env2 = Maze(agentXY,goalXY,wall_shape,pits, name2, SHOW_RENDER)
    RL2 = SARSAAlgo(actions=list(range(env2.n_actions)), **HYPERPARAM_KWARGS)
    data2={}
    env2.after(10, update(env2, RL2, data2))
    env2.mainloop()
    experiments.append((name2, env2, RL2, data2))

    # Q-Learning
    name3 = "Q-Learning on Task " + str(USE_TASK)
    env3 = Maze(agentXY,goalXY,wall_shape,pits, name3, SHOW_RENDER)
    RL3 = QLearningAlgo(actions=list(range(env3.n_actions)), **HYPERPARAM_KWARGS)
    data3={}
    env3.after(10, update(env3, RL3, data3))
    env3.mainloop()
    experiments.append((name3, env3, RL3, data3))

    # Expected SARSA
    name4 = "Expected SARSA on Task " + str(USE_TASK)
    env4 = Maze(agentXY,goalXY,wall_shape,pits, name4, SHOW_RENDER)
    RL4 = ExpectedSARSAAlgo(actions=list(range(env4.n_actions)), **HYPERPARAM_KWARGS)
    data4={}
    env4.after(10, update(env4, RL4, data4))
    env4.mainloop()
    experiments.append((name4, env4, RL4, data4))

    # Double Q-Learning
    name5 = "Double Q-Learning on Task " + str(USE_TASK)
    env5 = Maze(agentXY,goalXY,wall_shape,pits, name5, SHOW_RENDER)
    RL5 = DoubleQLearningAlgo(actions=list(range(env5.n_actions)), **HYPERPARAM_KWARGS)
    data5={}
    env5.after(10, update(env5, RL5, data5))
    env5.mainloop()
    experiments.append((name5, env5, RL5, data5))

    print("All experiments complete")

    # print(f"Experiment Setup:\n - episodes:{EPISODES} VI_sweeps:{VI_sweeps} sim speed:{SIM_SPEED}") 
    print(f"Experiment Setup:\n - episodes:{EPISODES}\n - sim speed:{SIM_SPEED}\n") 

    for name, env, RL, data in experiments:
        logging.info("[{}] : {} : max-rew={:.3f} med-{}={:.3f} var-{}={:.3f} max-episode-len={}".format(
            name, 
            RL.display_name, 
            np.max(data['global_reward']),
            WINDOW,
            np.median(data['global_reward'][-WINDOW:]), 
            WINDOW,
            np.var(data['global_reward'][-WINDOW:]),
            np.max(data['ep_length'])))

    if(DO_PLOT_REWARDS):
        #Simple plot of summed reward for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments, WINDOW, OUT_DIR)

    if(DO_PLOT_LENGTH):
        #Simple plot of summed reward for each episode and algorithm, you can make more informative plots
        plot_length(experiments, OUT_DIR)
    

    with open(OUT_DIR / "experiments.pkl", 'wb') as f:
        exps = [(name, RL.display_name, data) for name, env, RL, data in experiments]
        pickle.dump(exps, f)
        logging.info(f"Experiments data saved to {OUT_DIR / f'experiments.pkl'}")


    # TODO: METRICS / MEASUREMENTS TO ADD:
        # runtime, 'bad' moves (pit, wall, edge), repeated visits to spaces? (on any single path)
