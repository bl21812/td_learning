import argparse
import warnings
import logging

from datetime import datetime
from pathlib import Path

import numpy as np

from maze_env import Maze
from utils import plot_experiments, export_pickle

from RL_brainsample_wrong import rlalgorithm as WrongAlgo 
from RL_brainsample_sarsa import rlalgorithm as SARSAAlgo
from RL_brainsample_qlearning import rlalgorithm as QLearningAlgo
from RL_brainsample_expsarsa import rlalgorithm as ExpectedSARSAAlgo
from RL_brainsample_doubqlearning import rlalgorithm as DoubleQLearningAlgo


CURR_DATETIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RUNS_DIR = Path("./runs")
OUT_DIR = RUNS_DIR / f"{CURR_DATETIME}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger('ECE750')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(OUT_DIR / "experiment.log.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def update(env, RL, data, episodes, show_render, sim_speed, render_every_nth):
    wallbump = np.zeros(episodes)
    data['wallbump'] = wallbump

    pitfall = np.zeros(episodes)
    data['pitfall'] = pitfall

    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward

    ep_length = np.zeros(episodes)
    data['ep_length']=ep_length

    for episode in range(episodes):  
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

        logger.debug('state(ep:{},t:{})={}'.format(episode, t, state))

        if(show_render and (episode % render_every_nth)==0):
            logger.info(f'Rendering Now Alg:{RL.display_name} Ep:{episode}/{episodes} at speed:{sim_speed}')

        # The main loop of the training on an episode
        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            if(show_render and (episode % render_every_nth)==0):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            if reward == -0.3:
                wallbump[episode] += 1

            global_reward[episode] += reward

            logger.debug('state(ep:{},t:{})={}'.format(episode, t, state))
            logger.debug('reward={:.3f} return_t={:.3f} Mean50={:.3f}'.format(reward, global_reward[episode], np.mean(global_reward[-50:])))

            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1
        
        if reward == -10:
            pitfall[episode] += 1

        logger.info(f"({RL.display_name}) Ep {episode} Length={t} Summed Reward={global_reward[episode]:.3}, pitfall={pitfall[episode]}, wallbumps={wallbump[episode]}")

        #save data about length of the episode
        ep_length[episode] = t

    logger.info('Algorithm {} completed'.format(RL.display_name))
    env.destroy()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run RL algorithm on maze environment")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3], help="Task to use (1, 2, or 3)")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for epsilon-greedy policy")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--show_render", action="store_true", help="Render environment")
    parser.add_argument("--sim_speed", type=float, default=0.001, help="Simulation speed for rendering")
    parser.add_argument("--render_every_nth", type=int, default=10000, help="Render every Nth episode")
    parser.add_argument("--window", type=int, default=75, help="Window size for moving average")
    parser.add_argument("--plot", action="store_true", default=True, help="Plot things")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    hyperparam_kwargs = dict(
        epsilon=args.epsilon,
        alpha=args.alpha,
        gamma=args.gamma
    )

    logger.info(f"Experiment Setup: Episodes: {args.episodes}, Task: {args.task}, Window: {args.window}") 
    logger.info(f"Algorithm Hyperparameters: {hyperparam_kwargs}")
    logger.info(f"Show Render: {args.show_render}, Sim Speed: {args.sim_speed}, Render Every Nth: {args.render_every_nth}")

    warnings.filterwarnings("ignore", message="The frame\.append method is deprecated.*")

    # Task Specifications
    # point [0,0] is the top left cloggingorner
    # point [x,y] is x columns over and y rows down
    # range of x and y is [0,9]
    # agentXY=[0,0] # Agent start position [column, row]
    # goalXY=[4,4] # Target position, terminal state

    if args.task == 1:
        agentXY=[1,6] # Agent start position
        goalXY=[8,1] # Target position, terminal state
        wall_shape=np.array([[2,6], [2,5], [2,4], [6,1],[6,2],[6,3]])
        pits=np.array([[9,1],[8,3], [0,9]])
    if args.task == 2: # cliff face
        agentXY=[0,2] # Agent start position
        goalXY=[2,6] # Target position, terminal state
        wall_shape=np.array([ [0,3], [0,4], [0,5], [0,6], [0,7], [0,1],[1,1],[2,1],[8,7],[8,5],[8,3],[2,7]])
        pits=np.array([[1,3], [1,4], [1,5], [1,6], [1,7], [2,5],[8,6],[8,4],[8,2]])
    if args.task == 3:
        agentXY=[3,1] # Agent start position
        goalXY=[3,8] # Target position, terminal state
        wall_shape=np.array([[1,2],[1,3],[2,3],[4,3],[7,4],[3,6],[3,7],[2,7]])
        pits=np.array([[2,2],[3,4],[5,2],[0,5],[7,5],[0,6],[8,6],[0,7],[4,7],[2,8]])

    # First Demo Experiment 
    # Each combination of Algorithm and environment parameters constitutes an experiment that we will run for a number episodes, restarting the environment again each episode but keeping the value function learned so far.
    # You can add a new entry for each experiment in the experiments list and then they will all plot side-by-side at the end.

    experiments=[]

    # name1 = "WrongAlg on Task " + str(args.task)
    # env1 = Maze(agentXY,goalXY,wall_shape, pits, name1, args.show_render)
    # RL1 = WrongAlgo(actions=list(range(env1.n_actions)))
    # data1={}
    # env1.after(10, update(env1, RL1, data1, args.episodes, args.show_render, args.sim_speed, args.render_every_nth))
    # env1.mainloop()
    # experiments.append((name1, env1,RL1, data1))

    # SARSA
    name2 = "SARSA on Task " + str(args.task)
    env2 = Maze(agentXY,goalXY,wall_shape,pits, name2, args.show_render)
    RL2 = SARSAAlgo(actions=list(range(env2.n_actions)), **hyperparam_kwargs)
    data2={}
    env2.after(10, update(env2, RL2, data2, args.episodes, args.show_render, args.sim_speed, args.render_every_nth))
    env2.mainloop()
    experiments.append((name2, env2, RL2, data2))

    # Q-Learning
    name3 = "Q-Learning on Task " + str(args.task)
    env3 = Maze(agentXY,goalXY,wall_shape,pits, name3, args.show_render)
    RL3 = QLearningAlgo(actions=list(range(env3.n_actions)), **hyperparam_kwargs)
    data3={}
    env3.after(10, update(env3, RL3, data3, args.episodes, args.show_render, args.sim_speed, args.render_every_nth))
    env3.mainloop()
    experiments.append((name3, env3, RL3, data3))

    # Expected SARSA
    name4 = "Expected SARSA on Task " + str(args.task)
    env4 = Maze(agentXY,goalXY,wall_shape,pits, name4, args.show_render)
    RL4 = ExpectedSARSAAlgo(actions=list(range(env4.n_actions)), **hyperparam_kwargs)
    data4={}
    env4.after(10, update(env4, RL4, data4, args.episodes, args.show_render, args.sim_speed, args.render_every_nth))
    env4.mainloop()
    experiments.append((name4, env4, RL4, data4))

    # Double Q-Learning
    name5 = "Double Q-Learning on Task " + str(args.task)
    env5 = Maze(agentXY,goalXY,wall_shape,pits, name5, args.show_render)
    RL5 = DoubleQLearningAlgo(actions=list(range(env5.n_actions)), **hyperparam_kwargs)
    data5={}
    env5.after(10, update(env5, RL5, data5, args.episodes, args.show_render, args.sim_speed, args.render_every_nth))
    env5.mainloop()
    experiments.append((name5, env5, RL5, data5))

    logger.info("All experiments complete")

    for name, env, RL, data in experiments:
        logger.info("[{}] : {} : max(reward)={:.3f} median(last-{}-rewards)={:.3f} var(last-{}-rewards)={:.3f} max(episode-len)={}".format(
            name, 
            RL.display_name,
            np.max(data['global_reward']),
            args.window,
            np.median(data['global_reward'][-args.window:]), 
            args.window,
            np.var(data['global_reward'][-args.window:]),
            np.max(data['ep_length'])))


    if(args.plot):
        plot_experiments(experiments, "global_reward", "Summed Rewards", "Reward", window=args.window, out_dir=OUT_DIR)
        plot_experiments(experiments, 'ep_length', "Total Path Length", "Length", window=args.window, out_dir=OUT_DIR)
        plot_experiments(experiments, "pitfall", "Number of Falls into Pits", "Number of Falls", window=args.window, out_dir=OUT_DIR)
        plot_experiments(experiments, 'wallbump', "Number of Bumps into Walls", "Number of Bumps", window=args.window, out_dir=OUT_DIR)
    

    export_pickle(experiments, OUT_DIR)
    logger.info(f"Experiments data saved to {OUT_DIR}.")

    # TODO: METRICS / MEASUREMENTS TO ADD:
        # runtime, 'bad' moves (pit, wall, edge), repeated visits to spaces? (on any single path)

if __name__ == "__main__":
    main()
