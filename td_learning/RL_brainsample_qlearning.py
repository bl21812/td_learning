
import numpy as np
import pandas as pd


class rlalgorithm:

    # TODO: what should default values of alpha and gamma be ??
    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.1, *args, **kwargs):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.display_name="Q-Learning"
        self.Q={}
        self.actions=actions
        self.num_actions = len(actions)

    # mostly same as sample code (except his epsilon check was weird?)
    def choose_action(self, observation, **kwargs):

        # add observed state as a key to value function if not yet there
        self.check_state_exist(observation)

        # epsilon-greedy
        if np.random.uniform() > self.epsilon:
            action = self.actions[np.argmax(self.Q[observation])]
        else:
            action = np.random.choice(self.actions)

        return action


    def learn(self, s, a, r, s_, **kwargs):
        
        # if any states havent been tracked yet, add as a key to value function
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # TODO: im not sure how s_ can be a string, i thought it was coords ...
        if s_ == 'terminal':
            # value functions for terminal states should always remain 0
                # so the max across all actions for a terminal state should also remain 0
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])
        else:
            # update value function
            self.Q[s][a] += self.alpha * (r + (self.gamma * self.Q[s_][np.argmax(self.Q[s_])])- self.Q[s][a])

        # choose next action with policy
        a_ = self.choose_action(s_)

        return s_, a_

    # just taken from his sample code
    def check_state_exist(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
