import logging
import numpy as np

logger = logging.getLogger('ECE750')


class rlalgorithm:
    
    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.9, *args, **kwargs):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.display_name="Double Q-Learning"
        self.Q1={}
        self.Q2={}
        self.actions=actions
        self.num_actions = len(actions)
        logger.info(f'Init new {self.display_name} Algorithm: eps={epsilon} alpha={alpha} gamma={gamma}')

    def choose_action(self, observation, **kwargs):

        # add observed state as a key to value functions if not yet there
        self.check_state_exist(observation)

        # epsilon-greedy with respect to sum of value functions
        if np.random.uniform() > self.epsilon:
            q_sum = self.Q1[observation] + self.Q2[observation]
            action = self.actions[np.argmax(q_sum)]
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, **kwargs):

        # if any states havent been tracked yet, add as a key to value function
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        # Randomly choose a value function to update
        rand = np.random.uniform()
        if rand < 0.5:
            q_update = self.Q1
            q_static = self.Q2
        else:
            q_update = self.Q2
            q_static = self.Q1

        if s_ == 'terminal':
            logger.warning("Terminal state reached")
            # value functions for terminal states should always remain 0
                # so the max across all actions for a terminal state should also remain 0
            q_update[s][a] += self.alpha * (r - q_update[s][a])
        else:
            # update value function
            q_update[s][a] += self.alpha * (r + (self.gamma * q_static[s_][np.argmax(q_update[s_])]) - q_update[s][a])

        # choose next action with policy
        a_ = self.choose_action(s_)

        return s_, a_

    # same as in other algorithms, but checks / adds to both value functions
    def check_state_exist(self, state):
        if state not in self.Q1:
            self.Q1[state] = np.zeros(self.num_actions)
        if state not in self.Q2:
            self.Q2[state] = np.zeros(self.num_actions)
