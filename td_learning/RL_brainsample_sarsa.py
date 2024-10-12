import logging
import numpy as np

logger = logging.getLogger('ECE750')


class rlalgorithm:

    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.9, *args, **kwargs):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.display_name="SARSA"
        self.Q={}
        self.actions=actions
        self.num_actions = len(actions)
        logger.info(f'Init new {self.display_name} Algorithm: eps={epsilon} alpha={alpha} gamma={gamma}')

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

        if s_ == 'terminal':
            logger.warning("Terminal state reached")
            # we can skip choosing an action - since the episode will terminate
                # and a_ is only needed for Q[s_][a_], which for terminal s_ should always be 0
            a_ = None
            # update value function (note that all Q values for terminal states = 0)
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])

        else:
            # choose a_
            a_ = self.choose_action(s_)
            # update value function
            self.Q[s][a] += self.alpha * (r + (self.gamma * self.Q[s_][a_]) - self.Q[s][a])

        return s_, a_

    # just taken from his sample code
    def check_state_exist(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
