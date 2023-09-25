import numpy as np

class Sarsa():
    def __init__(self,
                 env,
                 state_num,
                 action_num,
                 learning_rate=0.8,
                 discount_factor=0.95,
                 epsilon=0.2):

        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.eps = epsilon

        self.state_num = state_num
        self.action_num = action_num
        self.Q_table = np.zeros((self.state_num, self.action_num))

    def get_action(self, state):
        state = np.round(state).astype(int)
        if np.random.uniform(0, 1) < self.eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state, :])
        action = np.clip(action, 0, 1)
        return action

    def Update_sarsa(self, state, action, reward, next_state, next_action):
        state = np.round(state).astype(int)
        next_state = np.round(next_state).astype(int)
        target = reward + self.gamma * self.Q_table[next_state, next_action]
        self.Q_table[state, action] = (1 - self.lr) * self.Q_table[state, action] + self.lr * target


