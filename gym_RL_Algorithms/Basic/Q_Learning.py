import numpy as np

class Qlearning():
    def __init__(self,
                 env,
                 state_num,
                 action_num,
                 lr,
                 gamma,
                 epsilon):

        self.env = env
        self.state_num = state_num
        self.action_num = action_num

        self.Q_table = np.zeros((state_num, action_num))

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        state_indices = np.round(state).astype(int)
        if np.random.rand(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_table[state_indices, :])
        return action

    def Update_Q_table(self, action, state, next_state, reward):
        state_indices = np.round(state).astype(int)
        next_state_indices = np.round(next_state).astype(int)
        self.Q_table[state_indices, action] = (1 - self.lr) * self.Q_table[state_indices, action] + \
                                 self.lr * (reward + self.gamma * np.max(self.Q_table[next_state_indices, :]))
