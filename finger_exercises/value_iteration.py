import numpy as np


class ValueIteration:
    def __init__(self, num_states, rewards, probabilities, gamma):
        self.states = num_states
        self.rewards = rewards
        self.transition = probabilities
        self.gamma = gamma
        self.values = np.empty(num_states)

    def value_iteration(self):
        term_a = np.sum(self.transition * self.rewards[None, :, None], axis=2)
        term_b = np.tensordot(self.transition, self.gamma * self.values, axes=(2, 0))
        self.values = np.max(term_a + term_b, axis=0)
        # self.values = np.max(np.tensordot(self.transition, self.rewards + self.gamma * self.values, axes=(2, 0)),
        #                      axis=0)
        print(self.values)

    def find_value(self, n, values=None):
        if values is None:
            self.values = np.zeros(self.states)
        else:
            self.values = values
        for i in range(n):
            print(f'------------Iteration {i + 1}------------')
            self.value_iteration()


if __name__ == '__main__':
    num_states = 5
    rewards = np.zeros(5)
    rewards[4] = 1
    transition = np.array([[[1 / 2, 1 / 2, 0, 0, 0], [1 / 4, 1 / 2, 1 / 4, 0, 0], [0, 1 / 4, 1 / 2, 1 / 4, 0],
                            [0, 0, 1 / 4, 1 / 2, 1 / 4], [0, 0, 0, 1 / 2, 1 / 2]],
                           [[1 / 2, 1 / 2, 0, 0, 0], [1 / 3, 2 / 3, 0, 0, 0], [0, 1 / 3, 2 / 3, 0, 0],
                            [0, 0, 1 / 3, 2 / 3, 0], [0, 0, 0, 1 / 3, 2 / 3]],
                           [[2 / 3, 1 / 3, 0, 0, 0], [0, 2 / 3, 1 / 3, 0, 0], [0, 0, 2 / 3, 1 / 3, 0],
                            [0, 0, 0, 2 / 3, 1 / 3], [0, 0, 0, 1 / 2, 1 / 2]
                            ]], dtype=float)
    gamma = 0.5
    values = ValueIteration(num_states, rewards, transition, gamma)
    values.find_value(100)
