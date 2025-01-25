# analyze states and actions

import numpy as np
import matplotlib.pyplot as plt

data = np.load('states_actions.npz')
states = data['states']
actions = data['actions']

duration = 60
dt = 0.05

# plot actions (x-axis is time, y-axis is action)
for i in range(actions.shape[0]):
    plt.plot(np.arange(0, duration, dt), actions[i, :, 0])
    plt.title(f'Action for pendulum {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Action')
plt.show()
