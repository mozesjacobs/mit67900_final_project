# ChatGPT Code

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data from the table
gravity = [-5, -8, -10, -11]
ddqn = [206.43, (200.58, 187.62), (134.13, 143.40), (92.09, 119.61)]
fittedq = [177.16, (219.15, 33.79), (192.99, 115.15), (161.53, 241.84)]
trpo = [119.20, (68.01, 129.41), (25.64, 158.95), (8.13, 64.01)]
ppo = [235.69, (213.85, 237.14), (207.61, 193.66), (178.51, 214.86)]

# Function to extract "before" and "after" values
def extract_values(data):
    values = []
    for item in data:
        if isinstance(item, tuple):
            values.append(item)
        else:
            values.append((item, 0))  # If there's only one value, set the second value to 0
    return values

# Extract values for each algorithm
ddqn_values = extract_values(ddqn)
fittedq_values = extract_values(fittedq)
trpo_values = extract_values(trpo)
ppo_values = extract_values(ppo)

# Plotting
sns.set()

bar_width = 0.2
index = np.arange(len(gravity))

fig, ax = plt.subplots(figsize=(10, 10))

bar1 = ax.bar(index - 1.5 * bar_width, [item[0] for item in ddqn_values], bar_width, label='DDQN Before', color='b')
bar2 = ax.bar(index - 1.5 * bar_width, [item[1] for item in ddqn_values], bar_width, label='DDQN After', color='g')

bar3 = ax.bar(index - 0.5 * bar_width, [item[0] for item in fittedq_values], bar_width, label='FittedQ Before', color='c')
bar4 = ax.bar(index - 0.5 * bar_width, [item[1] for item in fittedq_values], bar_width, label='FittedQ After', color='m')

bar5 = ax.bar(index + 0.5 * bar_width, [item[0] for item in trpo_values], bar_width, label='TRPO Before', color='y')
bar6 = ax.bar(index + 0.5 * bar_width, [item[1] for item in trpo_values], bar_width, label='TRPO After', color='r')

bar7 = ax.bar(index + 1.5 * bar_width, [item[0] for item in ppo_values], bar_width, label='PPO Before', color='orange')
bar8 = ax.bar(index + 1.5 * bar_width, [item[1] for item in ppo_values], bar_width, label='PPO After', color='purple')

# Add labels, title, and legend
ax.set_xlabel('Gravity')
ax.set_ylabel('Performance')
ax.set_title('Performance Comparison for Different Algorithms')
ax.set_xticks(index)
ax.set_xticklabels(gravity)
ax.legend(loc='upper left')

# Show the plot
plt.savefig("gravity.png")
plt.show()