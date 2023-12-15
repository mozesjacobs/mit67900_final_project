# ChatGPT Code

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data from the table
wind = [0, 5, 10, 15]
ddqn = [229.10, (127.74, 222.54), (94.91, 226.52), (32.20, 216.53)]
fittedq = [95.68, (136.68, -28.23), (102.98, -69.67), (68.44, 111.46)]
trpo = [86.68, (90.86, 187.03), (86.15, 152.07), (45.45, 70.15)]
ppo = [66.16, (20.65, 37.33), (-10.88, 66.19), (-34.55, -9.19)]

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
index = np.arange(len(wind))

fig, ax = plt.subplots(figsize=(10, 10))

bar1 = ax.bar(index - 1.5 * bar_width, [item[0] for item in ddqn_values], bar_width, label='DDQN Before', color='b')
#bar2 = ax.bar(index - 1.5 * bar_width, [item[1] for item in ddqn_values], bar_width, bottom=[item[0] for item in ddqn_values], label='DDQN After', color='g')
bar2 = ax.bar(index - 1.5 * bar_width, [item[1] for item in ddqn_values], bar_width, label='DDQN After', color='g')

bar3 = ax.bar(index - 0.5 * bar_width, [item[0] for item in fittedq_values], bar_width, label='FittedQ Before', color='c')
#bar4 = ax.bar(index - 0.5 * bar_width, [item[1] for item in fittedq_values], bar_width, bottom=[item[0] for item in fittedq_values], label='FittedQ After', color='m')
bar4 = ax.bar(index - 0.5 * bar_width, [item[1] for item in fittedq_values], bar_width, label='FittedQ After', color='m')

bar5 = ax.bar(index + 0.5 * bar_width, [item[0] for item in trpo_values], bar_width, label='TRPO Before', color='y')
#bar6 = ax.bar(index + 0.5 * bar_width, [item[1] for item in trpo_values], bar_width, bottom=[item[0] for item in trpo_values], label='TRPO After', color='r')
bar6 = ax.bar(index + 0.5 * bar_width, [item[1] for item in trpo_values], bar_width, label='TRPO After', color='r')

bar7 = ax.bar(index + 1.5 * bar_width, [item[0] for item in ppo_values], bar_width, label='PPO Before', color='orange')
#bar8 = ax.bar(index + 1.5 * bar_width, [item[1] for item in ppo_values], bar_width, bottom=[item[0] for item in ppo_values], label='PPO After', color='purple')
bar8 = ax.bar(index + 1.5 * bar_width, [item[1] for item in ppo_values], bar_width, label='PPO After', color='purple')

# Add labels, title, and legend
ax.set_xlabel('Gravity')
ax.set_ylabel('Performance')
ax.set_title('Performance Comparison for Different Algorithms')
ax.set_xticks(index)
ax.set_xticklabels(wind)
ax.legend(loc='upper left')

# Show the plot
plt.savefig("wind.png")
plt.show()