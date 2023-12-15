import gymnasium as gym
from gymnasium.wrappers.monitoring import video_recorder
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import subprocess
import pdf2image

class CustomCallback(BaseCallback):
    def __init__(self, model, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.positive_rewards_count = []
        self.env = gym.make('CustomEnv')
        self.model = model
    def _on_step(self) -> bool:

        return True  # Continue training

    def _on_rollout_end(self) -> None:

        # Access the total reward for the current rollout
        rewards, pos_rewards = compute_average_reward(self.model, self.env, 15)

        # Store the information
        self.episode_rewards.append(rewards)
        self.positive_rewards_count.append(pos_rewards)

    def get_info(self):
        return {'episode_rewards': self.episode_rewards, 'positive_rewards_count': self.positive_rewards_count}

def show_figure():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def show_video(env_name):
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = 'video/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
        
def show_video_of_model(agent, env_name):
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(env, path="video/{}.mp4".format(env_name))
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        vid.capture_frame()
        
        action = agent.act(state)

        state, reward, done, _ = env.step(action)        
    env.close()

def compute_average_reward(model, env, num_runs=100, nb_pos_r=True):
    """
    Compute average reward of the model over `num_runs` runs.
    If `nb_pos_r`, also compute the average number of times the
    agent received positive rewards per run.
    """
    all_rewards = 0
    all_pos_rewards = 0

    for _ in range(num_runs):
        obs, _ = env.reset()
        total_reward = 0
        pos_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if reward > 0:
                pos_reward += 1

            if terminated or truncated:
                break

        all_rewards += total_reward
        all_pos_rewards += pos_reward

    all_rewards /= num_runs
    all_pos_rewards /= num_runs
    if nb_pos_r:
        return all_rewards, all_pos_rewards
    return all_rewards

def add_enemies(env):
    for i in [(2, 7), (2, 19)]:
        env.board[i[0]][i[1]].state = 'E'
        env.enemies.append(env.board[i[0]][i[1]])

# show_video_of_model(agent, 'LunarLander-v2')

# show_video('LunarLander-v2')

class Visual():
    '''
    Class for visualization.
    write_file: writes a latex file with a frame per element in the argument liste.
    visualize: writes the latex document into a pdf.
    '''
    def __init__(self,liste):
        self.liste=liste
        self.viewer="open"

    def write_file(self):
        #subprocess.Popen(["touch","visualisierung.tex"])

        with open('visualisierung.tex','w') as v:
            print('\\documentclass{beamer}', file=v)
            print('\\usepackage{adjustbox}', file=v)
            print('\\usepackage{graphicx}', file=v)

            print('',file=v)
            print('\\begin{document}', file=v)


            for x in self.liste:
                print('\\begin{frame}', file=v)
                print(x,file=v)
                print('\\end{frame}',file=v)
                print('',file=v)
            print('\\end{document}',file=v)

    def set_viewer(self,viewername):
        self.viewer=viewername


    def visualize(self):
        self.write_file()
        subprocess.Popen(["pdflatex","visualisierung.tex"])

    def create_gif(self):
        images=pdf2image.convert_from_path('visualisierung.pdf',fmt='png')
        images[0].save('visualization.gif',save_all=True,append_images=images[1:],duration=150,loop=0)

def smooth_list(input_list, k):
    """
    Smooths a list by taking the average of the nearest k entries for each element.

    Parameters:
    - input_list: The input list to be smoothed.
    - k: The number of nearest neighbors to include in the average.

    Returns:
    - A new list of the same length as the input, with each element being the average
      over the nearest k elements in the original list.
    """
    smoothed_list = []
    n = len(input_list)

    for i in range(n):
        start_index = max(0, i - k)
        end_index = min(n, i + k + 1)
        neighbors = input_list[start_index:end_index]
        smoothed_list.append(sum(neighbors) / len(neighbors))

    return smoothed_list

def plot_lines_with_timesteps(*args, title="Line Plot", xlabel="Number of Timesteps", ylabel="Y-axis"):
    """
    Creates a line plot of multiple input lists with corresponding timesteps.

    Parameters:
    - *args: Variable number of input lists, each followed by its timestep and label.
    - title: The title of the plot (default: "Line Plot").
    - xlabel: The label for the x-axis (default: "Number of Timesteps").
    - ylabel: The label for the y-axis (default: "Y-axis").
    """
    if len(args) % 3 != 0:
        raise ValueError("Each input list must be followed by its timestep and label.")

    for i in range(0, len(args), 3):
        input_list = args[i]
        timestep = args[i + 1]
        label = args[i + 2]

        # Create x-axis values based on timesteps
        x_values = [timestep * i/(len(input_list)-1) for i in range(len(input_list))]
        plt.plot(x_values, input_list, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def create_column_chart(column_data, column_labels, title="Column Chart", xlabel="Columns", ylabel="Values", legend_labels=None, column_colors=None):
    """
    Creates a column chart with different colors for each set of columns.

    Parameters:
    - column_data: List of scalars or lists of scalars corresponding to the height of each column.
    - column_labels: List of labels for each column.
    - title: The title of the chart (default: "Column Chart").
    - xlabel: The label for the x-axis (default: "Columns").
    - ylabel: The label for the y-axis (default: "Values").
    - legend_labels: List of labels for the legend (default: None).
    - column_colors: List of colors for the columns (default: None).
    """
    fig, ax = plt.subplots()

    if all(isinstance(item, list) for item in column_data):
        # Multiple columns for each label
        num_columns = len(column_data[0])
        bar_width = 0.2
        index = np.arange(len(column_labels))

        if column_colors is None:
            # Generate a list of different colors if colors not provided
            colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, num_columns)]
        else:
            colors = column_colors

        for i in range(num_columns):
            values = [item[i] for item in column_data]
            label = f"Column {i+1}" if legend_labels is None else legend_labels[i]
            color = colors[i]
            ax.bar(index + i * bar_width, values, bar_width, label=label, color=color)

        ax.set_xticks(index + bar_width * (num_columns - 1) / 2)
        ax.set_xticklabels(column_labels)
    else:
        # Single column for each label
        if column_colors is None:
            # Generate a list of different colors if colors not provided
            colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(column_labels))]
        else:
            colors = column_colors

        for i, (label, value) in enumerate(zip(column_labels, column_data)):
            color = colors[i]
            ax.bar(label, value, color=color, label=legend_labels[i] if legend_labels else None)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_labels:
        ax.legend(title="Column Legend", loc="upper right")
    plt.show()