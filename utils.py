# Code from:
# https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html

from gymnasium.wrappers.monitoring import video_recorder
import numpy as np
import matplotlib.pyplot as plt

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

# show_video_of_model(agent, 'LunarLander-v2')

# show_video('LunarLander-v2')