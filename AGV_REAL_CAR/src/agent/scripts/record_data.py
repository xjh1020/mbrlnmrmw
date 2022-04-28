def record_poistion_step_reward(p_x, p_y, step_reward):
    data = "{:.3f},{:.3f},{:.3f}\n".format(p_x, p_y, step_reward)
    with open("./record/poistion_step_record.txt", "a") as f:
        f.write(data)


def record_episode_loss(episode, ep_loss):
    data = "{:d},{:.4f}\n".format(episode, ep_loss)
    with open("./record/episode_loss.txt", "a") as f:
        f.write(data)

def record_episode_reward(episode, ep_reward):
    data = "{:d},{:.4f}\n".format(episode, ep_reward)
    with open("./record/episode_reward.txt", "a") as f:
        f.write(data)





if __name__ == "__main__":
    p_x = 0.3
    p_y = 0.4
    step_reward = 0.001
    
    for i in range(5):
        record_poistion_step_reward(p_x, p_y, step_reward)
        record_episode_loss(i, 10)
        