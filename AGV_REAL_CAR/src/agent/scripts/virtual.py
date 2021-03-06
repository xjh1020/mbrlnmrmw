import os
import sys
pathx = os.path.abspath(".")
sys.path.insert(0,pathx + "/src/agent/scripts")

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from robot_env import *
import random
import os
import matplotlib.pyplot as plt
from record_data import *
from SE import *
import gpytorch
import load_dataset as ld
from quaternions import Quaternion as Quaternion
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import random

SEED_NUM = 666

MEMORY_CAPACITY = 2000
TARGET_NET_INTERATION = 250
BATCH_SIZE = 64
GAMMA = 0.9
MAX_PER_EPIOSDE_STEPS = 300
START_TO_LEARN = 3000

moveBindings = {
            '0' : (1., 0, 0),
            '1' : (-1., 0, 0), 
            '2' : (0, 1., 0), 
            '3' : (0, -1., 0),
            '4': (0, 0, 1.),
            '5': (0, 0, -1.),
}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return

def save_checkpoint(evaule_model, target_model, optimizer, epoch):
    print('Model Saving...')
    evaule_model_state_dict = evaule_model.state_dict()
    target_model_state_dict = target_model.state_dict()
    torch.save({
        'evaule_model_state_dict': evaule_model_state_dict,
        'target_model_state_dict': target_model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('checkpoints', str(epoch) + '_checkpoint.pth'))
    return


def _init_weights(m):
    """
    weight initialization
    :param m: module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.ones_(m.weight)
    return

def image_transpose(obs):
    obs = torch.FloatTensor(obs)
    obs = torch.transpose(obs, 0, 2)
    obs = torch.transpose(obs, 1, 2).numpy()
    return obs

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)




# SumTree
# a binary tree data structure where the parent???s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory_Buffer_PER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, memory_size=1000, a = 0.6, e = 0.01):
        self.tree =  SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.e = e
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        return idxs, np.concatenate(states), actions, rewards, np.concatenate(next_states), dones
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def size(self):
        return self.tree.n_entries




class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, (5, 5))
        self.conv2 = torch.nn.Conv2d(32, 128, (5, 5))
        self.conv3 = torch.nn.Conv2d(128 , 64, (3, 3))
        self.maxpool = torch.nn.MaxPool2d((4, 4))
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(64*4*7, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, 128)
        self.linear5 = torch.nn.Linear(128, 64)
        self.linear6 = torch.nn.Linear(64, 32)
        
        self.value_layer = torch.nn.Linear(32, 1)
        self.advantage_layer = torch.nn.Linear(32, 6)
        
        def _init_weights(m):
            """
            weight initialization
            :param m: module
            """
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)
            return
        
        _init_weights(self.conv1)
        _init_weights(self.conv2)
        _init_weights(self.conv3)
        
        _init_weights(self.linear1)
        _init_weights(self.linear2)
        _init_weights(self.linear3)
        _init_weights(self.linear4)
        _init_weights(self.linear5)
        _init_weights(self.linear6)
        _init_weights(self.value_layer)
        _init_weights(self.advantage_layer)
        
        return
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 64*4*7)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.relu(self.linear6(x))
        
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        # print("value.shape: ", value.shape)  # torch.Size([1, 1])
        # print("advantage.shape: ", advantage.shape)  #  torch.Size([1, 6])
        
        Q = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        # print("Q.shape: ", Q.shape)
        
        return Q
        
    

class DQN(object):
    def __init__(self, model, obs_space, act_space):
        self.evalue_net, self.target_net = model, model

        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.evalue_net.parameters(), lr=0.0001)

        self.obs_space = obs_space
        self.act_space = act_space

        self.epsilon = 0.1
        self.is_explore = False
        self.learn_step = 0
        self.loss= 0.
        
        self.memory_buffer = Memory_Buffer_PER(memory_size=MEMORY_CAPACITY, a=0.6, e=0.001)

        return

    def predict(self, obs):
        action_value = self.evalue_net(obs)
        # print(action_value.detach())
        _, action = torch.max(action_value, dim=1)
        return action.item()  # int
    
    def sample(self, obs, flag=False):
        obs = torch.unsqueeze(torch.FloatTensor(obs), dim=0)
        if np.random.uniform(0, 1) < (1 - self.epsilon):
            if flag:
                self.epsilon -= self.epsilon / 10000
            self.is_explore = False
            action = self.predict(obs.cuda())
        else:
            self.is_explore = True
            action = np.random.randint(0, self.act_space)
        return action  # int
    
    
    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        segment = self.memory_buffer.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory_buffer.tree.get(s)
            
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            priorities.append(p)
            idxs.append(idx)
        # return idxs, torch.cat(states), actions, rewards, torch.cat(next_states), dones
        return idxs, states, actions, rewards, next_states, dones
    
    def learn(self, a, lock, mylist):
        if self.learn_step % TARGET_NET_INTERATION == 0:
            self.target_net.load_state_dict(self.evalue_net.state_dict())
        self.learn_step += 1

        idxs, states, actions, rewards, next_states, dones = self.sample_from_buffer(BATCH_SIZE)
        
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        actions = np.array(actions)
        
        obs = torch.FloatTensor(states).view(-1, 3, 314, 512).cuda()
        next_obs = torch.FloatTensor(next_states).view(-1, 3, 314, 512).cuda()
        reward = torch.FloatTensor(rewards).view(-1, 1).cuda()
        done = torch.FloatTensor(dones).view(-1, 1).cuda()
        act = torch.LongTensor(actions).view(-1, 1).cuda()
        
        if a >= 0.75:
            lock.acquire()
            if len(mylist) >= 10 * BATCH_SIZE:
                mylist = []
            temp = []
            for i in range(BATCH_SIZE):
                temp.append(obs[i].cpu())
                temp.append(act[i].cpu())
                temp.append(reward[i].cpu())
                temp.append(next_obs[i].cpu())
                temp.append(done[i].cpu())
                mylist.append(temp)
            lock.release()

        q_evalue = self.evalue_net(obs).gather(1, act).view(-1, 1)  # torch.Size([64, 1])

        _, act_max = torch.max(self.evalue_net.forward(next_obs).detach(), dim=1)
        act_max = act_max.view(-1, 1)
        q_next = self.target_net.forward(next_obs).gather(1, act_max).detach()
        q_target = reward + GAMMA * q_next * (1 - done)
        
        
        errors = (q_evalue - q_target).detach().cpu().squeeze().tolist()
        self.memory_buffer.update(idxs, errors)

        loss = self.criterion(q_evalue, q_target)
        self.loss = loss.detach().cpu().item()
        # print("loss: %.2f" % loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(self.evalue_net.parameters(), max_norm=1, norm_type=2)
        self.optim.step()
        self.optim.zero_grad()

        return

def getPose(lock, myPose):
    data = []
    lock.acquire()
    n = len(myPose)
    if n > 100:
        for l in myPose:
            data.append(l)
    lock.release()
    return data

def actToVel(act):
    v_x = 0
    v_y = 0
    w = 0

    key = str(act)
    if key in moveBindings.keys():
        v_x = moveBindings[key][0]
        v_y = moveBindings[key][1]
        w = moveBindings[key][2]

    else:
        v_x = 0
        v_y = 0
        w = 0
    return v_x, v_y, w
    
def agv_run(lock, mylist, myAcc, myPose):
    setup_seed(SEED_NUM)

    env = RobotEnv()
    if not env.initRobotEnv():
        return
    else:
        print("RobotEnv init success!")
    
    obs_space = env.observation_space
    act_space = env.action_dim
    
    # model = Model()
    model = resnet18()

    dqn = DQN(model.cuda(), obs_space, act_space)


    flag = False
    checkpoints = None
    if checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', checkpoints))
        dqn.evalue_net.load_state_dict(checkpoints['evaule_model_state_dict'])
        dqn.target_net.load_state_dict(checkpoints['target_model_state_dict'])
        dqn.optim.load_state_dict(checkpoints['optimizer_state_dict'])
    
    total_episodes = 1200
    ep_r_list = []
    ep_l_lsit = []
    dqn.epsilon = 0.
    for episode in range(total_episodes):
        data = getPose(lock, myPose)
        train_x, train_y = ld.generate_train_dataset(data)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        model = MultitaskGPModel(train_x, train_y, likelihood)
        # this is for running the notebook in our testing framework
        import os
        smoke_test = ('CI' in os.environ)
        training_iterations = 2 if smoke_test else 130
        
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()


        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        # Set into eval mode
        model.eval()
        likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_data = getPose(lock, myPose)
            test_data = random.sample(test_data, len(test_data) // 2)
            test_x, test_y = ld.generate_train_dataset(test_data)
            predictions = likelihood(model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
            
            accuracy = abs((mean.numpy() - test_y.detach().numpy()))
            accuracy = np.sum(accuracy, axis=1)
            correct = 0
            for j in accuracy:
                if j < 0.038:
                    correct += 1
            
            a = correct / test_y.detach().numpy().shape[0]
            lock.acquire()
            myAcc = a
            lock.release()
        
        obs = env.reset()
        obs = image_transpose(obs)
        ep_reward = 0
        ep_loss = 0
        step_count = 0
        
        while step_count <= MAX_PER_EPIOSDE_STEPS:
            step_count += 1
            act = dqn.sample(obs, flag)
            
            v_x, v_y, w = actToVel(act)
            
            x = torch.FloatTensor([AGV_START_POSITION_X, AGV_START_POSITION_Y, AGV_START_ANGLE, v_x, v_y, w]).view(-1, 1)
            predictions = likelihood(model(x))
            pre = predictions.mean
            
            next_obs, reward, done = env.step(act, pre[0], pre[1], pre[2])
            next_obs = image_transpose(next_obs)

            dqn.memory_buffer.push(obs, act, reward, next_obs, done)
            obs = next_obs
            
            ep_reward += reward
            ep_loss += dqn.loss
            
            print("episode: %4d, step: %.3d, loss: %6.4f" % (episode, step_count, dqn.loss))
            
            
            if dqn.memory_buffer.size() >= MEMORY_CAPACITY:
                flag = True
                dqn.learn(a)
                torch.cuda.empty_cache()

            if done:    
                break
        ep_r_list.append(ep_reward)
        ep_l_lsit.append(ep_loss / step_count)
        print("Episode: %d, Episode Reward: %.2f" % (episode + 1, ep_reward))
        record_episode_reward(episode+1, ep_reward)
        record_episode_loss(episode+1, ep_loss / step_count)
        
        if episode % 10 == 0 and episode >= 500:
            save_checkpoint(dqn.evalue_net, dqn.target_net, dqn.optim, episode)   
    

def run(lock, mylist, myAcc, myPose):
    agv_run(lock, mylist, myAcc,myPose)

if __name__ == "__main__":
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    agv_run()





