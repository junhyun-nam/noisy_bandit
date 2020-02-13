import preprocessing
import algorithms as alg
import numpy as np
import sys
import os
import time

import cPickle


theta = np.random.uniform(-1.0, 1.0, size=35)
dimension = 35
sampling_rate = 1.0
subsampling = True
num_arms = 10

seed = 0
r_mask = np.random.RandomState(seed)
r_subsample = np.random.RandomState(seed+1)


# create algorithms
algorithms = dict()
cum_rewards = dict()
algorithms['Random'] = alg.Random()
algorithms['LinUCB'] = alg.LinUCB(dimension, alpha=0.25)
algorithms['DropLinUCB'] = alg.DropLinUCB(theta, dimension, alpha=0.25)

for algo in algorithms:
    cum_rewards[algo] = [0]


start_time = time.time()
cur_time = time.time()

num_data = 1660000
f = open('datasets/yahoo-a1.txt')

for t in range(num_data):

    line = f.readline()
    data = preprocessing.Data(line.strip().split(' |'))
    
    chosen_arm = data.chosen_arm
    chosen_idx = data.arms.index(chosen_arm)
    is_clicked = data.is_clicked
    contexts = data.contexts
    arms = data.arms

    # subsampling
    if subsampling and len(arms) > num_arms:

        subsample = []
        while chosen_idx not in subsample:
            subsample = np.sort(r_subsample.choice(len(arms), num_arms, replace=False))
        arms_subsample = np.array(arms)[subsample]
        chosen_idx_subsample = np.where(subsample == chosen_idx)[0][0]
        contexts_subsample = contexts[subsample, :]

        chosen_idx = chosen_idx_subsample
        contexts = contexts_subsample
        arms = arms_subsample


    # sampling
    mask = r_mask.binomial(1, sampling_rate, (num_arms, dimension))
    contexts = contexts[:, 1:] * mask
    
    # choose arm for each algorithm and update ctr and record
    for algo in algorithms:
        if algorithms[algo].choose(contexts, mask) == chosen_idx:
            algorithms[algo].update(reward=int(is_clicked))
            cum_rewards[algo].append(cum_rewards[algo][-1] + int(is_clicked))

            
    if (t+1)%100000 == 0 or t+1 == num_data:
        print('[{}/{}]'.format(t+1, num_data))
        print('Current 100k iteration: {} sec'.format(time.time() - cur_time))
        print('Elapsed: {} sec'.format(time.time() - start_time))
        print('Random: {}'.format( float(cum_rewards['Random'][-1])/(len(cum_rewards['Random'])-1) ))
        print('LinUCB: {}'.format( float(cum_rewards['LinUCB'][-1])/(len(cum_rewards['LinUCB'])-1) ))
        print('DropLinUCB: {}'.format( float(cum_rewards['DropLinUCB'][-1])/(len(cum_rewards['DropLinUCB'])-1) ))
        cur_time = time.time()

f.close()

with open('results/real_cum_rewards.pkl', 'w') as f:
    cPickle.dump(cum_rewards, f)
    f.close()




