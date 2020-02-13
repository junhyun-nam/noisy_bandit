import numpy as np
import math
import sys
import os
import algorithms as alg

import cPickle


def run_experiment(options):

    experiment = options.experiment
    time = options.time
    tau = time / 20
    sampling_rate = options.sampling_rate
    arms = options.arms
    dimension = options.dimension
    seed = int(options.randomseed)

    d = 0.1

    # assign true parameter theta for the experiment
    r_theta = np.random.RandomState(1)
    theta = r_theta.uniform(-1.0, 1.0, size=dimension)
    theta /= np.linalg.norm(theta)

    # assign true mean and covariance
    r_param = np.random.RandomState(2)
    mean = r_param.uniform(0.0, 1.0, size=dimension)
    A_feature = r_param.normal(0, 1, size=(dimension, dimension))
    cov_feature = A_feature.dot(A_feature.T)
    A_noise = r_param.normal(0, 1, size=(dimension, dimension))
    cov_noise = A_noise.dot(A_noise.T)

    theta_bar = np.linalg.solve(cov_feature+cov_noise, np.dot(cov_feature, theta))


    # fix random seed
    np.random.seed(seed)
    r_context = np.random.RandomState(seed + 1000)
    r_noise = np.random.RandomState(seed + 2000)
    r_mask = np.random.RandomState(seed + 3000)
    r_reward = np.random.RandomState(seed + 4000)

    # create algorithms
    algorithms = dict()
    cum_rewards = dict()
    algorithms['Random'] = alg.Random()
    algorithms['Oracle'] = alg.Oracle(theta, mean, cov_feature, cov_noise)
    algorithms['LinUCB'] = alg.LinUCB(dimension)
    algorithms['GCESF'] = alg.GCESF(dimension, tau)
    algorithms['DropLinUCB'] = alg.DropLinUCB(theta_bar, dimension)

    for algo in algorithms:
        cum_rewards[algo] = [0]


    # iteration over time
    for i in xrange(time):

        # D-dimensional true context of K arms
        Z = np.zeros((arms, dimension))
        for arm in xrange(arms):
            ctx = r_context.multivariate_normal(mean=mean, cov=cov_feature)
            Z[arm] += ctx

        # non-identical D-dimensional noise for arms
        X = np.array(Z)
        for arm in xrange(arms):
            epsilon = r_noise.multivariate_normal(mean=np.zeros(dimension), cov=cov_noise)
            X[arm] += epsilon

        # drop context
        M = r_mask.binomial(1, sampling_rate, (arms, dimension))
        X *= M

        # choose arms for each algorithm
        for algo in algorithms:
            algorithms[algo].choose(np.array(X), np.array(M))

        # compute rewards
        reward_expected = Z.dot(theta)
        delta = r_reward.normal(scale=d, size=arms)
        reward = reward_expected + delta

        # update
        for algo in algorithms:
            algorithms[algo].update(reward[algorithms[algo].arm])
            cum_rewards[algo].append(cum_rewards[algo][-1] + reward[algorithms[algo].arm])

        if i%1000 == 0:
            print(i)

    for algo in algorithms:
        print(algo, algorithms[algo].cum_reward)

    print(algorithms['Oracle'].theta_bar)
    print(algorithms['LinUCB'].theta_hat[1:])
    print(algorithms['GCESF'].theta_hat[1:])
    print(algorithms['DropLinUCB'].theta_hat[1:])
    print('LinUCB cos dist')
    print(np.dot(algorithms['Oracle'].theta_bar, algorithms['LinUCB'].theta_hat[1:]) / 
        (np.linalg.norm(algorithms['Oracle'].theta_bar) * np.linalg.norm(algorithms['LinUCB'].theta_hat[1:]) ) )
    print('GCESF cos dist')
    print(np.dot(algorithms['Oracle'].theta_bar, algorithms['GCESF'].theta_hat[1:]) / 
        (np.linalg.norm(algorithms['Oracle'].theta_bar) * np.linalg.norm(algorithms['GCESF'].theta_hat[1:]) ) )
    print('DropLinUCB cos dist')
    print(np.dot(algorithms['Oracle'].theta_bar, algorithms['DropLinUCB'].theta_hat[1:]) / 
        (np.linalg.norm(algorithms['Oracle'].theta_bar) * np.linalg.norm(algorithms['DropLinUCB'].theta_hat[1:]) ) )

    filename = 'S{}_K{}_D{}_P{}.pkl'.format(seed, arms, dimension, sampling_rate)

    with open('results/'+filename, 'w') as f:
        cPickle.dump(cum_rewards, f)
        f.close()



def default(str):
    return str + ' [Default: %default]'


def read_command(argv):
    """Processes the command used to run from the command line."""
    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option('-L', '--linucb', help=default('Run LinUCB'), default=False)
    parser.add_option('-E', '--experiment', help=default('Name of experiment'), default='temp')
    parser.add_option('-S', '--randomseed', help=default('Random Seed'), default=2001)
    parser.add_option('-T', '--time', help=default('time horizon'), type=int, default=50000)
    parser.add_option('-K', '--arms', help=default('number of arms'), type=int, default=5)
    parser.add_option('-D', '--dimension', help=default('dimension of context'), type=int, default=10)
    parser.add_option('-P', '--sampling_rate', help=default('sampling rate'), type=float, default=0.5)


    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))

    return options

options = read_command( sys.argv[1:])
run_experiment(options)
if __name__ == 'main':
    # Read input
    options = read_command( sys.argv[1:] )
    # Run experiment
    run_experiment(options)




