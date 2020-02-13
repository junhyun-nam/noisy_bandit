import math
import numpy as np


class Random:

    def __init__(self):
        self.arm = 0
        self.cum_reward = .0

    def choose(self, contexts, arms):
        K = contexts.shape[0]
        arm_random = np.random.choice(K)
        self.arm = arm_random
        return arm_random

    def update(self, reward):
        self.cum_reward += reward



class Oracle:

    def __init__(self, theta, nu, Sigma_feature, Sigma_noise):
        self.theta = theta
        self.nu = nu
        self.Sigma = Sigma_feature + Sigma_noise
        self.theta_bar = np.linalg.solve(self.Sigma, np.dot(Sigma_feature, self.theta))
        self.d = self.theta.shape[0]
        self.arm = 0
        self.context = np.zeros(self.d)
        self.cum_reward = .0

    def choose(self, contexts, mask):

        # observe K features
        X = contexts
        K = X.shape[0]

        # recover missing parts
        X_bar = np.zeros((K, self.d))
        for i in range(K):
            q = len(np.nonzero(mask[i])[0]) # number of nonzero elements in x_i
            R = np.zeros((self.d, self.d)) # coordinate change matrix
            seen = np.where(mask[i] != 0)[0]
            unseen = np.where(mask[i] == 0)[0]
            coords = np.append(seen, unseen)
            R[np.arange(self.d), coords] = 1
            x = R.dot(X[i]) # sort nonzero elements
            nu = R.dot(self.nu)
            sigma = np.matmul(R, np.matmul(self.Sigma, R.T))
            sigma_11_inv = np.linalg.inv(sigma[:q, :q])
            sigma_21 = sigma[q:, :q]
            x_bar_unseen = nu[q:] + np.matmul(sigma_21, sigma_11_inv).dot(x[:q] - nu[:q])
            x_bar = X[i]
            x_bar[unseen] = x_bar_unseen
            # X_bar[i][0]
            X_bar[i] = x_bar

        # choose arm
        p = X_bar.dot(self.theta_bar)
        arm_oracle = np.argmax(p)
        self.arm = arm_oracle

        return arm_oracle

    def update(self, reward):
        self.cum_reward += reward



class GCESF:

    def __init__(self, dim, tau):
        self.d = dim
        self.tau = tau

        self.t = 1
        self.n = 0

        self.X = np.zeros((self.d+1, self.d+1))
        self.Y = np.zeros(self.d + 1)
        self.Z = np.zeros((self.d, self.d))
        self.xi = np.zeros(self.d)

        self.cum_reward = 0.

        self.phase = 1


    def choose(self, contexts, mask):

        X = contexts
        K = contexts.shape[0]
        self.K = K

        if self.phase == 1:

            arm_gcesf = np.random.choice(K)
            self.arm = arm_gcesf

        else:

            # recover missing parts
            X_bar = np.zeros((K, self.d +1))
            for i in range(K):
                q = len(np.nonzero(mask[i])[0]) # number of nonzero elements in x_i
                R = np.zeros((self.d, self.d)) # coordinate change matrix
                seen = np.where(mask[i] != 0)[0]
                unseen = np.where(mask[i] == 0)[0]
                coords = np.append(seen, unseen)
                R[np.arange(self.d), coords] = 1
                x = R.dot(X[i]) # sort nonzero elements
                nu = R.dot(self.nu_hat)
                sigma = np.matmul(R, np.matmul(self.Sigma_hat, R.T))
                sigma_11_inv = np.linalg.inv(sigma[:q, :q])
                sigma_21 = sigma[q:, :q]
                x_bar_unseen = nu[q:] + np.matmul(sigma_21, sigma_11_inv).dot(x[:q] - nu[:q])
                x_bar = X[i]
                x_bar[unseen] = x_bar_unseen
                X_bar[i][0] = 1
                X_bar[i][1:] = x_bar

            arm_gcesf = np.argmax(X_bar.dot(self.theta_hat))
            self.arm = arm_gcesf

        self.context = X[self.arm]
        self.contexts = X
        self.mask = mask

        return arm_gcesf


    def update(self, reward):

        if self.phase == 1:

            x = np.zeros(self.d+1)
            x[0] = 1
            x[1:] = self.context
            y = reward
            X = self.contexts

            self.t += 1
            self.n += np.sum(self.mask)

            self.X += np.outer(x, x)
            self.Y += x*y

            self.Z *= float(self.t -1)/self.t
            self.Z += X.T.dot(X)/self.t/X.shape[0]

            self.xi *= float(self.t -1)/self.t
            self.xi += np.sum(X, axis=0)/self.t/X.shape[0]

            # if self.t < 10:
            #     print('gcesf')
            #     print(self.xi)
            #     print(X)

            if self.t > self.tau:

                self.phase = 2
                self.estimate()
                print('p_hat: {}'.format(self.p_hat))
        
        self.cum_reward += reward


    def estimate(self):

        self.p_hat = max(1.0, self.n) / float(self.tau*self.K*self.d)

        # calibration
        X_mask = np.zeros((self.d+1, self.d+1))
        X_mask[0, 0] = 1
        X_mask[0, 1:] = 1 / self.p_hat
        X_mask[1:, 0] = 1 / self.p_hat
        X_mask[1:, 1:] = ( (self.p_hat - 1) * np.eye(self.d) + np.ones((self.d, self.d)) ) / self.p_hat**2

        Y_mask = np.ones(self.d+1)
        Y_mask[1:] = 1 / self.p_hat

        self.X_hat = self.X * X_mask
        self.Y_hat = self.Y * Y_mask

        self.theta_hat = np.linalg.solve(self.X_hat, self.Y_hat)
        self.nu_hat = self.xi / self.p_hat
        self.Sigma_hat = self.Z * ( (self.p_hat - 1) * np.eye(self.d) + np.ones((self.d, self.d)) ) / self.p_hat**2
        self.Sigma_hat -= np.outer(self.nu_hat, self.nu_hat)

        print('GCESF theta_hat')
        print(self.theta_hat)



class LinUCB:

    def __init__(self, dim, alpha=0.25):
        self.d = dim
        self.alpha = alpha
        self.X = np.eye(self.d +1)
        self.X_inv = np.linalg.inv(self.X)
        self.Y = np.zeros(self.d +1)
        self.theta_hat = np.linalg.solve(self.X, self.Y)
        self.arm = 0
        self.context = np.zeros(self.d)
        self.cum_reward = .0

    def choose(self, contexts, arms):
        # compute new parameters
        self.X_inv = np.linalg.inv(self.X)
        self.theta_hat = np.linalg.solve(self.X, self.Y)

        # observe K features
        X = contexts
        X_ = np.zeros((X.shape[0], self.d+1))

        # concat 1 for bias
        X_[:, 0] = 1
        X_[:, 1:] = X

        # compute upper confidence bound
        p = X_.dot(self.theta_hat) + self.alpha * np.sqrt((X_.dot(self.X_inv)*X_).sum(axis=1))

        # choose action
        arm_linucb = np.argmax(p)
        self.arm = arm_linucb
        self.context = X[self.arm]

        return arm_linucb

    def update(self, reward):
        # update parameters
        x = np.zeros(self.d+1)
        x[0] = 1
        x[1:] = self.context
        y = reward

        self.X += np.outer(x, x)
        self.Y += x*y

        self.cum_reward += reward



class DropLinUCB:

    def __init__(self, theta_bar, dim, alpha=0.25):
        self.theta_bar = theta_bar
        self.d = dim
        self.alpha = alpha
        self.X = np.eye(self.d + 1)
        self.X_inv = np.linalg.inv(self.X)
        self.X_hat = np.eye(self.d + 1)
        self.X_hat_inv = np.linalg.inv(self.X_hat)
        self.Y = np.zeros(self.d + 1)
        self.Y_hat = np.zeros(self.d + 1)
        self.Z = np.eye(self.d)
        self.Z_hat = np.eye(self.d)
        # self.Z = dict()
        # self.Z_hat = dict()
        self.xi = np.zeros(self.d)
        self.nu_hat = np.zeros(self.d)
        self.Sigma_hat = np.eye(self.d)
        self.p_hat = .0
        self.theta_hat = np.linalg.solve(self.X_hat, self.Y)
        self.t = 1
        self.arm = 0
        self.cum_reward = .0
        # self.use_part1 = use_part1
        # self.use_part2 = use_part2

    def choose(self, contexts, mask):

        # observe K features
        X = contexts
        K = X.shape[0]

        X_ = np.zeros((X.shape[0], self.d+1))

        s = np.sum(mask)
        p_hat = max(1.0, s + (self.t-1)*K*self.d*self.p_hat) / (self.t*K*self.d)

        # calibration
        X_mask = np.zeros((self.d+1, self.d+1))
        X_mask[0, 0] = 1
        X_mask[0, 1:] = 1 / p_hat
        X_mask[1:, 0] = 1 / p_hat
        X_mask[1:, 1:] = ( (p_hat - 1) * np.eye(self.d) + np.ones((self.d, self.d)) ) / p_hat**2

        Y_mask = np.ones(self.d+1)
        Y_mask[1:] = 1 / p_hat

        self.X_hat = self.X * X_mask
        self.Y_hat = self.Y * Y_mask


        # compute new parameters
        self.X_hat_inv = np.linalg.inv(self.X_hat)
        self.theta_hat = np.linalg.solve(self.X_hat, self.Y_hat)
        self.nu_hat = self.xi / p_hat
        self.Sigma_hat = self.Z * ( (p_hat - 1) * np.eye(self.d) + np.ones((self.d, self.d)) ) / p_hat**2
        self.Sigma_hat -= np.outer(self.nu_hat, self.nu_hat)


        # recover missing parts
        X_bar = np.zeros((K, self.d +1))
        for i in range(K):
            q = len(np.nonzero(mask[i])[0]) # number of nonzero elements in x_i
            R = np.zeros((self.d, self.d)) # coordinate change matrix
            seen = np.where(mask[i] != 0)[0]
            unseen = np.where(mask[i] == 0)[0]
            coords = np.append(seen, unseen)
            R[np.arange(self.d), coords] = 1
            x = R.dot(X[i]) # sort nonzero elements
            nu = R.dot(self.nu_hat)
            sigma = np.matmul(R, np.matmul(self.Sigma_hat, R.T))
            sigma_11_inv = np.linalg.inv(sigma[:q, :q])
            sigma_21 = sigma[q:, :q]
            x_bar_unseen = nu[q:] + np.matmul(sigma_21, sigma_11_inv).dot(x[:q] - nu[:q])
            x_bar = np.array(X[i])
            x_bar[unseen] = x_bar_unseen
            X_bar[i][0] = 1
            X_bar[i][1:] = x_bar

        # concat 1 for bias
        X_[:, 0] = 1
        X_[:, 1:] = X


        # compute upper confidence bound
        ucb = X_bar.dot(self.theta_hat) + self.alpha * np.sqrt((X_.dot(self.X_hat_inv)*X_).sum(axis=1))


        # choose action
        arm_linucb = np.argmax(ucb)
        self.arm = arm_linucb
        self.context = X[self.arm]
        self.contexts = X
        self.mask = mask

        return arm_linucb


    def update(self, reward):

        # update parameters
        x = np.zeros(self.d+1)
        x[0] = 1
        x[1:] = self.context
        y = reward
        X = self.contexts

        s = np.sum(self.mask)
        K = X.shape[0]
        self.p_hat = max(1.0, s + (self.t-1)*K*self.d*self.p_hat) / (self.t*K*self.d)
        self.t += 1

        self.X += np.outer(x, x)
        self.Y += x*y

        self.Z *= float(self.t -1)/self.t
        self.Z += X.T.dot(X)/self.t/X.shape[0]

        self.xi *= float(self.t -1)/self.t
        self.xi += np.sum(X, axis=0)/self.t/X.shape[0]

        
        self.cum_reward += reward








