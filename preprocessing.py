import numpy as np
import cPickle

def txt_to_data(filename):

    data = []
    f = open(filename, 'r')
    while True:
        line = f.readline()
        if not line: break
        data.append(Data(line.strip().split(' |')))
    f.close()

    return data


def txt_to_data_sample(filename, sample_size):

    data = []
    f = open(filename, 'r')
    for i in range(sample_size):
        line = f.readline()
        if not line: break
        data.append(Data(line.strip().split(' |')))
    f.close()

    return data



class Data:

    def __init__(self, raw_data):

        info = raw_data[0].split(' ')
        self.time = int(info[0])
        self.chosen_arm = int(info[1])
        self.is_clicked = bool(int(info[2]))
        self.user_context = self.parse_context(raw_data[1].split(' '))[1]
        self.arms = []
        self.arm_contexts = []
        for i in range(2, len(raw_data)):
            arm, context = self.parse_context(raw_data[i].split(' '))
            self.arms.append(arm)
            self.arm_contexts.append(context)
        self.contexts = np.outer(self.user_context, self.arm_contexts[0]).flatten()
        for j in range(1, len(self.arm_contexts)):
            self.contexts = np.vstack((self.contexts,
                                       np.outer(self.user_context, self.arm_contexts[j]).flatten()))

    @staticmethod
    def parse_context(raw_context):

        if raw_context[0] == 'user':
            arm = 'user'
        else:
            arm = int(raw_context[0])

        context = np.zeros(6)
        for i in range(1, len(raw_context)):
            idx_ctx = raw_context[i].split(':')
            context[int(idx_ctx[0])-1] = float(idx_ctx[1])

        return arm, context



class DataArm:

    def __init__(self, raw_data):

        info = raw_data[0].split(' ')
        self.time = int(info[0])
        self.chosen_arm = int(info[1])
        self.is_clicked = bool(int(info[2]))
        self.user_context = self.parse_context(raw_data[1].split(' '))[1]
        self.arms = []
        self.arm_contexts = []
        for i in range(2, len(raw_data)):
            arm, context = self.parse_context(raw_data[i].split(' '))
            self.arms.append(arm)
            self.arm_contexts.append(context)
        self.contexts = np.array(self.arm_contexts)

    @staticmethod
    def parse_context(raw_context):

        if raw_context[0] == 'user':
            arm = 'user'
        else:
            arm = int(raw_context[0])

        context = np.zeros(6)
        for i in range(1, len(raw_context)):
            idx_ctx = raw_context[i].split(':')
            context[int(idx_ctx[0])-1] = float(idx_ctx[1])

        return arm, context



