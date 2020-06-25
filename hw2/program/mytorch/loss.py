# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        #logits是作为softmax的输入。经过softmax的加工，就变成“归一化”的概率（设为q)qi=xi/sm
        self.labels = y
        #p
        #目标:求-pilogqi

        # LogSumExp trick: please refer to https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        #防止计算y=log(e^(x1)+e^(x2)+...)的时候由于指数的缘故上下溢出,改为计算等价式y=a+log(e^(x1-a)+e^(x2-a)+...),a=x_max
        #[:, np.newaxis]作用:在np.newaxis位置增加新维度 见https://blog.csdn.net/weixin_42866962/article/details/82811082
        #maxx因为是对axis=1求max得，故形状为(batch size,),maxx[:, np.newaxis]之后是(batch size,1)? 为啥不直接keepdims=True?
        maxx = np.max(x, axis = 1)
        self.sm = maxx + np.log(np.sum(np.exp(x - maxx[:, np.newaxis]), axis=1))#log(sum_exp(x))
        
        # ToDo:
        # Hint: use self.logits, self.labels, self.sm, and np.sum(???, axis = 1)#??怎么没说用log
        return -np.sum(self.labels * np.log(np.exp(self.logits)/np.exp(self.sm[:,np.newaxis])),axis=1)
        # raise NotImplemented

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        # ToDo:
        # Hint: fill in self.logits and self.labels in the following sentence
        return (np.exp(self.logits) / np.exp(self.sm)[:, np.newaxis]) - self.labels
        # raise NotImplemented