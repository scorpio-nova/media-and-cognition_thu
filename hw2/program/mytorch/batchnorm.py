# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))#方差
        self.mean = np.zeros((1, in_feature))#均值

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature) #就是y吧
        """

        self.x = x

        # Inference mode 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        if eval:
            self.mean = None
            self.var = None
            self.norm = None
            # ToDo:
            # Hint: use self.gamma, self.x, self.running_mean,self.running_var, self.eps,self.beta and np.sqrt() to calculate self.out
            # Hint: Please refer to https://kevinzakka.github.io/2016/09/14/batch_normalization/
            # 参考:https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.10_batch-norm
            self.out=self.gamma*(x - self.running_mean) / np.sqrt(self.running_var + self.eps)+self.beta
            return self.out

        # Training mode
        # 对初始数据求均值方差
        self.mean = np.mean(self.x, axis = 0)
        self.var = np.var(self.x, axis = 0)

        # ToDo:
        # Hint: use self.x, self.mean,self.var, self.eps and np.sqrt() to calculate self.norm
        # Hint: use self.gamma, self.norm and self.beta to calculate self.out
        # Hint: Please refer to https://kevinzakka.github.io/2016/09/14/batch_normalization/
        self.norm=(self.x-self.mean)/np.sqrt(self.var+self.eps)
        self.out=self.gamma*self.norm+self.beta

        # update running batch statistics
        self.running_mean = self.running_mean * self.alpha + self.mean * (1 - self.alpha)#这里是momentum机制，alpha是动量参数

        # ToDo:
        # Hint: update self.running_var similar to self.running_mean
        self.running_var=self.running_var*self.alpha+self.var * (1 - self.alpha)
        return self.out
        raise NotImplemented


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)=d f/d y
        Return:
            out (np.array): (batch size, in feature)
        """
        m = self.x.shape[0] # batch size
        dgamma = np.sum(delta * self.norm, axis = 0)#=pd f/pd gamma
        dbeta = np.sum(delta, axis = 0)#=pd f/pd beta

        self.dgamma[0] = dgamma
        self.dbeta[0] = dbeta

        dx_hat = self.gamma * delta
        dvar = np.sum(dx_hat * (self.x - self.mean), axis=0) * (-1 / 2) * (self.var + self.eps)**(-3 / 2)
        dmean = -np.sum(dx_hat, axis = 0) / np.sqrt(self.var + self.eps) - 2 / m * dvar * np.sum(self.x - self.mean, axis=0)

        # ToDo:
        # Hint: use dx_hat,self.var, self.eps, dvar, batch size (m), self.x self.mean, dmean and np.sqrt() to calculate dx
        # Hint: please refer to https://kevinzakka.github.io/2016/09/14/batch_normalization/
        dx =dx_hat*(1/np.sqrt(self.var+self.eps))+dmean/m+dvar*2*(self.x-self.mean)/m
        return dx