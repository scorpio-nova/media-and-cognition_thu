

# 3.深度学习基础

## 3.1 线性回归

线性回归输出是一个连续值，因此适用于回归问题。softmax回归则适用于分类问题。由于线性回归和softmax回归都是单层神经网络，它们涉及的概念和技术同样适用于大多数的深度学习模型。我们以线性回归为例，介绍大多数深度学习模型的基本要素和表示方法。

###3.1.1线性回归的基本要素

3.1.1.1模型定义

3.1.1.2模型训练

3要素：训练数据、损失函数、优化算法

在求数值解的优化算法中，小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。

![image-20200322140640540](/Users/serena/Library/Application Support/typora-user-images/image-20200322140640540.png)

$|\mathcal{B}|$代表每个小批量中的样本个数（批量大小，batch size），$\eta$ 称作学习率（learning rate）并取正数。→人为设定，超参数。

3.1.1.3模型预测



###3.1.2线性回归的表示方法

3.1.2.1神经网络图

3.1.2.2矢量计算表达式

![image-20200322141208595](/Users/serena/Library/Application Support/typora-user-images/image-20200322141208595.png)





## 3.4 softmax回归

CELoss

![image-20200322151239908](/Users/serena/Library/Application Support/typora-user-images/image-20200322151239908.png)

![image-20200322151732915](/Users/serena/Library/Application Support/typora-user-images/image-20200322151732915.png)'