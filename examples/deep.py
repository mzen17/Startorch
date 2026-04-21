#########################################################
# An example of a basic linear layer network using SGD  #
#########################################################

import random as rnd

from vectorlib.matrix import Matrix
from vectorlib.vnorms import MSE

from deep.backprop import gradient
from deep.layer import Layer

# our input and our output stored in nice matrices
# input: 5x4 (5 rows, 4 features)
# output: 5x1 (5 rows, 1 output)

n = 1
x_dim = 4

# 4 x 2 | 2 x 1 neural network
l1 = Layer(x_dim, 2, "relu")
l2 = Layer(2, 1, "id")

print("GRADIENT")

print("SGD")
lr = 0.01 # < sgd learning rate
epochs = 400

for i in range(epochs):
    data = []
    eval = []

    # SGD, b=1
    for _ in range(n):
        row = []
        w_sum = 1
        for j in range(x_dim):
            tv = rnd.random()
            row.append(tv)
            w_sum *= tv
        data.append(row)
        eval.append([w_sum/len(row)])

    x = Matrix(n, x_dim, 0, data)
    t = Matrix(1, 1, 0, eval)

    out = l1.forward(x)
    out2 = l2.forward(out)

    lmat, na = MSE(t, out2)
    print(f"LOSS: {na}")
    
    tg = gradient([l1, l2], [x, out, out2], t)

    l2.matrix.sub(tg[0].scale(lr))
    l1.matrix.sub(tg[1].scale(lr))
