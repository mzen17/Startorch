#########################################################
# An example of a basic single layer network            #
#########################################################

from vectorlib.matrix import Matrix
from vectorlib.vnorms import MSE
import random as rnd

# our input and our output stored in nice matrices
# input: 5x4 (5 rows, 4 features)
# output: 5x1 (5 rows, 1 output)

n = 1000000

data = []
eval = []
for i in range(n):
    i1, i2, i3, i4 = rnd.random(), rnd.random(), rnd.random(), rnd.random()
    data.append([i1, i2, i3, i4])
    eval.append([i1 + i2 + i3 + i4])

x = Matrix(n, 4, 0, data)
t = Matrix(n, 1, 0, eval)

# 1xn weight matrix
weights = Matrix(1, 4, 1)

# operations
# y = x w^T
out = x.multiply(weights.transpose())

# Remember, we are using MSE
# AKA: least square loss
# sum (t_n - x)**2

# we actually don't need this for single layer regression
# this is because the entire layer can be optimized w/o SGD
loss, na = MSE(t, out)
print(f"OG LOSS {na}")
# => [[8100], [729], [2025], [5184], [8100]]

# loss calculation
# Not needed for single linear network since optimum is easy to compute
# We use a pre-calculated derivative of the loss function to avoid compute graphs
# Recall that w* MLE is given by (x^T x)^(-1) x^T t

xxtI = x.transpose().multiply(x).invert() # 4 x 5 * 5 x 4 => 4 x 4
xt = x.transpose().multiply(t) # 4 x 5 | 5 x 1 => 4 x 1
wMLE = xxtI.multiply(xt) # 4 x 4 | 4 x 1 => 4 x 1

out2 = x.multiply(wMLE)

loss, na = MSE(t, out2)
# => [[16873.686370489217], [92084.66115702533], [276.4952847203851], [37.79006990187767], [2713.8217837918023]]
print(na)