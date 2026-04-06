#########################################################
# An example of a very very basic single layer network  #
#########################################################

from vectorlib.matrix import Matrix
from vectorlib.vnorms import MSE

# our input and our output stored in nice matrices
x = Matrix(5, 4, 0, [[1, 2, 3, 4], [1, 1, 0, 1], [1,2,1, 1], [2,4,1, 1], [1, 2,5,2]])
t = Matrix(5, 1, 0, [[100], [30], [50], [80], [100]])

#test_x = [11, 14, 2, 7]
#val_y = [110, 140, 20, 70]

# 4x1
weights = Matrix(1, 4, 1, [[1, 1, 1, 1]])

# operations
# y = w^T x / x w^T
out = x.multiply(weights.transpose())
print(out.dim)

# we actually don't need this for single layer regression
# this is because the entire layer can be optimized w/o SGD
loss = MSE(t, out)
print(loss)

# Remember, we are using MSE
# AKA: least square loss
# sum (t_n - x)**2

# loss calculation
# Not needed for single linear network since optimum is easy to compute
# We use a pre-calculated derivative of the loss function to avoid compute graphs
# Recall that w* MLE is given by (x^T x)^(-1) x^T t

xtx = x.transpose().multiply(x).inverse()
xtt = x.transpose().multiply(t)
print(xtx.multiply(xtt))