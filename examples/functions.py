from vectorlib.matrix import Matrix
from functions.softmax import softmax


m = Matrix(1, 4, 1, [[1,2,3,4]])
print(softmax(m))
