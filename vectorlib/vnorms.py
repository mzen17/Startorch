from vectorlib.matrix import Matrix

def MSE(t: Matrix, x: Matrix):
    # expects two matrices of dims (1,n)

    if t.dim != x.dim:
        raise ValueError("Cannot do MSE of different vec sizes")

    loss = Matrix(x.dim[0], x.dim[1], 0)
    numerical_loss = 0
    for i in range(t.dim[0]):
        for j in range(t.dim[1]):
            loss.value[i][j] = (t.value[i][j] - x.value[i][j])**2
            numerical_loss += loss.value[i][j]
    
    return loss, numerical_loss/(t.dim[0] * t.dim[1])
