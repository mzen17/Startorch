from vectorlib.matrix import Matrix

def MSE(t: Matrix, x: Matrix):
    # expects two matrices of dims (1,n)
    #
    if t.dim != x.dim:
        raise ValueError("Cannot do MSE of different vec sizes")

    if t.dim[1] != 1 or x.dim[1] != 1:
        raise ValueError("Cannot do vector norm on fat matrices")
    
    loss = Matrix(x.dim[0], 1, 0)
    for i in range(t.dim[0]):
        loss.value[i][0] = (t.value[i][0] - x.value[i][0])**2
    
    return loss
