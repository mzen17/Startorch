from vectorlib.matrix import Matrix
def softmax(input: Matrix):
    # takes in amatrix and outputs a softmax of the matrix
    # softmaxes along the rows

    m = Matrix(input.dim[0], input.dim[1], 1)

    # compute e^sum for each row lol
    row_esum = []

    for x in range(input.dim[0]):
        sum = 0
        for y in range(input.dim[1]):
            sum += 2.718**input.value[x][y]
        
        row_esum.append(sum)
    
    for x in range(input.dim[0]):
        for y in range(input.dim[1]):
            m.value[x][y] = 2.718**(input.value[x][y]) / row_esum[x]
    
    return m