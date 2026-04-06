class Matrix():
    def __init__(self, x_size, y_size, init, start_mat = None):
        # remember
        # x_size is length of list, y_size is amount of lists
        # e.g: [[1,2], [1,2], [1,2]]
        # x = 2, y = 3
        self.value = [([init] * y_size) for _ in range(x_size)]
        self.dim: list[int] = [x_size, y_size]
        if start_mat:
            self.value = start_mat

    def __str__(self):
        return str(self.value)
    

    def copy(self):
        new_mat = Matrix(self.dim[0], self.dim[1], 1)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                new_mat.value[i][j] = self.value[i][j]
        return new_mat

    def add(self, another_mat: Matrix):
        if self.dim != another_mat.dim():
            raise ValueError("Can only add matrix of same size")

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.value[i][j] += another_mat.value[i][j]

        return self.value
    

    def sub(self, another_mat: Matrix):
        if self.dim != another_mat.dim():
            raise ValueError("Can only add matrix of same size")

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.value[i][j] -= another_mat.value[i][j]

        return self.value


    def multiply(self, another_mat: Matrix):
        if self.dim[1] != another_mat.dim[0]:
            raise ValueError("Inner product term does not match")
        
        shared = self.dim[1]
    
        new_mat = Matrix(self.dim[0], another_mat.dim[1], 1)

        for i in range(self.dim[0]):
            for j in range(another_mat.dim[1]):
                tv = 0
                for k in range(shared):
                    tv += self.value[i][k] * another_mat.value[k][j]
                new_mat.value[i][j] = tv
        return new_mat


    def transpose(self):
        new_shit = Matrix(self.dim[1], self.dim[0], 1) 
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                new_shit.value[j][i] = self.value[i][j]
        return new_shit


    def row_reduce(self):
        # this function does Gassian Elimination on the matrix
        # it returns the matrix in an upper triangular form

        
        pass


    def inverse(self):
        return self


    def det(self):
        # calculate the determinant of the matrix
        pass
        
