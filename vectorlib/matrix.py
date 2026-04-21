class Matrix():
    def __init__(self, arr_size, amount_col, init, start_mat = None):
        # remember
        # arr_size is length of list (columns), amount_col is amount of lists (x)
        # e.g: [[1,2], [1,2], [1,2]]
        # x = 2, y = 3
        # note that matrix has dims 
        self.value = [([init] * amount_col) for _ in range(arr_size)]
        self.dim: list[int] = [arr_size, amount_col]
        if start_mat and ((len(start_mat) != arr_size) or len(start_mat[0]) != amount_col):
            raise ValueError(f"Invalid start_mat entered for dim {arr_size} and {amount_col}")
        if start_mat:
            self.value = start_mat
        
        self.u, self.l, self.p = -1, -1, -1

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
        if self.dim != another_mat.dim:
            raise ValueError("Can only add matrix of same size")

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.value[i][j] -= another_mat.value[i][j]

        return self.value


    def scale(self, scalar):
        new_mat = Matrix(self.dim[0], self.dim[1], 1)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                new_mat.value[i][j] = self.value[i][j] * scalar
        return new_mat

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


    def apply_func(self, func):
        new_shit = Matrix(self.dim[0], self.dim[1], 0)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                new_shit.value[i][j] = func(self.value[i][j])
        
        return new_shit


    def lu_fact(self) -> tuple[Matrix, Matrix, Matrix]:
        # this function does Gassian Elimination on the matrix
        # Returns LU decomposition and P for row transforms

        if self.dim[0] != self.dim[1]:
            raise ValueError("Cannot ref on non-square mat for this program's purposes")
    
        
        if(self.l != -1 and self.u != -1 and self.p != -1):
            # cache exploit
            return self.u, self.l, self.p

        u_mat = self.copy()
        l_mat = Matrix(self.dim[0], self.dim[1], 0)

        pmat = Matrix(self.dim[0], self.dim[1], 0)
        num_swaps = 0 #<- used for determinant calc

        for i in range(self.dim[0]):
            # diagonalize these MFs
            pmat.value[i][i] = 1
            l_mat.value[i][i] = 1
        
        for col in range(self.dim[1]):
            # note that col is equiv to the target row
            # e.g, we are using col as the target row to do elimination
            max_val, indx = -1, -1
            for row in range(col, self.dim[0]): #<- calculate the best pivot to swap
                if u_mat.value[row][col] > max_val:
                    max_val = u_mat.value[row][col]
                    indx = row 

            # partial pivot, swap rows i and indx
            if indx != col and indx >= 0:
                num_swaps+=1
                og = u_mat.value[col]
                u_mat.value[col] = u_mat.value[indx]
                u_mat.value[indx] = og

                og = pmat.value[col]
                pmat.value[col] = pmat.value[indx]
                pmat.value[indx] = og
            
            # row reduction system
            # start at next row [i+1]
            for reduce_row in range(col+1, self.dim[0]):
                row_m = 0
                if u_mat.value[col][col] != 0: # <- layer is not already reduced
                    row_m = u_mat.value[reduce_row][col]/u_mat.value[col][col]
                    # mutlipler is the front guy divided by the non-zero GT

                for reduce_col in range(col, self.dim[1]): # step through and update the MF
                    l_mat.value[reduce_row][col] = row_m #<- set the L block to the multiplier
                    u_mat.value[reduce_row][reduce_col] -= u_mat.value[col][reduce_col] * row_m

        self.numswap = num_swaps
            
        # cache these operations to the matrix so that ref doesn't need to run multiple times
        # remember this has time complexity of O(N^3)
        self.u = u_mat
        self.l = l_mat
        self.p = pmat
            
        return u_mat, l_mat, pmat
        

    def det(self):
        # calculate the determinant of the matrix
        # utilizes the LU factorization to compute determinant

        u, _, _ = self.lu_fact() # <- throw away l and p; not needed for det

        # det(L) always is 1 since L is diag
        # det(U) is the product of diagonal
        detLU = 1
        for i in range(0, self.dim[0]):
            detLU *= u.value[i][i]
        return detLU * (-1)**(self.numswap)

    def solve(self, target: Matrix):
        # (LU)^-1
        # we can get both via backsub

        if self.det() < 0:
            raise ValueError("Cannot take inverse of non-singular matrix")

        if target.dim[0] != self.dim[0]:
            raise ValueError("Target must share same rows as input")
        
        u, l, _ = self.lu_fact() #<- factorize this MF

        # forward subbing L
        inner_answer = Matrix(target.dim[0], target.dim[1], 0) # outer inversion
            
        for t_col in range(target.dim[1]):
            l_forward = []

            for row in range(self.dim[0]):
                backstep = target.value[row][t_col]
                for col in range(len(l_forward)):
                    backstep -= l.value[row][col] * l_forward[col]
                
                l_forward.append(backstep)
                inner_answer.value[row][t_col] = backstep

        #inner_answer = Matrix(3, 2, 1, [[0, 3], [1, 2], [4, 5]])
        tmp_ans = Matrix(target.dim[0], target.dim[1], 1)
        # backward subbing U
        for t_col in range(inner_answer.dim[1]):
            u_backward = []
            for row in range((self.dim[0])):
                reverse_dim = self.dim[0] - row - 1
                backstep = inner_answer.value[reverse_dim][t_col]

                for col in range(len(u_backward)):
                    reverse_col = self.dim[1] - col - 1
                    backstep -= u.value[reverse_dim][reverse_col] * u_backward[col]
                backstep /= u.value[reverse_dim][reverse_dim]
                
                u_backward.append(backstep)

                tmp_ans.value[reverse_dim][t_col] = backstep
        return tmp_ans
                
                
    def invert(self):
        return self.solve(IMAT(self.dim[0]))

        
# basic matrices
def IMAT(n):
    m = Matrix(n, n, 0)
    for i in range(n):
        m.value[i][n - i - 1] = 1
    return m