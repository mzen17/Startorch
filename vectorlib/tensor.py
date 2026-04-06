# The most inefficient Tensor library
from copy import deepcopy
class Tensor():
    def __init__(self, dims, init=0):
        self.value = [init] * dims[0]
        self.dims = dims
        for i in dims[1:]:
            new_val = []
            for _ in range(i):
                new_val.append(deepcopy(self.value))
            self.value = new_val

    def generate_ij_index(self):
        # time complexity is self.size()
        inputs = []
        for i in self.dims:
            arr = []
            for j in range(0, i):
                arr.append(j)
            inputs.append(arr)
        
        currboi = [[]]
        for i in inputs:
            newarr = []
            for j in i:
                for k in currboi:
                    dup = k.copy()
                    dup.append(j)
                    newarr.append(dup)
            currboi=newarr
        
        return currboi

    def size(self) -> int:
        prod = 1
        for i in self.dims:
            prod*=i
        return prod
        
    def add(self, another_mat: Tensor):
        if another_mat.dims != self.dims:
            raise ValueError("Cannot add different dimension matrices")

        indexes = self.generate_ij_index()
        new_mat = deepcopy(self.value)

        for iteration in indexes:
            mat = deepcopy(self.value)
            mat2 = another_mat.value

            for i in range(0, len(iteration) -1):
                mat = mat[i]
                mat2 = mat2[i]
                indx = new_mat[i]

            for i in iteration[:1]:
                indx = new_mat[i]
            
            indx[iteration[-1]] = mat[iteration[-1]] + mat2[iteration[-1]]

        self.value = new_mat
        return new_mat

    def multiply(self, another_mat: Tensor):
        curr_index = self.generate_ij_index()
        ant_index = another_mat.generate_ij_index()

        # remember: 
        # 4x5x7

    def __str__(self):
        return str(self.value)
    
    
m = Tensor([4,4], 2)
m2 = Tensor([4, 4], 54)
print(m.add(m2))

