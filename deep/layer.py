from vectorlib.matrix import Matrix
import functions.activations as func

class Layer():
    def __init__(self, input_dim, output_dim, activation):
        self.matrix = Matrix(output_dim, input_dim, 1) # initialize the matrix with weight 1

        self.input = input_dim
        self.output = output_dim
        
        # always use relu for activation
        if activation != "relu":
            self.activation = func.identity
        else:
            self.activation = func.relu

        # implement He initialization later

    def __str__(self):
        return str(self.matrix)

    def forward(self, input: Matrix):
        # this is xw^T
        wt = self.matrix.transpose()
        out = input.multiply(wt)

        activated_out = out.apply_func(self.activation)
        return activated_out