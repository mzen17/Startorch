# get the loss gradient via backprop
from deep.layer import Layer
from vectorlib.matrix import Matrix

def gradient(network: list[Layer], forward: list[Matrix], output: Matrix):
    # note: list must be same size as network for each matrix inside
    
    # output MUST be the same dim as the final output of network
    # output layer is 1 x 1 matrix

    # assume loss is MSE for now

    grad = []

    weights = network[::-1]
    forward = forward[::-1]

    for i in range(len(weights)):
        layer = weights[i]

        loss = Matrix(layer.output, layer.input, 0)

        for wi in range(layer.output):
            for wj in range(layer.input):
                a_ki = forward[i + 1].value[0][wj] # h(a_ki)

                if i > 0:
                    last_grad = grad[-1].value[0][wi] #  dE/da_j by taking the last gradient of the output                    
                    dndm = layer.matrix.value[wi][wj] # dn_dm

                    e_aj = last_grad * dndm

                else: 
                    e_aj = -2*(output.value[0][wi] - forward[i].value[0][wi]) # 2 (t - y)
                    #print(f"dE/dw{wi}{wj} = ({e_aj}) * {a_ki} ")

                loss.value[wi][wj] = e_aj * a_ki
        grad.append(loss)


    return grad