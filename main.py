import torch
import torch.nn as nn
import numpy as np

def learn_tensor():

    # initialization
    tensor0 = torch.empty(2,3,4) # create multidimensional tensor
    print(tensor0)
    tensor1 = torch.rand(2,3) # tensor with random values
    print(tensor1)
    tensor2 = torch.zeros(3,3, dtype=torch.double) # tensor with all zeros
    print(tensor2)
    print(tensor2.dtype)
    print(tensor2.size())
    tensor3 = torch.tensor([1.2, 3]) # initialize with value
    print(tensor3)

    # operations
    tensor4 = torch.rand(2,3)
    print(tensor1 + tensor4) # element wise addition
    print(torch.add(tensor1, tensor4))
    tensor4.add_(tensor1) # add in place
    print(tensor4)
    # note: any pytorch function with _ at the end will be in place

    # slicing / indexing
    print(tensor4[:,0]) # first column
    print(tensor4[1,1]) # index
    print(tensor4[1,1].item()) # get value

    # resize
    print(tensor4.view(6)) # convert to 1d
    # note: number of elements must be the same
    print(tensor4.view(-1,2)) # any number of rows, 2 columns

    # convert
    # note: if tensor is in CPU, not GPU, both objects will use same memory location
    array4 = tensor4.numpy() # tensor to numpy
    print(array4)
    array5 = np.ones((3,3))
    tensor5 = torch.from_numpy(array5) # numpy to tensor
    print(tensor5)

    # use cuda
    print(torch.cuda.is_available()) # check if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor6 = torch.ones(5,device=device)

    tensor7 = torch.ones(5, requires_grad=True) # tell pytorch it will need to calculate gradient later, used for optimization

def learn_gradient():

    tensor0 = torch.rand(3, requires_grad=True) # must specify requires_grad to use grad later
    tensor1 = tensor0 + 2
    print(tensor1) # grad function is AddBackward

    tensor2 = tensor0 * tensor0 * 2
    print(tensor2) # grad function is MulBackward

    tensor3 = tensor0.mean()
    print(tensor3) # grad function is MeanBackward

    print(tensor0.grad) # get gradient
    print(tensor3.backward()) # calculate gradient

    weights = torch.ones(4, requires_grad=True)
    for epoch in range(3): # training loop
        model_output = (weights * 3).sum()
        model_output.backward()
        print(weights.grad)
        weights.grad.zero_() # reset gradients at the end of training loop

def learn_back_propagation():

    # example input and weights
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)
    w = torch.tensor(1.0, requires_grad=True)

    # forward pass and compute loss
    y_hat = w * x
    loss = (y_hat - y) ** 2
    print(loss)

    # backward pass
    print(w.grad)
    loss.backward() # do backward pass, will populate w.grad
    print(w.grad) # dloss / dw

    # update weights
    # more passes

# doing gradient descent manually using numpy
def learn_gradient_descent_np():

    # we want to approximate a function where the output is input * 2
    # f = w * x
    # f = 2 * x
    X = np.array([1,2,3,4], dtype=np.float32) # input
    Y = np.array([2,4,6,8], dtype=np.float32) # expected output

    w = 0.0 # initialize weights

    # model prediction
    def forward(x):
        return w * x

    # loss = mean squared error
    # loss is how much error there is between output and expected output
    def loss(y, y_predicted):
        return ((y_predicted-y) ** 2).mean()

    # gradient
    # MSE = 1/N * (w*x - y) ** 2
    # dJ/dw = 1/N 2x (w*x - y) # numerically computed derivative
    def gradient(x,y,y_predicted):
        return np.dot(2*x, y_predicted-y).mean()

    print(f'prediction before training: f(5) = {forward(5):.3f}')

    # training
    learning_rate = 0.01
    n_iterations = 20

    for epoch in range (n_iterations): # training loop
        # predictions = forward pass
        y_pred = forward(X)

        # loss
        l = loss(Y, y_pred)

        # gradient
        dw = gradient(X,Y,y_pred)

        # update weight
        w -= learning_rate * dw # move in the direction of the gradient by learning_rate

        # print data
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

    # in the end, we expect weight = 2
    print(f'prediction after training: f(5) = {forward(5):.3f}')

# doing gradient descent automatically with pytorch
def learn_gradient_descent_torch():

    # we want to approximate a function where the output is input * 2
    # f = w * x
    # f = 2 * x
    X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # input
    Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32) # expected output

    X_test = torch.tensor([5], dtype=torch.float32)

    n_samples, n_features = X.shape # 4 samples, 1 feature per sample

    model = nn.Linear(n_features, n_features) # 1 input, 1 output

    # creating custom model
    class LinearRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__init__()
            # define layers
            self.lin = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.lin(x)

    model = LinearRegression(n_features, n_features)

    print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

    # training
    learning_rate = 0.01
    n_iterations = 1000

    # loss
    # calculate loss using pytorch neural network package
    loss = nn.MSELoss()

    # optimizer
    # SGD = stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range (n_iterations): # training loop
        # predictions = forward pass
        y_pred = model(X)

        # loss
        l = loss(Y, y_pred)

        # gradient = backward pass
        # calculate gradient automatically using pytorch
        # backward calculation is not as exact as calculating numerically
        l.backward() # dl/dw

        # update weight
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        # print data
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

    # in the end, we expect weight = 2
    print(f'prediction after training: f(5) = {model(X_test).item():.3f}')



if __name__ == '__main__':

    # https: // www.youtube.com / watch?v = c36lUUr864M

    # learn_tensor()
    #learn_gradient()
    # learn_back_propagation()
    learn_gradient_descent_torch()