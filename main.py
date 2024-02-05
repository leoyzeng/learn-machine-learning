import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


def learn_linear_regression():
    # prepare data
    X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) # get data from sklearn

    X = torch.from_numpy(X_numpy.astype(np.float32)) # convert numpy to tensor
    y = torch.from_numpy(Y_numpy.astype(np.float32))

    y = y.view(y.shape[0], 1) # reshape y
    n_samples, n_features = X.shape # n_features is dimension of input

    # model
    input_size = n_features
    output_size = 1
    model = nn.Linear(input_size, output_size)

    # loss and optimizer
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # pass in model shape and learning rate to optimizer

    # training loop
    num_epoch = 100
    for epoch in range(num_epoch):

        # forward pass
        y_predicted = model(X)

        loss = criterion(y_predicted, y)

        # backward pass
        loss.backward()

        # update
        optimizer.step()

        optimizer.zero_grad() # always reset gradient at the end of each training loop

        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    # plot
    predicted = model(X).detach()
    plt.plot(X_numpy, Y_numpy, 'ro')
    plt.plot(X_numpy, predicted, 'b')
    plt.show()

def learn_logistics_regression():

    ### prepare data
    bc = datasets.load_breast_cancer() # dataset for predicting cancer based on input
    X, y = bc.data, bc.target

    n_samples, n_features = X.shape
    print(X.shape) # 569 samples, 30 features (each input and output is a 30 length array)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

    # scale features, make features have 0 mean, and standard variance
    sc = StandardScaler() # always do this when doing logistics regression
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.view(y_train.shape[0], 1) # convert n x 1 vector to 1 x n vector
    y_test = y_test.view(y_test.shape[0], 1)

    ### model
    # f = wx + b, sigmoid function at the end
    # model is a linear function
    # sigmoid function will return value between 0 and 1

    class LogisticRegression(nn.Module):

        def __init__(self, n_input_features):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_input_features, 1) # 30 inputs, 1 output (output is boolean)

        def forward(self, x):
            y_predicted = torch.sigmoid(self.linear(x))
            return y_predicted

    model = LogisticRegression(n_features)

    ### loss, optimizer

    learning_rate = 0.01
    criterion = nn.BCELoss() # binary cross entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent

    ### training loop
    num_epoch = 100
    for epoch in range(num_epoch):

        # forward pass and loss
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)

        # backward pass
        loss.backward()

        # update
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss = {loss.item():.4f}')

    with torch.no_grad(): # we don't want this to be part of the computation graph, no gradient tracking
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round() # round to 0 or 1
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f'accuracy = {acc:.4f}')

    for i in range(10): # printing out 10 test and outputs
        print(X_test[i])
        print(model(X_test[i]))
        print(y_test[i])


if __name__ == '__main__':

    # https: // www.youtube.com / watch?v = c36lUUr864M

    # learn_tensor()
    # learn_gradient()
    # learn_back_propagation()
    # learn_gradient_descent_torch()
    # learn_linear_regression()
    learn_logistics_regression()