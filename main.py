import torch
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

if __name__ == '__main__':

    # https: // www.youtube.com / watch?v = c36lUUr864M

    # learn_tensor()
    #learn_gradient()
    learn_back_propagation()