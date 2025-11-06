import torch
import torch.nn as nn


class myNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out


def main():
    nn_input_size = 2  # Input size (x, y)
    nn_hidden_size = 2  # I am looking for m and b


if __name__ == "__main__":
    main()    # Define a simple neural network model and practrice how to train it using Pytorch
