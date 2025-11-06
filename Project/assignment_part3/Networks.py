import torch
import torch.nn as nn


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()  # Initialize nn module
        # Define layers
        self.input_size = None  # Define input size
        self.hidden_size = None  # Define hidden layer sizes
        self.output_size = None  # Define output size

        self.input_to_hidden = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        ## Input to Hidden Actions ##
        hidden = self.input_to_hidden(input)
        hidden = self.activation(hidden)
        ## Input to Hidden Actions ##

        ## Hidden to Output Actions ##
        hidden = self.hidden_to_output(hidden)
        output = self.activation(hidden)
        ## Hidden to Output Actions ##
        return output

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        return loss


def main():
    model = Action_Conditioned_FF()


if __name__ == '__main__':
    main()
