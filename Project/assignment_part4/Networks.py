import torch
import torch.nn as nn

## Bryan Debbug Code ##
# import Data_Loaders


class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=200, output_size=1):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()  # Initialize nn module
        # Define layers
        # Define output size -> Single output either collision or no collision

        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        ## Input to Hidden Actions ##
        hidden_first = self.input_to_hidden(input)
        hidden_second = self.activation(hidden_first)
        ## Hidden to Output Actions ##
        output = self.hidden_to_output(hidden_second)
        return output

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.

        ## Initialize total loss and sample count ##
        total_loss = 0
        sample_count = len(test_loader.dataset)

        model.eval()
        with torch.no_grad():  # Do not waste memory on gradients ##
            for index, sample in enumerate(test_loader):
                input, label = sample['input'], sample['label']
                # Reshape label to match model output shape
                label = label.view(-1, 1)

                model_output = model(input)

                loss = loss_function(model_output, label)
                # total_loss += loss.item()
                total_loss += loss.item()*input.size(0)

        ## Return average loss over all samples ##
        avg_loss = 0 if sample_count == 0 else total_loss / sample_count
        # avg_loss=total_loss/len(test_loader)
        return avg_loss


def main():
    model = Action_Conditioned_FF()

    ## Bryan Debbug Code ##
    # Data_Loaders.main()
    # print("Data Loaders ran successfully.")
    # print(f"Loss evaluation: {model.evaluate(model, Data_Loaders.Data_Loaders(16).test_loader, nn.MSELoss())}   ")


if __name__ == '__main__':
    main()
