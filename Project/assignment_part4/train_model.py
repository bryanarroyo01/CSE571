from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Bryan Arroyo Code ##
import torch.optim as optimizer
## Bryan Arroyo Code ##


def train_model(no_epochs):
    ## Bryan Arroyo Code ##
    batch_size = 16  # Following the convention stablished in Assignment Part 2 & 3
## Bryan Arroyo Code ##

    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

## Bryan Arroyo Code ##
    # Provide Loss Function Def -> This is one of the parameters we might need to change to improve the model
    loss_function = nn.MSELoss()
    # We can use momentum if needed in SGD
    gradient_descent_method = optimizer.SGD(model.parameters(), lr=0.01)

## Bryan Arroyo Code ##

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    test_losses = []
    test_losses.append(min_loss)

## Bryan Arroyo Code ##
# We will use early stopping strategy. This will be a long training process
# Plan:
# - Calculate loss against training dataset (L_training)
# - Modify weights
# - Swith model to inference mode
# - Calculate current error against test dataset (E_test)
# - Document current E_test vs past E_test
# - Decide if go for another training round or stop here.
# - Repeat (if applicable)

    for epoch_i in range(no_epochs):
        model.train()
        group_loss = 0
        epoch_loss = 0
        # sample['input'] and sample['label']
        for idx, sample in enumerate(data_loaders.train_loader):
            gradient_descent_method.zero_grad()  # Clear buffers
            output = model(sample['input'])
            current_loss = loss_function(output, sample['label'])
            current_loss.backward()  # Run backpropagation
            gradient_descent_method.step()  # Move down the hill one step
            group_loss += current_loss.item()
        epoch_loss = group_loss/len(data_loaders.train_loader)
        losses.append(epoch_loss)
        test_losses.append(model.evaluate(
            model, data_loaders.test_loader, loss_function))

        print(f"Epoch [{epoch_i+1}/{no_epochs}], Loss: {epoch_loss:.6f}")

        # Check stop training condition
        if stop_training(test_losses,):
            break
    display_training_progress(min(epoch_i, no_epochs), losses, test_losses)


## Bryan Arroyo Code ##


def display_training_progress(x_max, function_1, function_2):
    fig = plt.subplots()
    ax1 = plt.subplots()
    X = range(1, x_max+1)
    # Training Loss
    ax1.plot(X, function_1, 'b-', label='Model Loss')
    ax1.set_xlabel('Training epoch')
    ax1.set_ylabel('Training Loss')
    ax1.tick_params('y', colors='b')
    # Test Loss
    ax2 = ax1.twinx()
    ax2.plot(X, function_2, label='Test Loss')
    ax2.set_ylabel('Test Loss', color='r')
    ax2.tick_params('y', colors='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title('Model Loss')
    plt.grid(True)
    plt.show()


def display_model_effectiveness():
    pass


def stop_training(test_error_array):
    if len(test_error_array) < 3:
        return False
    sub_array = test_error_array[-3:]
    # If true, this means we have consistently increased model errors against test dataset en the last 3 epochs
    return sub_array[0] < sub_array[1] < sub_array[2]

## Bryan Arroyo Code ##


if __name__ == '__main__':
    ## Bryan Arroyo Code #
    no_epochs = 10000  # Start with a very large training budget
    ## Bryan Arroyo Code ##
    train_model(no_epochs)

    ## Bryan Arroyo Code ##
    display_training_progress()
    ## Bryan Arroyo Code ##
