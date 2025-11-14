from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Bryan Arroyo Code ##
import torch.optim as optimizer
from sklearn.metrics import confusion_matrix
import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
            label = sample['label'].unsqueeze(1)
            current_loss = loss_function(output, label)
            current_loss.backward()  # Run backpropagation
            gradient_descent_method.step()  # Move down the hill one step
            group_loss += current_loss.item()
            # debug=True
            # if debug:
            #     print(sample['input'].shape, sample['label'].shape)
            #     debug=False
            #     break

        epoch_loss = group_loss/len(data_loaders.train_loader)
        losses.append(epoch_loss)
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        test_losses.append(test_loss)

        # print(f"Epoch [{epoch_i+1}/{no_epochs}], Loss: {epoch_loss:.6f}")
        # print(f"Epoch [{epoch_i+1}/{no_epochs}], Test Loss: {test_loss:.6f}")
        print(f"{epoch_i},{epoch_loss:.6f},{test_loss:.6f}")

        # Check stop training condition
        num_elements_back=10
        test_loss_subarray = test_losses[-num_elements_back:]
        if epoch_i>1000:
            if stop_training(test_loss_subarray):
                break

    display_training_progress(losses, test_losses)
    cm = compute_confusion_matrix(model, data_loaders.test_loader)
    plot_confusion_matrix(cm)



## Bryan Arroyo Code ##


def display_training_progress(train_losses, test_losses):
    epochs = range(len(train_losses))   # X-axis

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_confusion_matrix(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sample in dataloader:
            inputs = sample['input']
            labels = sample['label']   # labels should be 0 or 1

            outputs = model(inputs)    # outputs should also be 0 or 1

            # Convert output shape [batch,1] → [batch]
            preds = outputs.squeeze().long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm         


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')

    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ['0', '1']
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha='center',
                     va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def display_model_effectiveness():
    pass
# def document(csv_file, epoch, training_loss, testing_loss):


def stop_training(test_error_array):
    if len(test_error_array) < 2:
        return False
    return all(test_error_array[i] > test_error_array[i - 1] for i in range(1, len(test_error_array)))


## Bryan Arroyo Code ##


if __name__ == '__main__':
    ## Bryan Arroyo Code #
    no_epochs = 10000  # Start with a very large training budget
    ## Bryan Arroyo Code ##
    train_model(no_epochs)

    ## Bryan Arroyo Code ##
    # display_training_progress()
    ## Bryan Arroyo Code ##
