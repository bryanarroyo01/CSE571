import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(
            self.data)  # fits and transforms
        ## Bryan Arroyo Code ##
        # Currently the scaler is feeding off unbalanced data to the scaler. I decided to leave it as is
        # because I was not sure if this code should be edited.
        # For real ML, consider changing this to balanced train data.##
        # self.normalized_data = self.scaler.fit_transform(
        #     self.balanced_training_dataset)  # fits and transforms
        ## Bryan Arroyo Code ##
        # save to normalize at inference
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        ## Bryan Arroyo Code ##
        return len(self.normalized_data)
        ## Bryan Arroyo Code ##

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        ## Bryan Arroyo ##
        x = np.array(self.normalized_data[idx][:6], dtype=np.float32)
        y = np.float32(self.normalized_data[idx][-1])
        return {'input': x, 'label': y}
        ## Bryan Arroyo ##
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        ## Bryan Arroyo Code ##
        # Separate 80% for training set
        training_set_size = int(0.8 * self.nav_dataset.__len__())
        # Get 20% of the data for test set
        test_set_size = self.nav_dataset.__len__() - training_set_size
        # Set seed for reproducibility during autograding
        rnd_generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.test_dataset = data.random_split(
            self.nav_dataset, [training_set_size, test_set_size], generator=rnd_generator)
        ## Balancing Test Dataset with UnderSampling##
        ## 1) Get training subset indices ##
        train_indeces = np.array(self.train_dataset.indices)
        train_labels = []
        for sample in train_indeces:
            train_labels.append(self.nav_dataset[sample]['label'])
        train_labels = np.array(train_labels)
        ## 2) Count number of samples in each class##
        collision_samples = np.nonzero(np.array(train_labels) == 1)[0]
        collision_indices = train_indeces[collision_samples]
        collision_count = len(collision_indices)

        no_collision_samples = np.nonzero(np.array(train_labels) == 0)[0]
        no_collision_indices = train_indeces[no_collision_samples]
        no_collision_count = len(no_collision_indices)

        ## 3) Determine the minority class##
        minority_class = {
            'class': 1 if min(collision_count, no_collision_count) == collision_count else 0,
            'count': min(collision_count, no_collision_count)
        }

        ## 4) Under-sample the majority class to match the minority class count##
        rng = np.random.default_rng(42)
        if minority_class['class'] == 1:  # Collision is minority class
            selected_collision_indices = collision_indices
            selected_no_collision_indices = rng.choice(
                no_collision_indices, size=minority_class['count'], replace=False)

        else:  # No-Collision is minority class
            selected_no_collision_indices = no_collision_indices
            selected_collision_indices = rng.choice(
                collision_indices, size=minority_class['count'], replace=False)

        ## 5) Combine the indexes of both classes to create a balanced dataset##
        balanced_indeces = np.concatenate(
            (selected_collision_indices, selected_no_collision_indices))
        rng.shuffle(balanced_indeces)  # randomize order
        balanced_dataset = data.Subset(self.nav_dataset, balanced_indeces)

        ## Create DataLoader properties##
        self.train_loader = data.DataLoader(
            balanced_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True)
        ## Bryan Arroyo Code ##


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']


if __name__ == '__main__':
    main()
