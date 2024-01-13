import os
import torch
import torchvision


custom_data_dir = '../MNIST Classifier'

if not os.path.exists(custom_data_dir):
    os.makedirs(custom_data_dir)

n_epochs = 3 #num times iterating through dataset

#size of sets for training and testing
batch_size_train = 64
batch_size_test = 1000

#optimizer hyperparams
learning_rate = 0.01
momentum = 0.5

log_interval = 10

random_seed = 1 #ensures random operations are reproducible
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#DataLoaders. Values 0.1307 and 0.3081 represent the global mean 
#and standard deviation for the MNIST dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(custom_data_dir, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(custom_data_dir, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape