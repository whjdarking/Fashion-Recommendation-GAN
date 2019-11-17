# Dictionary storing network parameters.
params = {
    'batch_size': 64,# Batch size.
    'num_epochs': 10,# Number of epochs to train for.
    'learning_rate': 2e-5,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 5,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'CelebA'}# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!