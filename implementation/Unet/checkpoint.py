import os
import torch

def save_checkpoint(path_to_model, file_name, model, epoch, optimizer, lr_scheduler, loss_value, verbose=False):
    """
    Save the current training status as a checkpoint

    Input:
    - path_to_model: The location to store the checkpoint
    - model: The model to be checkpointed
    - epoch: Number of epochs this model has been trained
    - optimizer: Optimizer used to optimize the mode
    - criterion: Loss function used to evaluate the model
    - loss_value: Loss value from the last epoch
    - train_acc: Training accuracy from the last epoch
    - val_acc: Validation accuracy from the last epoch
    - verbose: If True print necessary information
    """
    # Create directory if not exist
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_value': loss_value,
                'lr_scheduler_state_dict' : lr_scheduler.state_dict(),
                }, path_to_model+file_name)
    if verbose:
        print('Saved the model to {}'.format(path_to_model))

def load_checkpoint(model, optimizer, lr_scheduler, path_to_model, verbose=True):
    """
    Load the checkpoint from the specified location

    Input:
    - model: The model to be checkpointed
    - optimizer: Optimizer used to optimize the mode
    - criterion: Loss function used to evaluate the model
    - path_to_model: The location to find the checkpoint
    - verbose: If True print necessary information

    Returns:
    - epoch: int storing the number of epoch this model had been trained
    - best_loss: floating number storing the last loss value
    - train_acc: floating number storing the last training accuracy
    - val_acc: floating number storing the last validation accuracy
    """
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler = checkpoint['lr_scheduler_state_dict']
    best_loss = checkpoint['loss_value']
    if verbose:
        print('Loaded checkpoint from {}'.format(path_to_model))
    return epoch, best_loss
