import numpy as np
from sklearn.metrics import f1_score
from models.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from dataprocessing.pre_process import *
from utils.utils import *
from utils.load_save import *

def train_epoch(model, optimizer, criterion, train_loader, device, deep_supervision=False):
    """This function is used to train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (nn.optim): The optimizer to be used for training.
        criterion (nn.loss): The loss function to be used for training.
        train_loader (Dataloader): The dataloader to be used for training.
        device (torch.device): The device to be used for training.
        deep_supervision (bool, optional): This argument is used to determin if we would be using deepsupervision. Defaults to False.

    Returns:
        loss_history: The training loss history of the model.
        mean_loss: The mean training loss of the model.
    """
    loss_history = []
    total_loss = 0
    num = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        aux = None
        output = None
        loss = 0
        if deep_supervision:
            output, aux = model(data)
            loss_aux = 0
            for d in aux:
                loss_aux += criterion(d, target.float())
            loss = loss_aux/4
        else:
            output, _ = model(data)
            loss = criterion(output.float(), target.float())
        loss_float = loss.item()
        total_loss += loss_float
        num += 1
        loss.backward()
        optimizer.step()
        loss_history.append(loss_float)
    return loss_history, total_loss/num

def train(model, optimizer, scheduler, criterion, data_loader, epochs, device, parameters, deep_supervision=False):
    """This function is used to train the model

    Args:
        model (nn.Module): The model to be trained
        optimizer (nn.optim): The optimizer to be used for training
        scheduler (nn.optim.lrscheduler): The scheduler to be used for training
        criterion (nn.Module): The loss function to be used for training
        data_loader (dict): Dictionary containing the training data and the validation data
        epoch (int): The number of epochs to train the model
        device (torch.device): The device to be used for training
        parameters (dict): multiple parameters to be used for training
        deep_supervision (bool, optional): This parameter will be enabling deep supervision or not. Defaults to False.

    Returns:
        loss_history: The training loss history of the model
        f1_history: The training f1 score history of the model
        f1_valid: The validation f1 score history of the model
        validation_loss_c: The validation loss history of the model
    """
    print("Start training")
    loss_history = []
    valid_f1 = []
    mean_losses = []
    validation_loss_c = []
    
    # We get the parameters from the dictionary
    name = parameters['model_name']
    L = parameters['L']
    C = parameters['C']
    BATCH_SIZE = parameters['BATCH_SIZE']
    path = parameters['path']
    batchNumber = parameters['batch_number']
    rng = parameters['rng']
    light = parameters['light']
    pipeline = parameters['pipeline']
    
    # We get the data from the dictionary
    x_train, y_train = data_loader['x_train'], data_loader['y_train']
    x_test, y_test = data_loader['x_test'], data_loader['y_test']
    del data_loader
    gc.collect()
    

    n = x_train.shape[0]
    max_f1 = 0
    q_epoch = 0
    model.train()
    for epoch in range(epochs):        
        # We then take a random sample so that we can get a batchNumber of batches to train with
        train_loader= None
        xr, yr = None, None
        if light: 
            train_indices = rng.choice(np.arange(n), size=(batchNumber*BATCH_SIZE)//60, replace=False)
            xr, yr, _, _= data_augmentation(x_train[train_indices], y_train[train_indices], None, None, pipeline=pipeline, rng=rng)
        else:
            train_indices = rng.choice(np.arange(n), size=batchNumber*BATCH_SIZE, replace=False)
            xr, yr = x_train[train_indices], y_train[train_indices]
        dataset = CustomDataset(xr, yr)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
        
        print(f"Start Epoch {(epoch + 1)} current max f1 is {max_f1} at epoch {q_epoch}")
        loss_history_ep, Total_loss = train_epoch(model, optimizer, criterion, train_loader, device, deep_supervision) 
        loss_history += loss_history_ep
        mean_losses.append(Total_loss)
        print(f"The mean trainng loss of epoch {epoch + 1} is {Total_loss}")
        output_val, _ = model(x_test)
        validation_loss = criterion(output_val, y_test)
        pred = (output_val > 0.5).float()
        f1_val = f1_score(y_test.cpu().detach().numpy().flatten(), pred.cpu().detach().numpy().flatten())
        valid_f1.append(f1_val)
        validation_loss_c.append(validation_loss.item())
        state = {
                'model_state_dict': model.state_dict(),
                'info': {
                    'epoch':epoch,
                    'model_name': name,
                    'L': L,
                    'C': C,
                    'in_channels': model.n_channels,
                    'out_channels': model.n_classes,
                    'deep_supervision': deep_supervision,
                    'validation_f1': f1_val,
                    'validation_loss': validation_loss.item(),
                    'loss': Total_loss,
                    'lr': optimizer.param_groups[0]['lr']
                }
            }
        max_f1, q_epoch = save_if_max(f1_val, max_f1, epoch, q_epoch, state, path)
        print(f"The validation loss of epoch {(epoch + 1)} is {validation_loss}")
        print(f"The validation f1 of epoch {(epoch + 1)} is {f1_val}")
        scheduler.step(f1_val)
    return loss_history, valid_f1, mean_losses, validation_loss_c