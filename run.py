import numpy as np
import torch
import torch.optim as optim
import sys
import argparse

from utils.utils import *
from dataprocessing.pre_process import *
from utils.tf_aerial_images import *
from scripts.train import *
from models.UNet import UNet
from losses.DiceLoss import DiceLoss
from models.NestedUNet import NestedUNet
from losses.IoULoss import IoULoss

if __name__ == "__main__":
    #We start by parsing the arguments
    parser = argparse.ArgumentParser(description='Train parameters')
    #We add the arguments to the parser
    parser.add_argument('--name', type=str, default='unet', help='Model to be used for training', choices=['unet', 'nestedunet'])
    parser.add_argument('--C', type=int, default=64, help='Number of base channels')
    parser.add_argument('--L', type=int, default=2, help='Number of sequential blocks in a level convultions')
    parser.add_argument('--device', type=str, default='cpu', help='Device to be used for training', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_number', type=int, default=150, help='Number of batches for training per epochs')
    parser.add_argument('--deep_supervision', type=bool, default=False, help='Use deep supervision for training')
    parser.add_argument('--path', type=str, default="", help='Path where to save the trained model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--light', type=bool, default=False, help='Light training')
    args = parser.parse_args()
    model_name =  args.name
    
    # We get the arguments from the parser
    #The model's base channels
    C = args.C
    #The number of sequential blocks in a level convultions
    L = args.L
    #Specify if we want a light training
    light = args.light
    #We setup the device to be used for training
    device = torch.device(args.device)
    #We setup the batch size to be used for training
    BATCH_SIZE = args.batch_size
    #We setup the number of epochs to be used for training
    num_epochs = args.epochs
    #We setup the number of batches to be used for training
    batch_number = args.batch_number
    #We setup the deep supervision to be used for training. Note that deep supervision is only available for NestedUNet
    deep_supervision = args.deep_supervision
    if deep_supervision and model_name == 'unet':
        raise ValueError("Deep supervision is only available for NestedUNet")
    
    # To ensure that the results are reproducible we setup a random generator with the same seed value so that our results are very close to the expected results
    rng = np.random.RandomState(args.seed)
    X = np.transpose(extract_data(f"data/training/images/", num_images=100), (0, 3, 1, 2))
    Y = (extract_data(f"data/training/groundtruth/", num_images=100)).astype(np.float32)
    
    # We split the data into training and testing data
    test_indices = np.array([62, 72, 74, 81])
    train_indices = np.array([i for i in range(100) if i not in test_indices])
    x_train, y_train = X[train_indices], Y[train_indices]
    x_test, y_test = X[test_indices], Y[test_indices]
    
    # We setup the operations that will be used for data augmentation
    main_pipeline = ['resize', 'shift', 'rot', 'flip', 'bright']
    if light:
        pipeline = ['resize']
    else:
        print("Start preprocessing")
        pipeline = main_pipeline
    # We augment the data
    X_train_norm, Y_train_norm, x_test, y_test = data_augmentation(x_train, y_train, x_test, y_test, pipeline=pipeline, rng=rng)
    Y_train_norm = (Y_train_norm > 0.5).astype(np.float32)
    y_test = (y_test > 0.5).astype(np.float32)
    x_test = torch.tensor(x_test).to(device)
    y_test = torch.tensor(y_test).to(device)
    
    in_channels = 3
    out_channels = 1
    model = None
    # We setup the parameters that will be used for training
    params = {'model_name': model_name, 
              'L': L, 
              'C': C, 
              'BATCH_SIZE': BATCH_SIZE, 
              'batch_number': batch_number, 
              'rng': rng,
              'validation_size': 4,
              'path': args.path, 
              'light': light,
              'pipeline': main_pipeline}
    # We keep the data in a dictionary to be used for training and validation
    data_loader = {'x_train': X_train_norm, 
                   'y_train': Y_train_norm,
                   'x_test': x_test,
                   'y_test': y_test}
    # We setup the model to be used for training
    match model_name:
        case 'unet':
            model = UNet(in_channels, out_channels, L=L, C=C).to(device)
        case 'nestedunet':
            model = NestedUNet(in_channels, out_channels, C=C, L=L).to(device)
    
    LR = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = DiceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    loss_history, valid_f1, mean_losses, validation_loss_c  = train(
        model=model,
        optimizer=optimizer, 
        scheduler=scheduler, 
        criterion=criterion, 
        data_loader=data_loader, 
        epochs=num_epochs, 
        device=device, 
        parameters=params,
        deep_supervision=deep_supervision
        )