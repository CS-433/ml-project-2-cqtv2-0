import torch
import torch.nn as nn
import argparse

from utils.load_save import load_checkpoint
from utils.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict parameters')
    parser.add_argument('--path', type=str, help='Path to the model')
    parser.add_argument('--out', type=str, default='submita.csv', help='submission file')
    
    args = parser.parse_args()
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, info = load_checkpoint(args.path)
    model = model.to(device)
    sub = submission_creating(model, path_testing='data/testing/test_set_images/')
    np.savetxt(args.out, sub, delimiter=",", fmt = '%s')
    