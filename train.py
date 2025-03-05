import os
import sys
import yaml
#import wandb
from tqdm import tqdm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from crepe.utils import frequency_to_activation
from crepe.model import CREPE

from dataset import MIR1KDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

from early_stopping import EarlyStopping

def compute_loss(model, audio, labels, samplerate, device, criterion):
    """
    Calculate loss for a given batch.

    Args:
        model: The CREPE model.
        audio (torch.Tensor): torch.Tensor for audio data.
        labels (torch.Tensor): torch.Tensor for label data.
        samplerate: The samplerate of the audio data (ex.16000)
        device: The device (CPU, GPU) to move the tensor data.
        criterion: The criterion like Binary Cross Entropy.
    """
    audio = audio.to(device)
    labels = labels.to(device)

    # calculate label activations
    labels_activations = frequency_to_activation(labels[0, :])

    # calculate model activations
    model_activations = model.get_activation(audio, samplerate).to(device)

    labels_activations = F.interpolate(
            labels_activations.unsqueeze(0).unsqueeze(0), 
            size=(model_activations.shape[0], labels_activations.shape[1]), 
            mode='bilinear',align_corners=False
            ).squeeze(0).squeeze(0)

    loss = criterion(model_activations, labels_activations)
    
    return loss


if __name__=="__main__":

    # load config
    config_path = './configs/config4crepe.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model_name']
    device_conf = config['device']
    runtime_name = config['runtime_name']
    seed = config['seed']
    learning_rate = config['learning_rate']
    num_epoch = config['num_epoch']
    num_batches_per_epoch = config['num_batches_per_epoch']
    max_epochs_without_improvement = config['max_epochs_without_improvement']
    samplerate = config['samplerate']
    model_capacity = config['model_capacity']
    path_save_model = config['path_save_model']
    #path_save_pth = f'{path_save_model}/crepe_{model_capacity}_best.pth'

    print(f"{model_name},\n\
          {device_conf},\n\
          {runtime_name},\n\
          {model_capacity},\n\
          {path_save_model},\n\
          {seed},\n\
          {learning_rate},\n\
          {num_epoch},\n\
          {num_batches_per_epoch},\n\
          {max_epochs_without_improvement},\n\
          {samplerate}\n"
        )

    # setting device
    if device_conf in ['gpu', 'cuda']:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print('GPU can not be recognized.')
            sys.exit(1)
    elif device_conf in ['mps', 'cpu']:
        print('This is a program to train models. \
               The device designation is not cuda.')
        sys.exit(1)
    else:
        print('device missmatch')
        sys.exit(1)

    # setting seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setting wandb
    package_name = 'wandb'
    flag_wandb = False
    try:
        import wandb
        #globals()[package_name] = __import__(package_name)
        print(f"'{package_name}' has been successfully imported.")
        flag_wandb = True
    except ImportError:
        print(f"Warning: '{package_name}' could not be imported. The program will continue.")       

    if flag_wandb == True:
        import wandb
        wandb.init(project=f"{model_name}",
                group=f"{model_name}_{model_capacity}",
                name=f"{runtime_name}",
                config=config)

    # load model
    print("Initializing model ...")
    #model = CREPE(model_capacity=model_capacity).to(device)
    model = CREPE(pretrained_path=path_save_model).to(device)

    # load dataset
    print("Loading Dataset ...")
    mir_1k = MIR1KDataset(root_dir="./data/MIR-1K")
    dataset = ConcatDataset([mir_1k])

    # setting dataloader
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # setting loss func
    criterion = nn.BCELoss()

    # setting optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # setting params for train
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    accumulation_steps = 4

    # early_stopping
    early_stopping = EarlyStopping(
            patience=max_epochs_without_improvement,
            verbose=True,
            delta=0,
            path_save=path_save_model
    )

    # training -------------------------------------------------
    print('[seq] run train')
    for epoch in range(1, num_epoch + 1):
        train_loss = 0.0

        for _ in range(num_batches_per_epoch):
            model.train()
            running_train_loss = 0.0
            for i, (audio, labels) in enumerate(tqdm(train_loader)):
                audio, labels = audio.to(device), labels.to(device)
                
                optimizer.zero_grad()

                # Calc loss
                tmp_train_loss = compute_loss(
                    model=model,
                    audio=audio,
                    labels=labels,
                    samplerate=samplerate,
                    device=device,
                    criterion=criterion
                )
                running_train_loss += tmp_train_loss.item() * audio.size(0)
                
                tmp_train_loss.backward()
                optimizer.step()

        train_loss += running_train_loss / len(train_loader.dataset)
        train_loss /= num_batches_per_epoch

        # for log by wandb.
        if flag_wandb == True:
            wandb.log(
                {"epoch": epoch,
                "train_loss": train_loss,
                }
            )

        # validation -----------------------------------------------
        print('[seq] run val')
        model.eval()
        val_loss = 0.0
        running_val_loss = 0.0

        with torch.no_grad():
            for i, (audio, labels) in enumerate(tqdm(val_loader)):
                audio, labels = audio.to(device), labels.to(device)
                    
                # Calc loss
                tmp_val_loss = compute_loss(
                    model=model,
                    audio=audio,
                    labels=labels,
                    samplerate=samplerate,
                    device=device,
                    criterion=criterion
                )
                running_val_loss += tmp_val_loss.item() * audio.size(0)

        val_loss = running_val_loss / len(val_loader.dataset)

        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # for log by wandb.
        if flag_wandb == True:
            wandb.log(
                {"epoch": epoch,
                "val_loss": val_loss,
                }
            )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
