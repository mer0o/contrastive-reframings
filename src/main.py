import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from data_functions.datasets import LargeTrainDataset, LargeTestDataset
from models.models import ClassificationHead, SimSkip
from train import train_mlp, evaluate_model
from utils.helper_utils import set_seed, make_loader, get_datasets
import wandb
from project.src.train import train_model
from test_val import validate_mlp, test

def main():
    # Configuration

    config = dict(
        # Encoder Training
        epochs=25,
        batch_size=32,
        num_workers=0,
        learning_rate=0.001,
        weight_decay = 1e-4,
        temperature = 0.1,
        dropout_rate = 0.1,

        # Early Stopping
        patience = 10,
        delta = 0.01,
        
        # Classification head
        mlp_epochs = 20,
        projection_dim = 128,
        negative_samples_count = 2,
        classifier_output_dim = 2,
        classifier_hidden_dim = 1024,
        classifier_dropout = 0.1,
        classifier_learning_rate = 0.001,

        # Data
        dataset_folder='/Users/mero/Library/Mobile Documents/com~apple~CloudDocs/Documents/Work/Ulm/project/data/reframe_thoughts_dataset', #TODO CHANGE TO YOUR PATH
        dataset_name="reframing_dataset.csv",
        dataset_path = 'Datasets/reframe_thoughts_dataset',
        text_model="all-mpnet-base-v2",
        input_dim = 768,
        downloaded = True,
        
        best_model_path ='models/best_model.pth',
        best_mlp_path ='models/best_mlp.pth',
        
        is_wandb_watching = False,
        colab = False
    )
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    g, seed_worker = set_seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # Initialize wandb
    if config['is_wandb_watching']:
        # Start Tracking
        wandb.init(project="SIM-SKIP Refinement", config = config)



    # Create datasets
    train_dataset, validation_dataset, test_dataset = get_datasets(dataset_path=config['dataset_folder'])

    # Create DataLoaders
    train_loader = make_loader(train_dataset, seed_worker, g, batch_size=config['batch_size'], num_workers = config['num_workers'])
    val_loader = make_loader(validation_dataset, seed_worker, g, batch_size= validation_dataset.__len__(), num_workers = config['num_workers'])
    test_loader = make_loader(test_dataset, seed_worker, g, batch_size=config['batch_size'], num_workers = config['num_workers'])

    
    # Initialize model
    model = SimSkip(
    input_dim=config['input_dim'],
    dropout_rate=config['dropout_rate'],
    projection_dim=config['projection_dim'],
    loss_temperature=config['temperature'],
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay']
    ).to(device)
    
    model = train_model(model, train_loader, val_loader, config, device=device)
    
    
    mlp_evaluator = ClassificationHead(
        input_dim=config['input_dim'],
        is_wandb_watching=config['is_wandb_watching'],
        classifier_hidden_dim=config['hidden_dim'],
        classifier_output_dim=config['classifier_output_dim'],
        negative_samples_count=config['negative_samples_count'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate']
    ).to(device)

    # Train model
    train_mlp(model, mlp_evaluator, train_loader, val_loader, config, device=device)

    # Get Validation stats
    threshold, avg_loss, score = validate_mlp(model, mlp_evaluator, val_loader)
    print(f"Best threshold is {threshold}, with F1 score of {score}, and an mlp cross entropy loss of {avg_loss}")

    # Test the model on downstream tasks
    test(model, mlp_evaluator, test_loader)


    if config['is_wandb_watching']:
        wandb.finish()

if __name__ == '__main__':
    main()
