import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from project.src.utils.logging_utils import encoder_train_log, mlp_train_log, mlp_epoch_log, plot_predictions
from project.src.models.models import device


def train_model(model, dataloader, val_loader, config, is_wandb_watching, device = 'cpu'):
    """
    Trains the given model using the provided data loader, criterion, and optimizer.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        config (Namespace): Configuration object with training parameters.

    Returns:
        model (torch.nn.Module): The trained model.
        mlp_evaluator (ClassificationHead): The trained evaluator head.
    """

    model.to(device)
    model.train()
    if is_wandb_watching:
        wandb.watch(model, model.loss, log='all', log_freq=1)

    early_stopping = EarlyStopping(patience=config['patience'], delta=config['delta'])

    example_count = 0  # number of total examples seen
    batch_count = 0 # number of batches gone through
    
    # Later on we will plot the progress of embedding space improvement by plotting
    # reduced dimensions representation using umap, so here we initialise the vectors
    # to be populated later.
    umap_df = None

    for epoch in tqdm(range(config['epochs'])):
        total_train_loss = 0

        for batch in dataloader:
            model.train()

            loss = model.training_step(batch)
            total_train_loss += loss
            
            # calculate the validation loss
            val_loss = model.validation_step(val_loader)
        
            example_count += len(batch[0])
            batch_count += 1

            # Log Metrics
            if batch_count % 2 == 0:
                encoder_train_log(loss, val_loss, example_count, epoch)

        # Average losses
        avg_train_loss = total_train_loss / len(dataloader)
        val_loss = model.validation_step(val_loader)
        """# TODO: Add the UMAP logging function
        epoch_umap_df['epoch'] = [epoch] * len(epoch_umap_df)
        plot_umap_representation(epoch_umap_df, title = f"UMAP Representation of the Validation Embedding Space", show = False)
        
        
        
        # Concatenate the umap vectors to the previous ones if found
        if umap_df is not None:
            umap_df = pd.concat([umap_df, epoch_umap_df], axis=0)
        else:
            umap_df = epoch_umap_df
            
        if epoch % 5 == 0:
            log_umap_validation(epoch, val_loader, model) #TODO: Implement this function"""
            

        # lr_scheduler.step()
        encoder_train_log(avg_train_loss, val_loss, example_count, epoch, is_logging_epoch = True)

        early_stopping(epoch, val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {config['patience']} epochs of no improvement.")
            break


    # UMAP
    ## We will plot the umap vectors to see the improvement of the embedding space, and we will log the
    # dataframe containing all the details.
    """if is_wandb_watching:
        wandb.log({"epochs_validation_set_umap_coordinates": wandb.Table(dataframe=umap_df)})
    
    plot_umap_representation(
        umap_df,
        title = "UMAP Representation of the Validation Embedding Space Throughout Training",
        )
    """ # TODO: Implement this function

    # Get the best model and evaluator
    # Load the best model checkpoint
    current_dir = os.getcwd()
    
    best_model_path = os.path.join(current_dir, config['best_model_path'])
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if is_wandb_watching:
        best_model_artifact = wandb.Artifact('best_model_checkpoint', type='model')
        best_model_artifact.add_file(best_model_path)
        wandb.log_artifact(best_model_artifact)

    return model


def train_mlp(model, mlp_evaluator, train_loader, val_loader, config, is_wandb_watching, device = 'cpu'):
    """
    Trains the MLP classification head, after the encoder is trained.

    Args:
        model (torch.nn.Module): The model with the encoder
        mlp_evaluator (ClassificationHead): the classification head to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function, default is BCEWithLogitsLoss.
        config (Namespace): Configuration object with training parameters.

    Returns:
        ClassificationHead: The trained evaluator head.
    """
    mlp_evaluator.to(device)
    criterion = mlp_evaluator.loss
    if is_wandb_watching:
        wandb.watch(mlp_evaluator, criterion, log='all', log_freq=3)
    
    early_stopping = EarlyStopping(is_mlp = True ,patience=config['patience'], delta=config['delta'], best_mlp_path=config['best_mlp_path'])

    total_batches = len(train_loader) * config['epochs']
    example_count = 0  # number of total examples seen during training
    
    # Initialise empty tensors to collect all predictions and labels to plot them later
    ## for training
    train_epoch_idx = np.array([])
    train_predictions = np.array([])
    train_labels = np.array([])
    train_acc_history = np.array([])
    ## for validation
    validation_epoch_idx = np.array([])
    val_sentence_text = []
    val_reframing_text = []
    validation_predictions = np.array([]) 
    validation_labels = np.array([])
    val_acc_history = np.array([])

    
    for epoch in tqdm(range(config['mlp_epochs'])):
        epoch_train_loss = 0
        
        for batch_count, batch in enumerate(train_loader):
            mlp_evaluator.train()
            
           # Unpack the batch
            thoughts_embeddings = batch[0].to(device) 
            reframings_embeddings = torch.cat(batch[1:], dim = 0).to(device)
                
            positives_per_thought = reframings_embeddings.shape[0]//thoughts_embeddings.shape[0]
            
            # Encoding inputs
            model.eval() # to skip projection in the encoder model
            with torch.no_grad(): # to avoid encoder gradients in mlp
                thoughts_encoded = model(thoughts_embeddings)
                reframings_encoded = model(reframings_embeddings)
                
            # Training Step and loss calculation
            train_batch_metrics, train_batch_predictions, train_batch_labels  = mlp_evaluator.training_step(thoughts_encoded, reframings_encoded)
            mlp_train_loss = train_batch_metrics['value_loss']
            
            # accumulate batch's loss
            epoch_train_loss += mlp_train_loss
            
            # Log the predictions for the batch, for plotting the model progress in the end
            num_predictions = len(train_batch_predictions)
            train_epoch_idx = np.append(train_epoch_idx, np.array([epoch] * num_predictions)) 
            train_predictions = np.append(train_predictions, train_batch_predictions.detach().numpy())
            train_labels = np.append(train_labels, train_batch_labels.detach().numpy())
            train_acc_history = np.append(train_acc_history, train_batch_metrics['value_acc'])
              
            # log training step stats
            if batch_count % 3 == 0:
                mlp_train_log(train_batch_metrics,
                              batch_count,
                              total_batches,
                              example_count,
                              epoch
                              )
                plot_predictions(
                train_batch_predictions,
                train_batch_labels,
                title = "Mlp Training Set Steps Predictions History",
                show = False,
                )
                

                
            # increment example counter
            example_count += len(thoughts_embeddings)
            

        # epoch stats
        epoch_train_loss /= len(train_loader)

        # Get the epoch's predictions:
        val_epoch_predictions, val_epoch_labels, validation_epoch_metrics, validation_epoch_text_pairs = mlp_evaluator.on_train_epoch_end(val_loader, model)
        
        validation_loss = validation_epoch_metrics['value_loss']
        f1_score = validation_epoch_metrics['value_f1']
        # Early stopping check
        early_stopping(epoch, -f1_score, mlp_evaluator)
        if early_stopping.early_stop:
            print(f"Early stopping triggered for MLP after {config['patience']} epochs of no improvement.")
            break
        
        # Log epoch's validation prediction to plot them later
        validation_epoch_idx = np.append(validation_epoch_idx, np.array([epoch] * len(val_epoch_predictions)))
        validation_predictions = np.append(validation_predictions, val_epoch_predictions.detach().numpy())
        validation_labels = np.append(validation_labels, val_epoch_labels.detach().numpy())
        val_acc_history = np.append(val_acc_history, validation_epoch_metrics['value_acc'])
        val_sentence_text.extend(validation_epoch_text_pairs['thought'])
        val_reframing_text.extend(validation_epoch_text_pairs['reframing'])
        
        # Log the epoch's metrics
        mlp_epoch_log(epoch, config['mlp_epochs'], epoch_train_loss, validation_epoch_metrics, config['is_wandb_watching'])
        

    train_predictions_data = pd.DataFrame({
        'epoch': train_epoch_idx,
        'predictions': train_predictions,
        'labels': train_labels
    })
    
    validation_predictions_data = pd.DataFrame({
        'epoch': validation_epoch_idx,
        'predictions': validation_predictions,
        'labels': validation_labels
    })
    
    
    if is_wandb_watching:
        # Log the dataframes as artifacts
        train_predictions_artifact = wandb.Table(dataframe=train_predictions_data)
        validation_predictions_artifact = wandb.Table(dataframe=validation_predictions_data)
        wandb.log({"train_predictions": train_predictions_artifact, "validation_predictions": validation_predictions_artifact})
    
    # plot the predictions history
    ## plot training predictions history
    plot_predictions(
        train_predictions,
        train_labels,
        title = "Training set Epochs Predictions History",
        density = False,
        animation_frame = train_epoch_idx,
    )
    
    ## plot the validation set predictions history
    plot_predictions(
        validation_predictions,
        validation_labels,
        title = "Validation Set Epochs Predictions History",
        density = False,
        animation_frame = validation_epoch_idx,
        hover_data = {'thought': val_sentence_text, 'reframing': val_reframing_text}
    )


    # Load the best MLP model
    best_mlp_model_path = config['best_mlp_path']
    checkpoint = torch.load(best_mlp_model_path)
    mlp_evaluator.load_state_dict(checkpoint['model_state_dict'])
    
    if is_wandb_watching:
        best_mlp_artifact = wandb.Artifact('best_mlp_checkpoint', type='model')
        best_mlp_artifact.add_file(best_mlp_model_path)
        wandb.log_artifact(best_mlp_artifact)
            
    # Return the trained mlp_evaluator
    return mlp_evaluator


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, is_mlp = False, patience=7, delta=0, best_model_path = 'models/best_model.pth', best_mlp_path = 'models/best_mlp.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.current_dir = os.getcwd()
        self.is_mlp = is_mlp
        self.best_model_path = best_mlp_path if is_mlp else best_model_path
        self.path = os.path.join(self.current_dir, self.best_model_path)

    def __call__(self, epoch, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(epoch, val_loss, model)

    def save_checkpoint(self, epoch, val_loss, model):
        """
        Saves the model if validation loss has decreased.

        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss.
            model (torch.nn.Module): The model to save.
            path (str): Path to save the model.
        """
        path = self.path
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'validation_loss': val_loss
        }, path)
        
        print(f'Model at epoch{epoch} saved with validation loss of {val_loss:.5f}')
