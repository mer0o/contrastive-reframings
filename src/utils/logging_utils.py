import wandb



def encoder_train_log(loss, val_loss, example_count, epoch, is_wandb_watching, is_logging_epoch = False, total_epochs = 25):
    """
    Logs training and validation losses to WandB.

    Args:
        loss (float): Training loss.
        val_loss (float): Validation loss.
        example_count (int): Number of examples seen.
        epoch (int): Current epoch.
    """
    if is_logging_epoch:
        print(f"Epoch {epoch + 1}/{total_epochs}, Average Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        print(['='*10])
        print('\n')
    else:
        if is_wandb_watching:
            wandb.log({
                "epoch": epoch,
                "model_batch_train_loss": loss,
                "model_batch_validation_loss": val_loss,
            })
        print(f"Loss after {str(example_count).zfill(5)} examples: train = {loss:.3f} || validation = {val_loss:.3f}")
        
def log_metrics(metrics, prefix='', log = True, print = True):
    """
    Log the metrics to the console and to Weights & Biases.

    This function logs the metrics to the console and to Weights & Biases
    using the `wandb.log` method.

    Args:
        metrics (dict): A dictionary containing the metrics to log.
        prefix (str): An optional prefix to add to the metric names.
    """

    for key, value in metrics.items():
        key = f'{prefix}/{key}'
        if log:
            wandb.log({key: value})
        if print:
            print(f'{key}: {value}')
            
        
            
def mlp_epoch_log(epoch, total_epochs, epoch_train_loss, validation_metrics, is_wandb_watching):
    """
    Logs the epoch metrics for the MLP training.

    Args:
        epoch (int): The current epoch number.
        tooal_epochs (int): The total number of epochs.
        epoch_train_loss (float): The average training loss for the epoch.
        metrics (dict): Dictionary containing evaluation metrics.
        is_wandb_watching (bool): Whether Weights & Biases is enabled.
    """
    print(f"Epoch {epoch + 1}/{total_epochs}:")
    print(f"  Average Train Loss: {epoch_train_loss:.4f}")
    print(f"  Validation Loss: {validation_metrics['value_loss']:.4f}")
    print(f"  Validation F1: {validation_metrics['value_f1']:.4f}")
    print(f"  Validation Accuracy: {validation_metrics['value_acc']:.4f}")

    if is_wandb_watching:
        wandb.log({
            "mlp_epoch": epoch,
            "mlp_avg_epoch_train_loss": epoch_train_loss,
            "mlp_epoch_validation_loss": validation_metrics['value_loss'],
            "mlp_epoch_validation_f1": validation_metrics['value_f1'],
            "mlp_epoch_validation_accuracy": validation_metrics['value_acc']
        })
            
            
            
def mlp_train_log(train_metrics, batch_count, total_batches, example_count, epoch, is_wandb_watching):
    """
    Logs training and validation losses to WandB.

    Args:
        train_metrics (dict): Dictionary containing training metrics, including the loss value.
        batch_count (int): Current batch number.
        total_batches (int): Total number of batches.
        example_count (int): Number of examples seen.
        epoch (int): Current epoch.
        is_wandb_watching (bool): Whether Weights & Biases is enabled.
    """
    loss = train_metrics['value_loss']
    if is_wandb_watching:
        wandb.log({
            "mlp_batch_train_accuracy": train_metrics['value_acc'],
            "mlp_batch_train_f1": train_metrics['value_f1'],
            "mlp_epoch": epoch,
            "mlp_batch_train_loss": loss,
        })
    print(f"Classification Head train Loss after {str(example_count).zfill(5)} examples, and {batch_count}/{total_batches} batches: {loss:.3f}, Accuracy: {train_metrics['value_acc']:.3f}, F1: {train_metrics['value_f1']:.3f}")