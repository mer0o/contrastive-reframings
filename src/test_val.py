import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import wandb

from utils.metrics_utils import evaluate_thresholds, plot_predictions
from utils.helper_utils import pair_text

def validate_mlp(encoder_model, mlp_evaluator, val_loader, is_wandb_watching, device = 'cpu'):
    """
    Validates the MLP model on the validation set. Compares and plots classification head predictions vs plain cosine similarity predictions.
    Args:
        encoder_model (torch.nn.Module): The encoder model used for encoding sentences and reframings.
        mlp_evaluator (MLPEvaluator): The MLP evaluator object.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
    Returns:
        tuple: A tuple containing the threshold, average loss, and results dictionary with all sentences.
    Raises:
        None
    """
    
    # freeze the encoder, and skip the projection
    encoder_model.eval()
    
    # Put classification head in evaluation mode
    mlp_evaluator.eval()
    
    num_batches = len(val_loader)
    total_loss = 0
    results = {
        'batch_number': [],
        'thought': [],
        'reframing': [],
        'prediction': [],
        'cos_sim_encoded': [],
        'cos_sim_baseline': [],
        'label': [],
        'correct': []
    }
    
    # iterate over the validation loader
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):            
            # Get the datasets embeddings for calculation, and text for visualisation
            # Unpack the batch
            text, embeddings = batch
            
            thoughts_text = list(text[0])
            reframings_text = [reframing for reframings_set in text[1:] for reframing in reframings_set]

            
            thoughts_embeddings = embeddings[0].to(device)
            reframings_embeddings = torch.cat(embeddings[1:], dim = 0).to(device)

            
            positives_per_thought = reframings_embeddings.shape[0]//thoughts_embeddings.shape[0]
            
            # getting baselilne cosine similarity (without the encoder)
            batch_baseline = thoughts_embeddings, reframings_embeddings
            _, predictions_cos_baseline, _, _, _, cos_baseline_metrics = mlp_evaluator.test_val_step(thoughts_embeddings, reframings_embeddings, cos_sim = True)
            
            # Passing sentences and reframings through the encoder
            thoughts_encoded = encoder_model(thoughts_embeddings)
            reframings_encoded = encoder_model(reframings_embeddings)
            
            batch_encoded = thoughts_encoded, reframings_encoded
            
            # Get the predictions and calculate the loss
            _, predictions, labels, pos_indices, neg_indices, metrics = mlp_evaluator.test_val_step(thoughts_encoded, reframings_encoded)
            
            # Get the plain cosine similarity predictions, in order to evaluate the classification head improvement.
            _, predictions_cos, _, _, _, cos_metrics = mlp_evaluator.test_val_step(thoughts_encoded, reframings_encoded, cos_sim = True)
            
            # Match the sentences text with the predictions.
            text_pairs = pair_text(
                thoughts_text,
                reframings_text,
                pos_indices,
                neg_indices
            )
            
            # Record the batch predictions
            results['batch_number'].extend([batch_idx] * len(predictions))
            results['thought'].extend(text_pairs['thought'])
            results['reframing'].extend(text_pairs['reframing'])
            results['prediction'].extend(predictions)
            results['cos_sim_encoded'].extend(predictions_cos)
            results['cos_sim_baseline'].extend(predictions_cos_baseline)
            results['label'].extend(labels)
            results['correct'].extend(torch.round(predictions) == labels) # assumes a threshold of 0.5
            
    # Calculate the average loss
    avg_loss = total_loss / num_batches

    # Get the metrics for the validation set
    all_predictions = torch.Tensor(results['prediction'])
    all_cos_sim = torch.Tensor(results['cos_sim_encoded'])
    all_labels = torch.Tensor(results['label'])
    all_cos_sim_baseline = results['cos_sim_baseline']
    
    metrics = mlp_evaluator.evaluate_predictions(all_predictions, all_labels, from_logits = False, divide_cm = True)
    cos_metrics = mlp_evaluator.evaluate_predictions(all_cos_sim, all_labels, from_logits = False, divide_cm = True)
    
    
    # Print the metrics in a table format in comparison to the cosine similarity metrics
    print("Validation Metrics:")
    print("{:<15} {:<15} {:<15}".format("Metric", "MLP", "Cosine Similarity"))
    print("----------------------------------------")
    for metric in metrics.keys():
        print("{:<15} {:<15.4f} {:<15.4f}".format(metric, metrics[metric], cos_metrics[metric]))
    

    # Looping on thresholds to identify the optimal threshold if in validation
    thresholds = np.arange(0, 1.0, 0.05)
    threshold, score = evaluate_thresholds(thresholds, all_predictions, all_labels, mlp_evaluator)
    print(f"Best threshold is {threshold}, with F1 score of {score}")

    # Plot predictions to visualise the choice of a threshold
    plot_predictions(
        all_predictions.numpy(),
        all_labels.numpy(),
        "Final Validation Set Predictions, mlp vs cos",
        threshold=threshold,
        cos_sim=all_cos_sim.numpy()
    )

    
    plot_predictions(
        np.array(all_cos_sim_baseline),
        all_labels.numpy(),
        "Baseline vs Encoded Cosine Similarity",
        threshold=threshold,
        cos_sim=all_cos_sim.numpy()
    )
    
    if is_wandb_watching:
        wandb.log({'Validation Final Results & Comparison': wandb.Table(dataframe=pd.DataFrame(results))})
    
    return threshold, avg_loss, metrics['value_f1']



def test(model, mlp_evaluator, test_loader, is_wandb_watching, threshold = 0.5, device = 'cpu'):
    model.eval()

    results = {
        'batch': [],
        'thought': [],
        'reframing': [],
        'prediction': [],
        'label': [],
        'correct': []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Unpack the batch
            text, embeddings = batch
            
            thoughts_text = text[0]
            reframings_text = [reframing for reframings_set in text[1:] for reframing in reframings_set]
            
            thoughts_embeddings = embeddings[0].to(device)
            reframings_embeddings = torch.cat(embeddings[1:], dim = 0).to(device)

            positives_per_thought = reframings_embeddings.shape[0]//thoughts_embeddings.shape[0]

            # Forward pass
            thoughts_encoded = model(thoughts_embeddings)
            reframings_encoded = model(reframings_embeddings)

            # Get the predictions from the mlp
            batch = thoughts_encoded, reframings_encoded
            loss, predictions, labels, pos_indices, neg_indices, batch_metrics = mlp_evaluator.test_val_step(thoughts_encoded, reframings_encoded)

            # Match text with the corresponding predictions, for evaluating model performance.
            text_pairs = pair_text(
                thoughts_text,
                reframings_text,
                pos_indices,
                neg_indices
            )

            # Record the batch predictions, for plotting later
            results['batch'].extend([batch_idx] * len(predictions))
            results['thought'].extend(text_pairs['thought'])
            results['reframing'].extend(text_pairs['reframing'])
            results['prediction'].extend(predictions)
            results['label'].extend(labels)
            results['correct'].extend(torch.round(predictions) == labels)

            print(f"Batch {batch_idx} accuracy: {batch_metrics['value_acc']}, F1: {batch_metrics['value_f1']}")

        # Calculate statistics about the predictions
        total_predictions = torch.tensor(results['prediction'])
        total_labels = torch.tensor(results['label'])
        total_metrics = mlp_evaluator.evaluate_predictions(total_predictions, total_labels, threshold = threshold, from_logits = False)

        final_metrics = {
            "test_avg_f1_score": total_metrics['value_f1'],
            "test_avg_accuracy": total_metrics['value_acc'],
            "test_avg_precision": total_metrics['value_prec'],
            "test_avg_recall": total_metrics['value_recall'],
            "test_avg_loss": total_metrics['value_loss'],
            "test_true_positives": total_metrics['value_confusion'][1,1],
            "test_true_negatives": total_metrics['value_confusion'][0,0],
            "test_false_positives": total_metrics['value_confusion'][0,1],
            "test_false_negatives": total_metrics['value_confusion'][1,0]
        }



        results_df = pd.DataFrame(results)

        print("\n------------–------------–------------–\n")
        print("\n------------–------------–------------–\n")
        print("Final Metrics:")
        print("{:<20} {:<15}".format("Metric", "Value"))
        print("----------------------------------------")
        for metric, value in final_metrics.items():
            print("{:<20} {:<15}".format(metric, value))
        print("\n------------–------------–------------–\n")
        print("\n------------–------------–------------–\n")

        # Log predictions and statistics
        if is_wandb_watching:
            results_table = wandb.Table(dataframe=results_df)
            wandb.log({'test results': results_table})
            wandb.log(final_metrics)
        plot_predictions(
            total_predictions,
            total_labels,
            title = "Final Test Set Predictions",
            show = True,
            threshold = threshold)

    
    if is_wandb_watching:
        wandb.save("model.onnx")
    # torch.onnx.export(model, sentence, "model.onnx") # This is not working locally, so we save the model as a checkpoint, and in wandb