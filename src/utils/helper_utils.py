import torch
import pandas as pd
import numpy as np
import os
import random
import wandb
from torch.utils.data import DataLoader
from data_functions.datasets import LargeTrainDataset, LargeTestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_pairs(thoughts, reframings, num_negatives = 2, shuffle = False):
    """
    Creates both positive and negative pairs and combines them into a single dataset.

    Args:
        thoughts (Tensor): Batch of thought tensors.
        reframings (Tensor): Batch of all concatenated reframings tensors.
        negative_samples_count (int, optional): Number of negative samples per sentence. Default is 2.

    Returns:
        dict: Dictionary with combined positive and negative pairs.
        Tensor: Combined labels tensor.
        Tensor: Tensor of the indices of the positives where the fifth element is the value of the index of the reframing corresponding to the fifth sentnence in the positive pairs.
        Tensor: Tensor of the indices of the negatives where the fifth element is the value of the index of the reframing corresponding to the fifth sentnence in the negative pairs.
    """
    # To avoid mlp affecting encooder weights
    thoughts = thoughts.detach()
    reframings = reframings.detach()

    # Create positive and negative pairs separately
    positive_pairs, positive_labels, pos_indices = create_positive_pairs(thoughts, reframings)
    negative_pairs, negative_labels, neg_indices = create_negative_pairs(thoughts, reframings, num_negatives)

    # Combine positive and negative pairs
    combined_pairs = {
        'thought': torch.cat((positive_pairs['thought'], negative_pairs['thought']), dim=0),
        'reframing': torch.cat((positive_pairs['reframing'], negative_pairs['reframing']), dim=0)
    }
    combined_labels = torch.cat((positive_labels, negative_labels), dim=0).to(device)

    if shuffle:
        # Shuffle the combined pairs and labels
        shuffled_pairs, shuffled_labels, shuffled_indices = shuffle_pairs_and_labels(combined_pairs, combined_labels)
        return shuffled_pairs, shuffled_labels


    return combined_pairs, combined_labels, pos_indices, neg_indices


def create_positive_pairs(thoughts, reframings):
    """
    Creates positive pairs from sentences and reframings. it keeps track
    of the indices of the positives, so we can retrieve the text of
    the pairs later.

    Args:
        thoughts (Tensor): Batch of thoughts tensors.
        reframings (Tensor): Batch of all reframings combined tensors.

    Returns:
        dict: Dictionary with positive pairs.
        Tensor: Tensor of positive labels.

    """
    num_samples = thoughts.shape[0]
    positives_per_thought = reframings.shape[0] // num_samples

    thoughts_repeated = thoughts.repeat(positives_per_thought, 1)

    positive_labels = torch.ones(reframings.shape[0], dtype=torch.long).to(device)
    pos_indices = torch.arange(reframings.shape[0])

    positive_pairs = {
        'thought': thoughts_repeated,
        'reframing': reframings
    }
    return positive_pairs, positive_labels, pos_indices


def create_negative_pairs(thoughts, reframings, negative_samples_count):
    """
    Creates negative pairs by selecting random reframings that are not
    true matches. it keeps track of the indices of the negatives, so we
    can retrieve the text of the pairs later.

    Args:
        thoughts (Tensor): Batch of thoughts tensors.
        reframings (Tensor): Batch of all reframings combined tensors.
        negative_samples_count (int): Number of negative samples per sentence.

    Returns:
        dict: Dictionary with negative pairs.
        Tensor: Tensor of negative labels.
        Tensor: Tensor of the negative indices.
    """
    num_samples = thoughts.shape[0]
    positives_per_thought = reframings.shape[0] // num_samples

    thoughts_repeated = thoughts.repeat(negative_samples_count, 1)

    neg_indices = torch.randint(0, reframings.shape[0], (num_samples * negative_samples_count,))
    true_indices = torch.arange(num_samples).repeat(negative_samples_count)

    # Ensure negative indices won't lead to positive examples
    mask = neg_indices % num_samples == true_indices
    while mask.any():
        neg_indices[mask] = torch.randint(0, reframings.shape[0], (mask.sum().item(),))
        mask = neg_indices % num_samples == true_indices

    negative_reframings = reframings[neg_indices]
    negative_labels = torch.zeros(num_samples * negative_samples_count, dtype=torch.long).to(device)

    negative_pairs = {
        'thought': thoughts_repeated,
        'reframing': negative_reframings
    }
    return negative_pairs, negative_labels , neg_indices


def pair_text(thoughts_text, reframings_text, pos_indices, neg_indices, is_wandb_watching):
  """
  Pairs the sentences text with their positive and negative reframings that
  were created in the negative and positive examples step, will be used to
  visualise the results.
  This sort of traces the steps we did when creating positive and negative
  examples from embeddings in 'create_pairs' function, but this time on the
  text itself.

  Args:
    thoughts_text: The thoughts text.
    reframings_text: The reframings text.
    pos_indices: indices relating each sentence from the positive examples to its reframing.
    neg_indices: indices relating each sentence from the negative examples to its reframing.

  Returns:
    dict: containing each sentence along side its corresponding reframing.

  """
  # Convert to list
  thoughts_text = list(thoughts_text)
  reframings_combined = list(reframings_text)

  negatives_per_sent = len(neg_indices) // len(thoughts_text)

  positive_reframings = [reframings_combined[i] for i in pos_indices]
  negative_reframings = [reframings_combined[i] for i in neg_indices]

  refarmings = positive_reframings + negative_reframings

  thoughts_pos = thoughts_text * 2
  thoughts_neg = thoughts_text * negatives_per_sent
  thoughts = thoughts_pos + thoughts_neg

  pairs = {
      'thought': thoughts,
      'reframing': refarmings
  }
  label = [1] * len(positive_reframings) + [0] * len(negative_reframings)
  pairs['label'] = label
  
  pairs_df = pd.DataFrame(pairs)
  if is_wandb_watching:
    wandb.log({"pairs": wandb.Table(dataframe=pairs_df)})
  
  return pairs

def shuffle_pairs_and_labels(pairs, labels):
        """
        Shuffles pairs and labels together, and returns the shuffled indices
        to keep track of the data.

        Args:
            pairs (dict): Dictionary containing 'thought'nd 'reframing' tensors.
            labels (Tensor): Labels tensor.

        Returns:
            dict: Shuffled pairs dictionary.
            Tensor: Shuffled labels tensor.
        """
        indices = torch.randperm(len(pairs['thought']))
        pairs['thought'] = pairs['thought'][indices]
        pairs['reframing'] = pairs['reframing'][indices]
        labels = labels[indices]
        return pairs, labels, indices
    
    
def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    # Reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(42)
    np.random.seed(42)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)
    return g, seed_worker
    

def make_loader(data, batch_size, seed_worker, g, shuffle = True):
    """
    Create a DataLoader for the given dataset.

    This function creates a PyTorch DataLoader with specified parameters,
    including custom worker initialization for reproducibility.

    Args:
        data (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader object.

    Note:
        This function assumes the existence of a `seed_worker` function and a
        generator `g` for reproducibility, which should be defined elsewhere
        in the code.
    """

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, worker_init_fn=seed_worker,
    generator=g)
    return dataloader


def get_datasets(dataset_path):
    """
    Load the datasets from the specified path.

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    cwd = os.getcwd()
    train_path = os.path.join(cwd, dataset_path, 'train_vectors.csv')
    val_path = os.path.join(cwd, dataset_path, 'valid_vectors.csv')
    test_path = os.path.join(cwd, dataset_path, 'test_vectors.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_dataset = LargeTrainDataset(train_df)
    val_dataset = LargeTestDataset(val_df)
    test_dataset = LargeTestDataset(test_df)

    return train_dataset, val_dataset, test_dataset