import os
# import ast
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


def get_embeddings_from_sentences(sentences, text_model_name = 'all-mpnet-base-v2'):
    model = SentenceTransformer(text_model_name)
    embeddings = model.encode(sentences)
    return embeddings


def csv_file_to_embeddings(input_csv_path, text_model_name = 'all-mpnet-base-v2', download = False, output_csv_path = None):
    """
    Reads text data from a csv file, and returns the embedded data, 
    and downloads them if necessary. This function ensures compatability of the 
    data format with the needed format for the model later on.

    Args:
    input_csv_path (str): path to the input csv file.
    text_model_name (str): name of the text model to be used.
    download (bool, optional): whether to download the embeddings, defaults to False.
    output_csv_path (str, optional): path to the output csv file, defaults to None.

    Returns:
    Dataframe: containing the embeddings.
    """

    raw_data_df = pd.read_csv(input_csv_path)

    # Organize data into sentences and reframings
    organised_data_df = pd.DataFrame({
        'sentence': raw_data_df.iloc[::2, 1].values,
        'reframing1': raw_data_df.iloc[::2, 2].values,
        'reframing2': raw_data_df.iloc[1::2, 2].values
    })

    # Initialize text model
    text_model = SentenceTransformer(text_model_name)

    # Compute embeddings
    sentence_embeddings = text_model.encode(organised_data_df['sentence'].tolist(), convert_to_tensor=False)
    reframing1_embeddings = text_model.encode(organised_data_df['reframing1'].tolist(), convert_to_tensor=False)
    reframing2_embeddings = text_model.encode(organised_data_df['reframing2'].tolist(), convert_to_tensor=False)

    # Create embedded dataframe
    embedded_data_df = pd.DataFrame({
        'sentence': organised_data_df['sentence'],
        'sentence_embedding': sentence_embeddings.tolist(),

        'reframing1': organised_data_df['reframing1'],
        'reframing1_embedding': reframing1_embeddings.tolist(),

        'reframing2': organised_data_df['reframing2'],
        'reframing2_embedding': reframing2_embeddings.tolist()
    })

    if download:
        # Save to CSV
        embedded_data_df.to_csv(output_csv_path, index=False)
        print(f"Embedded data saved to {output_csv_path}")

    return embedded_data_df


def split_data(data_df, dataset_folder_path = None, validation_size = 0.05, test_size = 0.15, random_state = 42):
    """
    Splits the embedded data into training, validation, and test sets, and downloads them as CSV files.

    This function performs the following operations:
    1. Splits the input dataframe into 80% training and 20% temporary data.
    2. Further splits the temporary data into 40% validation and 60% test data.
    3. Saves the resulting datasets as CSV files.
    4. Prints the size of each dataset.

    Args:
        data_df (pandas.DataFrame): The input dataframe containing embedded data.
        dataset_folder_path (str, optional): The path to the folder where the CSV files will be saved defaults to None.
        validation_size (float, optional): The size of the validation set, defaults to 0.05.
        test_size (float, optional): The size of the test set, defaults to 0.15.
        random_state (int, optional): The random state for the split, defaults to 42.

    Returns:
        tuple: A tuple containing three pandas DataFrames (train_df, val_df, test_df).

    Side effects:
        - Creates 'train_vectors.csv', 'val_vectors.csv', and 'test_vectors.csv' files.
        - Prints the size of each dataset to the console.
    """
    # Calculate the two splits
    first_split = validation_size + test_size
    second_split = validation_size / first_split

    # Split data
    train_df, temp_df = train_test_split(data_df,
                                         test_size=first_split,
                                         random_state=random_state
                                         )
    val_df, test_df = train_test_split(temp_df,
                                       test_size=second_split,
                                       random_state=random_state
                                       )

    # Save datasets
    train_df.to_csv(os.path.join(dataset_folder_path, 'train_vectors.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_folder_path, 'val_vectors.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_folder_path, 'test_vectors.csv'), index=False)

    print(f"Datasets saved to {dataset_folder_path}")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, val_df, test_df