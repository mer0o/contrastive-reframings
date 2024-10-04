from collections import Counter
from statistics import mean, median
import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


def check_file_exists(rel_path = 'Datasets/reframe_thoughts_dataset/valid.txt'):
    cwd = os.getcwd()
    path = os.path.join(cwd, rel_path)
    return os.path.isfile(path)


def parse_json_file(path):
    """
    Parses a JSON file containing thoughts and reframings. Each line in the file should be a JSON object, 
    where each object represents a thought entry. Each object should have a 'thought' key with a string value.
    Args:
        path (str): The file path to the JSON file.
    Returns:
        list: A list of dictionaries where each dictionary represents a thought entry.
                Each entry should have a 'thought' key with a string value and a 'reframes' key
                with a list of dictionaries, each containing a 'reframe' key with a string value.
    Raises:
        json.JSONDecodeError: If a line in the file is not valid JSON.
    Notes:
        - Skips empty lines in the file.
        - Prints an error message if a line cannot be decoded as JSON.
    """
    with open(path, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Line: {line}")
    return data


def analyse_reframings_count(data, print_stats = False):
    """
    Analyzes the count of reframings in a given dataset of thoughts.
    Args:
        data (list): A list of dictionaries where each dictionary represents a thought entry.
                        Each entry should have a 'thought' key with a string value and a 'reframes' key
                        with a list of dictionaries, each containing a 'reframe' key with a string value.
        print_stats (bool, optional): If True, prints the statistics. Defaults to False.
    Returns:
        dict: A dictionary containing the following statistics:
            - 'total_thoughts': Total number of thoughts.
            - 'min_reframings': Minimum number of reframings per thought.
            - 'max_reframings': Maximum number of reframings per thought.
            - 'mean_reframings': Average number of reframings per thought.
            - 'median_reframings': Median number of reframings per thought.
            - 'reframing_counts_distribution': A dictionary with the distribution of reframing counts.
    """
    reframing_counts = []
    thoughts_data = []

    for entry in data:
        thought = entry.get('thought', '')
        reframes = entry.get('reframes', [])
        reframing_texts = [r.get('reframe', '') for r in reframes]
        reframing_counts.append(len(reframes))
        thoughts_data.append({
            'thought': thought,
            'reframes': reframing_texts
        })

    # Calculate statistics
    num_thoughts = len(thoughts_data)
    min_reframings = min(reframing_counts)
    max_reframings = max(reframing_counts)
    mean_reframings = mean(reframing_counts)
    median_reframings = median(reframing_counts)
    counts_counter = Counter(reframing_counts)

    if print_stats:
        # Print statistics
        print(f"Total number of thoughts: {num_thoughts}")
        print(f"Minimum number of reframings per thought: {min_reframings}")
        print(f"Maximum number of reframings per thought: {max_reframings}")
        print(f"Average number of reframings per thought: {mean_reframings:.2f}")
        print(f"Median number of reframings per thought: {median_reframings}")
        print("Reframing counts distribution:")
        for count, num in sorted(counts_counter.items()):
            print(f"  {count} reframings: {num} thoughts"
                )
    stats = {
        'total_thoughts': num_thoughts,
        'min_reframings': min_reframings,
        'max_reframings': max_reframings,
        'mean_reframings': mean_reframings,
        'median_reframings': median_reframings,
        'reframing_counts_distribution': dict(counts_counter)
    }
    return stats


def create_reframing_df(data, desired_num_reframings = 3):
    """
    Creates a DataFrame containing thoughts and their corresponding reframings.
    This function processes a list of dictionaries, each representing an entry with a 
    'thought' and a list of 'reframes'. It filters entries to include only those with 
    the specified number of reframings and constructs a DataFrame with the thought and 
    each reframe as separate columns.
    Args:
        data (list): A list of dictionaries, where each dictionary contains a 'thought' 
                        (str) and optionally a list of 'reframes' (list of dicts).
        desired_num_reframings (int, optional): The number of reframings required for 
                                                an entry to be included in the DataFrame. 
                                                Defaults to 3.
    Returns:
        pandas.DataFrame: A DataFrame with columns for the thought and each reframe, 
                            containing only entries with the specified number of reframings.
    Example:
        data = [
            {
                'thought': 'I am not good at my job.',
                'reframes': [
                    {'reframe': 'I am still learning and improving.'},
                    {'reframe': 'Everyone makes mistakes.'},
                    {'reframe': 'I have received positive feedback before.'}
                ]
            },
            {
                'thought': 'I will never finish this project.',
                'reframes': [
                    {'reframe': 'I can break it down into smaller tasks.'},
                    {'reframe': 'I have completed difficult projects before.'}
                ]
            }
        ]
        df = create_reframing_df(data, desired_num_reframings=3)
        print(df)
    """
    rows = []
    for entry in data:
        thought = entry['thought']
        reframes = entry.get('reframes', [])
        # Only include thoughts with the desired number of reframings
        if len(reframes) == desired_num_reframings:
            reframes = reframes[:desired_num_reframings]
            row = {'thought': thought}
            for i in range(desired_num_reframings):
                reframe_text = reframes[i]['reframe']  # Extract only the 'reframe' text
                row[f'reframing{i+1}'] = reframe_text
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Number of thoughts included: {len(df)}")
    return df


def json_to_csv(input_path, output_path, desired_num_reframings = 3):
    """
    Converts a JSON file to a CSV file with a specified number of reframings, in the format that fits our training.
    Args:
        input_path (str): The path to the input JSON file.
        output_path (str): The path to the output CSV file.
        desired_num_reframings (int, optional): The desired number of reframings to include in the CSV. Defaults to 3.
    Returns:
        None
    Side Effects:
        - Parses the JSON file at the given input path.
        - Analyzes the reframings count and prints statistics.
        - Creates a DataFrame with the specified number of reframings.
        - Saves the DataFrame to a CSV file at the given output path.
        - Prints a message indicating the location of the saved CSV file.
    """
    data = parse_json_file(input_path)
    stats = analyse_reframings_count(data, print_stats=True)
    data_df = create_reframing_df(data, desired_num_reframings=desired_num_reframings)
    data_df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def csv_text_to_embeddings_csv(csv_path, text_model, output_path):
    data_df = pd.read_csv(csv_path)

    # Extract texts
    thought_texts = data_df['thought'].tolist()
    reframing1_texts = data_df['reframing1'].tolist()
    reframing2_texts = data_df['reframing2'].tolist()
    reframing3_texts = data_df['reframing3'].tolist()

    # Generate embeddings
    thought_embeddings = text_model.encode(thought_texts, convert_to_tensor=False).tolist()
    reframing1_embeddings = text_model.encode(reframing1_texts, convert_to_tensor=False).tolist()
    reframing2_embeddings = text_model.encode(reframing2_texts, convert_to_tensor=False).tolist()
    reframing3_embeddings = text_model.encode(reframing3_texts, convert_to_tensor=False).tolist()

    # Add embeddings to DataFrame
    data_df['thought_embedding'] = thought_embeddings
    data_df['reframing1_embedding'] = reframing1_embeddings
    data_df['reframing2_embedding'] = reframing2_embeddings
    data_df['reframing3_embedding'] = reframing3_embeddings
    
    # Reorder columns
    data_df = data_df[['thought',  'thought_embedding', 'reframing1', 'reframing1_embedding', 'reframing2', 'reframing2_embedding', 'reframing3', 'reframing3_embedding']]

    # Save to new CSV
    data_df.to_csv(output_path, index=False)
    print(f"Embedded data saved to {output_path}")


if __name__ == '__main__':
    text_model = SentenceTransformer('all-mpnet-base-v2')
    
    input_train_path = 'Datasets/reframe_thoughts_dataset/train_text.csv'
    output_train_path = 'Datasets/reframe_thoughts_dataset/train_vectors.csv'
    csv_text_to_embeddings_csv(input_train_path, text_model, output_train_path)
    
    input_valid_path = 'Datasets/reframe_thoughts_dataset/valid_text.csv'
    output_valid_path = 'Datasets/reframe_thoughts_dataset/valid_vectors.csv'
    csv_text_to_embeddings_csv(input_valid_path, text_model, output_valid_path)
    
    input_test_path = 'Datasets/reframe_thoughts_dataset/test_text.csv'
    output_test_path = 'Datasets/reframe_thoughts_dataset/test_vectors.csv'
    csv_text_to_embeddings_csv(input_test_path, text_model, output_test_path)
    
    



