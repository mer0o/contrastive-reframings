import torch
from torch.utils.data import Dataset
import ast

class LargeTrainDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract embeddings
        thought_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['thought_embedding']))
        reframing1_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['reframing1_embedding']))
        reframing2_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['reframing2_embedding']))
        reframing3_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['reframing3_embedding']))

        # Extract texts
        thought_text = self.data.iloc[idx]['thought']
        reframing1_text = self.data.iloc[idx]['reframing1']
        reframing2_text = self.data.iloc[idx]['reframing2']
        reframing3_text = self.data.iloc[idx]['reframing3']

        # Optionally apply transformations (augmentations)
        if self.transform:
            thought_embedding = self.transform(thought_embedding)
            reframing1_embedding = self.transform(reframing1_embedding)
            reframing2_embedding = self.transform(reframing2_embedding)
            reframing3_embedding = self.transform(reframing3_embedding)
            
        
        
        sample = {
            'thought_embedding': thought_embedding,
            'reframing1_embedding': reframing1_embedding,
            'reframing2_embedding': reframing2_embedding,
            'reframing3_embedding': reframing3_embedding,
        }

        if self.transform:
            sample = self.transform(sample)
        
        items = (
                sample['thought_embedding'],
                sample['reframing1_embedding'],
                sample['reframing2_embedding'],
                sample['reframing3_embedding']      
            )

        return items
        

class LargeTestDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract embeddings
        thought_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['thought_embedding']))
        reframing1_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['reframing1_embedding']))
        reframing2_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['reframing2_embedding']))
        reframing3_embedding = torch.tensor(ast.literal_eval(self.data.iloc[idx]['reframing3_embedding']))

        # Extract texts
        thought_text = self.data.iloc[idx]['thought']
        reframing1_text = self.data.iloc[idx]['reframing1']
        reframing2_text = self.data.iloc[idx]['reframing2']
        reframing3_text = self.data.iloc[idx]['reframing3']

        sample = {
            'thought_embedding': thought_embedding,
            'reframing1_embedding': reframing1_embedding,
            'reframing2_embedding': reframing2_embedding,
            'reframing3_embedding': reframing3_embedding
            }

        if self.transform:
            sample = self.transform(sample)

        text = (
            thought_text,
            reframing1_text,
            reframing2_text,
            reframing3_text
        )
        embeddings = (sample['sentence_embedding'],
                    sample['reframing1_embedding'],
                    sample['reframing2_embedding'],
                    sample['reframing3_embedding']
        )

        return text, embeddings
