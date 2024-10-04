import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, include_thoughts_as_negatives = True):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.include_thoughts_as_negatives = include_thoughts_as_negatives
        
    def forward(self, thoughts, reframings):
        dim = thoughts.shape[1]
        batch_size = thoughts.shape[0]
        positives_per_thought = reframings.shape[0] // batch_size

        # Combine thoughts and reframings into a single tensor, this is so we can include other thoughts as negatives for each thought
        features = torch.cat([thoughts, reframings], dim=0)
        
        # normalize the vectors to get cosine similarity later by matrix multiplication
        features_norm = nn.functional.normalize(features, dim=1)
        
        # Create the labels this is similar to having a class for each thought and its reframings
        # later we will construct the targets/poisitve masks by matching the labels
        labels = torch.arange(batch_size).repeat(positives_per_thought+1).to(device)
        
        # Create mask for removing self similarities, this mask will exclude them from calculations
        N = batch_size * (positives_per_thought+1) # this to inchude the thoughts and reframings
        mask = torch.eye(N, device=device).bool()
        
        # Remove thoughts indices from calculations if we want to only compare thoughts to reframings in training
        if not self.include_thoughts_as_negatives:
            # Exclude thought-to-thought comparisons
            thoughts_indices = torch.arange(batch_size)
            thoughts_mask = torch.zeros_like(mask)
            thoughts_mask[thoughts_indices.unsqueeze(1), thoughts_indices.unsqueeze(0)] = True
            mask = mask | thoughts_mask
        
        
        # calculate the cosine similarity, and apply the temperature scaling
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # Adjust the similarities for numerical stability, later on when using logarithms.
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Remove excluded similarities we decided before in the mask, and then get the log probability
        ## This calculation uses the log identitiy log(a/b) = log(a) - log(b), and log(exp(a)) = a
        ## the nominator of the loss function is thus just the similarity of the positive examples (logits)
        logits_masked = logits.masked_fill(mask, float('-inf')) # as exp(-inf) = 0
        log_prob = logits - torch.logsumexp(logits_masked, dim=1, keepdim=True)
        

        # Define the targets
        ## The targets represent the location of positive examples in the logits matrix as ones, and zeros elsewhere
        ## We will use the previously created labels to find areas of agreement. Note that currently targets will remain
        ## a boolean with the same shape as the logits matrix, but we will use it as a positives_mask later on.
        labels = labels.unsqueeze(0)
        targets = labels == labels.T
        targets = targets & ~mask # remove the excluded similarities


        # This will get us the sum of the positives only for each thought, and then we nomralise by the number of positives
        mean_log_prob_pos = (targets.float() * log_prob).sum(dim=1) / targets.float().sum(dim=1)

        # The loss is the sum of previous operations across the batch
        # Here we use the mean instead of sum as we don't want the batch size to affect the loss and gradients-
        #  magnitude, this helps with numerical stability and is common practice,
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss
