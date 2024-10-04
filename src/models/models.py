import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import sys
sys.path.append('/Users/mero/Library/Mobile Documents/com~apple~CloudDocs/Documents/Work/Ulm/project')


from src.losses import ContrastiveLoss
from src.utils.helper_utils import create_pairs, pair_text
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimSkipEncoder(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(SimSkipEncoder, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.BatchNorm1d(input_dim // 2, eps = 1e-4),
        nn.ReLU(),
        nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim),
            nn.BatchNorm1d(input_dim, eps = 1e-4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.linear(x)
        return x + identity # The skip connection
    

class SimSkipProjector(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(SimSkipProjector, self).__init__()
        self.layer1 = nn.Linear(input_dim, input_dim)
        self.layer2 = nn.Linear(input_dim, projection_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x) # Adds non linearity to the projection which should perform better.
        x = self.layer2(x)
        return x
    
    


class SimSkip(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1, projection_dim = 128, loss_temperature = 0.1, learning_rate = 0.001, weight_decay = 1e-4):
        super(SimSkip, self).__init__()
        self.encoder = SimSkipEncoder(input_dim, dropout_rate)
        self.projector = SimSkipProjector(input_dim, projection_dim)
        
        self.loss = ContrastiveLoss(temperature=loss_temperature)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                     lr=learning_rate,
                                     weight_decay = weight_decay
                                     )
        #self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])


    def forward(self, x, use_projector = None):
        def forward(self, x, use_projector=None):
            """
            Forward pass through the model.
            Parameters:
            x (torch.Tensor): Input tensor to the model.
            use_projector (bool, optional): Flag to determine whether to use the projector. 
                                            If None, defaults to using the projector during training.
                                            This is mostly when needing to force using the projector while evaluating,
                                            e.g. for loss calculation.
            Returns:
            torch.Tensor: Output tensor after passing through the encoder (and projector if applicable).
            """
        
        if self.training or use_projector:
          x = self.encoder(x)
          x = self.projector(x)
        else:
          # if we are evaluating, we don't need gradients, this also ensures that training the linear classifier doesn't extend to the encoder
          with torch.no_grad():
            x = self.encoder(x)
        return x
      
    
    def training_step(self, batch):
        self.train()

        # Unpack the batch
        thoughts = batch[0].to(device) 
        reframings = torch.cat(batch[1:], dim = 0).to(device)
        
        # Encode the Batch
        thoughts_encoded = self(thoughts)
        reframings_encoded = self(reframings)

        # Forward pass
        self.optimizer.zero_grad()

        loss = self.loss(thoughts_encoded, reframings_encoded)
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
      
    def validation_step(self, validation_loader):
        """
        This functon is the evaluation function for the model, it calculates the loss on the validation set. and 
        returns the loss.
        Args:
            validation_loader (DataLoader): the validation set dataloader.
            model (torch.nn.Module): The encoder model to evaluate.

        Returns:
            _type_: _description_
        """        
        
        sum_loss = 0
        num_batches = len(validation_loader)
        
        for batch in validation_loader:
            text, embeddings = batch
            thoughts, reframings = embeddings[0].to(device), torch.cat(embeddings[1:], dim = 0).to(device)
            # Set the model to training to get the loss after projection, as projection is disabled on evaluation
            self.eval()
            thoughts_encoded = self(thoughts, use_projector=True)
            reframings_encoded = self(reframings, use_projector=True)
            loss = self.loss(thoughts_encoded, reframings_encoded)
            sum_loss += loss

        avg_validation_loss = sum_loss / num_batches
        return avg_validation_loss
    
    
    

class ClassificationHead(nn.Module):
    def __init__(self, 
                 input_dim, 
                 is_wandb_watching,
                 hidden_dim=1024, 
                 output_dim=2, 
                 negative_samples_count= 2,
                 dropout_rate=0.05, 
                 learning_rate=0.01
                 ):
        """
        Initializes the classification head with a simple MLP.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int, optional): Dimension of the hidden layer. Default is 1024.
            output_dim (int, optional): Dimension of the output layer. Default is 1.
            dropout_rate (float, optional): Dropout rate for regularization. Default is 0.1.
            negative_samples_count (int, optional): Number of negative samples per sentence. Default is 2.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 0.01.
        """
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer1 = nn.Linear(2 * input_dim, hidden_dim) # Concatenated sentence and reframing embeddings
        self.layer2 = nn.Linear(hidden_dim, output_dim)


        self.negative_samples_count = negative_samples_count
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.is_wandb_watching = is_wandb_watching

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the MLP.
        """
        x = self.dropout(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def predict(self, sentence1, sentence2, from_logits = True, cos_sim = False):
        """
        Makes predictions by passing concatenated sentence pairs through the model.

        Args:
            sentence1 (Tensor): First sentence tensor.
            sentence2 (Tensor): Second sentence tensor.
            from_logits (bool, optional): Whether to return logits or probabilities. Default is True (logits).
            cos_sim (bool, optional): Whether to use plaincosine similarity as the prediction, or the mlp itself. Default is False (mlp).

        Returns:
            Tensor: Predictions after applying the model.
        """
        sentence1 = sentence1.to(device)
        sentence2 = sentence2.to(device)
        
        if cos_sim:
            prediction = F.cosine_similarity(sentence1, sentence2, dim=1)
            prediction = torch.clamp(prediction, min=0, max=1)
            
        else:
            input = torch.cat([sentence1, sentence2], dim=1)
            prediction = self.forward(input)

            # gets the normalised predictions when not training
            if not from_logits:
                prediction = F.softmax(prediction, dim = 1)[:, 1] # probability of class 1
            
        return prediction


    def training_step(self, thoughts, reframings):
        """
        Defines the logic for a training step in the classification head.

        Args:
            thoughts (Tensor): Encoded thought embeddings.
            reframings (Tensor): Encoded reframing embeddings.

        Returns:
            Loss: (Tensor): Loss value for the current training step.
        """
        self.train()
        
        positives_per_thought = reframings.shape[0] // thoughts.shape[0]
        negative_samples_count = positives_per_thought # We use the same number of negatives as positives to assure balance
        
        pairs, labels, _, _ = create_pairs(thoughts, reframings, num_negatives = negative_samples_count)
        predictions = self.predict(pairs['thought'], pairs['reframing'])
        
        loss = self.loss(predictions, labels)
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Backpropagate the loss
        loss.backward()
        self.optimizer.step()


        with torch.no_grad():
            # Get metrics
            metrics = self.evaluate_predictions(predictions, labels)
            
            # Normalise predictions to probability with softmax
            predictions = F.softmax(predictions, dim = 1)[: , 1] # probability of class 1, this acts as the probability of being true

        return metrics, predictions, labels
    
    
    def test_val_step(self, thoughts_embeddings, reframings_embeddings, cos_sim = False):
        """
        A function used to validate or test the model on a batch it provides insights and statistics about a given batch,
        the stats can be using the mlp itself or just plain cosine similarity, mostly used to compare the two.

        Args:
        thoughts_embeddings: The thoughts embeddings
        reframings_embeddings: The reframings embeddings
        cos_sim: a boolean indicating whether the model degrades to plain cosine similarity
                   usually used as some sort of baseline this is usually opposite to cos sim, as cos sim is 
                   never logits, but always actual probabilities.

        Returns:
          Tensor: loss value
          Tensor: predictions of the model for the pairs
          Tensor: ground truth labels for the pairs
            dict: indices of the positive pairs
            dict: indices of the negative pairs
            dict: evaluation metrics
        """
        positives_per_thought = reframings_embeddings.shape[0]//thoughts_embeddings.shape[0]
        
        pairs, labels, pos_indices, neg_indices = create_pairs(thoughts_embeddings, reframings_embeddings, num_negatives = positives_per_thought)
        
        # Get logits, as we'll calculate loss later, and the loss takes logits
        predictions = self.predict(pairs['thought'], pairs['reframing'], from_logits = True, cos_sim=cos_sim)
        
        # Get prediction metrics from the logits we got for plotting or evaluation.
        metrics = self.evaluate_predictions(predictions, labels, from_logits = not cos_sim)
        loss = metrics['value_loss']
        
        # Convert logits into predictions, cosine similarity doesn't have logits
        if not cos_sim:
            predictions = F.softmax(predictions, dim = 1)[:, 1] # probability of class 1, this acts as the probability of being true
        
        return loss, predictions, labels, pos_indices, neg_indices, metrics

    def on_train_epoch_end(self, val_loader, encoder_model):
        """
        Computes metrics on validation set at the end of a training epoch of the classification head.

        Args:
            val_loader (DataLoader): Validation data loader.
            encoder_model (nn Module): The model to encode the embeddings before predictions

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # put encoder in eval mode, mostly to skip the projection
        encoder_model.eval()
        
        # Initialise empty tensors to collect all predictions and labels to plot them later
        all_predictions = torch.Tensor()
        all_labels = torch.Tensor()
        
        for batch_idx, batch in enumerate(val_loader):
            # Unpack the batch
            text, embeddings = batch
            
            thoughts_text = text[0]
            reframings_text = [reframing 
                            for reframings_set in text[1:] 
                            for reframing in reframings_set
                            ]
            
            thoughts_embeddings = embeddings[0].to(device)
            reframings_embeddings = torch.cat(embeddings[1:], dim = 0).to(device)

            positives_per_thought = reframings_embeddings.shape[0]//thoughts_embeddings.shape[0]

            # Record the predictions of the current training step for plotting later, and get prediction metrics
            loss, predictions, labels, pos_ind, neg_ind, metrics = self.test_val_step(thoughts_embeddings, reframings_embeddings)
            text_pairs = pair_text(thoughts_text, reframings_text, pos_ind, neg_ind, self.is_wandb_watching)
            
            all_predictions = torch.cat((all_predictions, predictions))
            all_labels = torch.cat((all_labels, labels))

        return all_predictions, all_labels, metrics, text_pairs

    def evaluate_predictions(self, predictions, labels, threshold=0.5, from_logits = True, divide_cm = False):
        """
        Evaluates predictions using various metrics.

        Args:
            predictions (Tensor): logits predictions.
            labels (Tensor): True labels.
            threshold (float, optional): Threshold for converting logits to binary predictions. Default is 0.5.
            from_logits (bool, optional): Whether the inputted predictions are logits, defaults to True. If the input is predictions calculated by cosine similarity this will be False as well.
            divide_cm (bool, optional): Whether to divide the confusion matrix into its components. Default is False.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        
        # Specify another loss function just in case the predictions are not logits,
        # but are already probabilities we don't want to use BCEWithLogits but normal BCE
        normalised_loss = nn.BCELoss()
        
        # Get the loss, we have 2 ways, either the predictions are logits then we can use the original one,
        # or they are already normalised to probabilities, in this case we use the already normalised loss.

        if from_logits:
            # calculate the loss from the logits, this will pass inputs to softmax before calculating the entropy
            val_loss = self.loss(predictions, labels).item()
            
            # Normalise predictions to get the prediction probabilities, to calculate metrics later
            predictions = F.softmax(predictions, dim = 1)[:, 1] # probability of class 1, this acts as the probability of being true
        else: 
            # We need to use something other then Cross entropy as here we have aready normalised predictions with softmax
            # Avoid log(0) by adding a small epsilon
            val_loss = nn.BCELoss()(predictions, labels.float()).item()


        # Get predictions as binary values
        inference = (predictions >= threshold).float()

        # Convert PyTorch tensors to NumPy arrays for sklearn metrics
        inference, labels = inference.detach().cpu().numpy(), labels.detach().cpu().numpy()

        # Calculate metrics
        metrics = {
            'value_loss': val_loss,
            'value_f1': f1_score(labels, inference),
            'value_acc': accuracy_score(labels, inference),
            'value_prec': precision_score(labels, inference, zero_division=0),
            'value_recall': recall_score(labels, inference),
            'value_confusion': confusion_matrix(labels, inference)
        }
        if divide_cm:
            metrics = {
            'value_loss': metrics['value_loss'],
            'value_f1': metrics['value_f1'],
            'value_acc': metrics['value_acc'],
            'value_prec': metrics['value_prec'],
            'value_recall': metrics['value_recall'],
            "true_positives": metrics['value_confusion'][1,1],
            "true_negatives": metrics['value_confusion'][0,0],
            "false_positives": metrics['value_confusion'][0,1],
            "false_negatives": metrics['value_confusion'][1,0]
            }
        return metrics