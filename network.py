# Cross validation Balance Accuracy = 94.00 %

import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score


TRAIN_DATASET = "dry_bean_train.csv"
TEST_DATASET = "dry_bean_test.csv"
OUTPUT = "network.csv"
MODEL_PATH = "bean_MLP.pt"

BATCH_SIZE = 32
LEARNING_RATE = 0.01
HIDDEN_LAYERS = [64,64]
EPOCHS = 40
NUM_FEATURES = 16
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 0.0001
NUM_BEAN_TYPES = 7
N_FOLDS = 5
SEED = 232134

# Keep results the same every run
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class BeanData(Dataset):

    def __init__(self, inputs, labels):
        self.features = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Store total number of samples
        self.n_samples = len(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


'''
Defining the architecture of MLP model
'''
class MLP(nn.Module):

    def __init__(self, input_size, hidden_layers, num_outputs, activation, dropout_rate):
        super().__init__()

        layers = []
        current_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))

            current_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(current_size, num_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


'''
# Calculates entropy loss to compare prediction result with labels
# Use softmax to convert scores to propabilities
'''
def cross_entropy_loss(scores, y):

    # softmax calculation, subtracting the max first so exp() doesnt blow up
    exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True).values)
    probabilities = exp_scores / exp_scores.sum(dim=1, keepdim=True)

    # Calculate loss for each sample
    batch_size = scores.shape[0]
    total_loss = 0.0

    for i in range(batch_size):
        correct_prob = probabilities[i, y[i]]
        # Avoiding log(0) by adding small epsilon 
        total_loss += -torch.log(correct_prob + 1e-9)

    loss = total_loss / batch_size

    return loss


''' 
load the training csv, convert bean labels to usable numeric values
'''
def load_training_data():

    df = pd.read_csv(TRAIN_DATASET)

    # Get the bean label column
    label_col = "Target" if "Target" in df.columns else df.columns[-1]

    label_names = sorted(df[label_col].unique())

    label_lookup = {}

    for i in range(len(label_names)):
        label_lookup[label_names[i]] = i

    inputs = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    labels = df[label_col].map(label_lookup).to_numpy(copy=True)

    return inputs, labels, label_names


def get_loader(inputs, labels, batch_size, shuffle):

    dataset = BeanData(inputs, labels)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


'''
Randomise rows and split in k chunks for cross validation step
'''
def get_folds(n, k, seed):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        #last fold gets whatever is remaining
        end = start + fold_size if i != k - 1 else n  
        folds.append(indices[start:end])

    return folds


'''
Feature values vary greatly, must be standardized
'''
def standardize(inputs, mean, std):
    return (inputs - mean) / std


'''
Full training run
'''
def train_model(model, train_loader, val_loader, writer, tag, weight_decay):

    # Chose gradient descent as most similar to lecture
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    val_acc = 0.0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        examples_seen = 0

        for input_batch, label_batch in train_loader:
            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device)

            # Clear gradients
            optimizer.zero_grad()

            scores = model(input_batch)
            loss = cross_entropy_loss(scores, label_batch)

            # Update weights 
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label_batch)
            examples_seen += len(label_batch)

        train_loss /= examples_seen
        writer.add_scalar(f"{tag}/train_loss", train_loss, epoch)
        
        if val_loader is None:
            continue

        # Measure performance with the model's current weights
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for input_batch, label_batch in val_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)

                scores = model(input_batch)
                predictions = scores.argmax(dim=1)

                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(label_batch.cpu().numpy())

        val_acc = balanced_accuracy_score(val_labels, val_preds)
        writer.add_scalar(f"{tag}/val_accuracy", val_acc, epoch)

    return val_acc

'''
Train and evaluate model using cross validation
'''
def cross_validate(inputs, labels, dropout_rate, weight_decay, run_name):

    folds = get_folds(len(inputs), N_FOLDS, SEED)

    writer = SummaryWriter(run_name)
    cv_scores = []

    for k in range(N_FOLDS):

        # Fold k is validation, the rest is training
        current_fold = folds[k]
        train_index = []
        for j in range(N_FOLDS):
            if j != k:
                train_index += folds[j]

        # standardise using training rows only
        mean = inputs[train_index].mean(axis=0)
        std = inputs[train_index].std(axis=0)

        train_loader = get_loader(standardize(inputs[train_index], mean, std), labels[train_index], BATCH_SIZE, shuffle=True)
        val_loader = get_loader(standardize(inputs[current_fold], mean, std), labels[current_fold], BATCH_SIZE, shuffle=False)

        model = MLP(NUM_FEATURES, HIDDEN_LAYERS, NUM_BEAN_TYPES, nn.ReLU(), dropout_rate).to(device)
        acc = train_model(model, train_loader, val_loader, writer, f"fold_{k + 1}", weight_decay)

        print(f"Fold {k + 1}: {acc * 100:.2f}%")
        cv_scores.append(acc)

    writer.close()

    avg_accuracy = np.mean(cv_scores) * 100
    print(f"Cross validation Balance Accuracy = {avg_accuracy:.2f} %")

    return avg_accuracy


'''
Train one final time with the settings of the winner of cross validation run (with or without dropout/weight decay)
'''
def train_final(inputs, labels, dropout_rate, weight_decay):

    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0)

    train_loader = get_loader(standardize(inputs, mean, std), labels, BATCH_SIZE, shuffle=True)

    model = MLP(NUM_FEATURES, HIDDEN_LAYERS, NUM_BEAN_TYPES, nn.ReLU(), dropout_rate).to(device)

    writer = SummaryWriter("runs/final")
    train_model(model, train_loader, None, writer, "final", weight_decay)
    writer.close()

    # save model parameters so they can be reloaded later
    torch.save(model.state_dict(), MODEL_PATH)

    return mean, std


'''
Perform forward pass on test set data 
Write each sample's predicted bean type to network.csv
'''
def bean_predictions(model, label_names, mean, std):

    test_df = pd.read_csv(TEST_DATASET)
    test_inputs = standardize(test_df.to_numpy(dtype=np.float32), mean, std)

    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(test_inputs, dtype=torch.float32).to(device))
    preds = scores.argmax(dim=1).cpu().numpy()

    test_df["Target"] = [label_names[p] for p in preds]
    test_df.to_csv(OUTPUT, index=False)
    print(f"Wrote predictions for {len(test_df)} test samples to {OUTPUT}")


def main():

    inputs, labels, label_names = load_training_data()

    # (name, dropout rate, weight decay) for each setup we compare
    settings = [
        ("baseline", 0.0, 0.0),
        ("dropout", DROPOUT_RATE, 0.0),
        ("weight_decay", 0.0, WEIGHT_DECAY),
        ("dropout_weight_decay", DROPOUT_RATE, WEIGHT_DECAY),
    ]

    best_acc = 0.0
    for name, dropout_rate, weight_decay in settings:
        print(f"\n{name}:")
        acc = cross_validate(inputs, labels, dropout_rate, weight_decay, f"runs/{name}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_dropout = dropout_rate
            best_weight_decay = weight_decay

    print(f"\nBest setup: {best_name} ({best_acc:.2f}%)")

    mean, std = train_final(inputs, labels, best_dropout, best_weight_decay)

    # rebuild the network and load the saved parameters back in
    model = MLP(NUM_FEATURES, HIDDEN_LAYERS, NUM_BEAN_TYPES, nn.ReLU(), best_dropout)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    bean_predictions(model, label_names, mean, std)


if __name__ == "__main__":
    main()