# Cross validation Balance Accuracy = 94.23%
# Best setup: {'hidden_layers': [128, 64], 'activation': <class 'torch.nn.modules.activation.ReLU'>, 'dropout_rate': 0.2, 'lr': 0.01, 'weight_decay': 0.0001} (94.23%)
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
EPOCHS = 40
NUM_FEATURES = 16
NUM_BEAN_TYPES = 7
N_FOLDS = 5
SEED = 232134

# Configs to try
PARAMETER_GRID_NETWORK = {
    "hidden_layers": [[64, 64], [128, 64]],
    "activation": [nn.ReLU, nn.LeakyReLU],
    "dropout_rate": [0.0, 0.2],
    "lr": [0.01],
    "weight_decay": [0.0001],
}

# Keep results the same every run
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Generates all different possible config combinations
def generate_parameter_configs(parameter_options):
    total_configs = 1
    for parameter_type, options_for_parameter in parameter_options.items():
        total_configs *= len(options_for_parameter)

    configs_list = []
    for i in range(total_configs):
        configs_list.append({})

    # basically same as "configs so far"
    # needed to get non harmonic parameters options
    block_size = 1
    for parameter_type, options_for_parameter in parameter_options.items():
        options_len = len(options_for_parameter)
        for i in range(total_configs):
            parameter_idx = (i // block_size) % options_len
            paramter = options_for_parameter[parameter_idx]
            configs_list[i][parameter_type] = paramter

        block_size *= options_len

    return configs_list


class BeanData(Dataset):
    def __init__(self, inputs, labels):
        self.features = torch.as_tensor(inputs, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        # Store total number of samples
        self.n_samples = len(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


"""
Defining the architecture of MLP model
"""


class MLP(nn.Module):
    def __init__(
        self, input_size, num_outputs, hidden_layers, activation, dropout_rate, **kwargs
    ):
        super().__init__()

        layers = []
        current_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))

            current_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(current_size, num_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


"""
# Calculates entropy loss to compare prediction result with labels
# Use softmax to convert scores to propabilities
"""


def cross_entropy_loss(scores, y):

    # softmax calculation, subtracting the max first so exp() doesnt blow up
    exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True).values)
    probabilities = exp_scores / exp_scores.sum(dim=1, keepdim=True)

    row_idxs = torch.arange(len(y), device=scores.device)

    correct_class_probabilities = probabilities[row_idxs, y]
    return -torch.mean(torch.log(correct_class_probabilities + 1e-9))


""" 
load the training csv, convert bean labels to usable numeric values
"""


def load_training_data():

    df = pd.read_csv(TRAIN_DATASET)

    # Get the bean label column
    label_col = "Class" if "Class" in df.columns else df.columns[-1]

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


"""
Randomise rows and split in k chunks for cross validation step
"""


def get_folds(n, k, seed):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        # last fold gets whatever is remaining
        end = start + fold_size if i != k - 1 else n
        folds.append(indices[start:end])

    return folds


"""
Feature values vary greatly, must be standardized
"""


def standardize(inputs, mean, std):
    return (inputs - mean) / std


"""
Full training run
"""


def train_model(model, train_loader, val_loader, writer, tag, lr, weight_decay):

    # Chose gradient descent as most similar to lecture
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

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


"""
Train and evaluate model using cross validation
"""


def cross_validate(inputs, labels, config, run_name):

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

        train_loader = get_loader(
            standardize(inputs[train_index], mean, std),
            labels[train_index],
            BATCH_SIZE,
            shuffle=True,
        )
        val_loader = get_loader(
            standardize(inputs[current_fold], mean, std),
            labels[current_fold],
            BATCH_SIZE,
            shuffle=False,
        )

        model = MLP(NUM_FEATURES, NUM_BEAN_TYPES, **config).to(device)

        acc = train_model(
            model,
            train_loader,
            val_loader,
            writer,
            f"fold_{k + 1}",
            config["lr"],
            config["weight_decay"],
        )

        print(f"Fold {k + 1}: {acc * 100:.2f}%")
        cv_scores.append(acc)

    writer.close()

    avg_accuracy = np.mean(cv_scores) * 100
    print(f"Cross validation Balance Accuracy = {avg_accuracy:.2f} %")

    return avg_accuracy


"""
Train one final time with the settings of the winner of cross validation run (with or without dropout/weight decay)
"""


def train_final(inputs, labels, config, label_names):

    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0)

    train_loader = get_loader(
        standardize(inputs, mean, std), labels, BATCH_SIZE, shuffle=True
    )

    model = MLP(NUM_FEATURES, NUM_BEAN_TYPES, **config).to(device)

    writer = SummaryWriter("runs/final")
    train_model(
        model, train_loader, None, writer, "final", config["lr"], config["weight_decay"]
    )
    writer.close()

    # save model parameters and config so they can be reloaded later
    state = {
        "config": config,
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "label_names": label_names,
    }
    torch.save(state, MODEL_PATH)


"""
Perform forward pass on test set data 
Write each sample's predicted bean type to network.csv
"""


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
    configs_list = generate_parameter_configs(PARAMETER_GRID_NETWORK)

    print(f"Configs to test: {len(configs_list)}")

    best_acc = 0.0
    best_config = None

    for i, config in enumerate(configs_list):
        print(f"\n{config}:")
        acc = cross_validate(inputs, labels, config, f"runs/{i}")

        if acc > best_acc:
            best_acc = acc
            best_config = config

    print(f"\nBest setup: {best_config} ({best_acc:.2f}%)")

    # also saves the final model
    train_final(inputs, labels, best_config, label_names)

    # rebuild the network and load the saved parameters back in
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    saved_config = state["config"]
    state_dict = state["state_dict"]
    mean = state["mean"]
    std = state["std"]
    label_names = state["label_names"]

    model = MLP(NUM_FEATURES, NUM_BEAN_TYPES, **saved_config)
    model.load_state_dict(state_dict)
    model = model.to(device)

    bean_predictions(model, label_names, mean, std)


if __name__ == "__main__":
    main()
