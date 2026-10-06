import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class SeedDataset(Dataset):
    def __init__(self, features, classes):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.classes[idx]


class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes):
        super(NeuralNet, self).__init__()
        self.hidden_layer = nn.Linear(in_features, hidden_dim)
        # Batch normalization apparently helps flatten out wild swings in weights
        # between batches
        self.batch_normalization = nn.BatchNorm1d(hidden_dim)
        # Just activation function, here same as max(0, x)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, out_classes)

    def forward(self, features_data):
        x = self.hidden_layer(features_data)
        x = self.batch_normalization(x)
        x = self.relu(x)
        return self.output_layer(x)


def soft_max(x):
    # Traps final output vectors between 0 and 1 to get probabilities
    exponentials = torch.exp(x)
    total = torch.sum(exponentials, dim=1, keepdim=True)

    # Apparently if 2 tensors are matching dimensions this is basically like a
    # fast for loop (instead of dividing each individually)
    return exponentials / total


def cross_entropy(logits, labels):
    probabilities = soft_max(logits)
    row_idxs = torch.arange(len(labels))

    # same as before, basically faster for loop to extract prob for the answer class
    # (32, 7) -> (32,1)
    correct_class_probabilities = probabilities[row_idxs, labels]
    return -torch.mean(torch.log(correct_class_probabilities))


# HELPERS (SOME DUPLICATED FROM FOREST.PY SINCE I THINK HAVE TO BE INDEPENDENT FILES)


def partition_datafram(dataframe, num_partitions):
    partitions = []
    size_partition = len(dataframe) // num_partitions
    left_over = len(dataframe) % num_partitions

    for i in range(num_partitions):
        start_idx = i * size_partition
        end_idx = start_idx + size_partition
        if i == num_partitions - 1:
            end_idx += left_over
        partition = dataframe.iloc[start_idx:end_idx]
        partitions.append(partition)

    partitons_split = []
    for i in range(num_partitions):
        test = partitions[i]
        train_list = []
        for j in range(num_partitions):
            if i == j:
                continue
            train_list.append(partitions[j])

        train = pd.concat(train_list).reset_index(drop=True)
        partitons_split.append((test, train))

    return partitons_split


def calc_accuracies(predicted, actual):
    # Hashmap of = class_name: (correct, total)
    count_by_classes = dict.fromkeys(actual, (0, 0))

    for i in range(len(predicted)):
        guessed_correct, actually_correct = count_by_classes[actual[i]]
        actually_correct += 1
        if predicted[i] == actual[i]:
            guessed_correct += 1
        count_by_classes[actual[i]] = (guessed_correct, actually_correct)

    total_guessed_correct = 0
    total_balance_p = 0
    for guessed_correct, actually_correct in count_by_classes.values():
        total_balance_p += guessed_correct / actually_correct
        total_guessed_correct += guessed_correct

    accuracy = total_guessed_correct / len(actual)
    balanced_accuracy = total_balance_p / len(count_by_classes)

    return (accuracy, balanced_accuracy)


# Network functions


def train_loop(train, writer, counter):
    train_classes = train["Class"].map(classes_name_dict).values
    train_features = train.drop(columns="Class").values

    train_dataset = SeedDataset(train_features, train_classes)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    model = NeuralNet(in_features=16, hidden_dim=32, out_classes=7)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    # alter this for total epochs, need to refactor later
    for epoch in range(100):
        train_epoch_loss = 0
        for batch_features, batch_labels in train_data_loader:
            optimiser.zero_grad()  # clear out prev run gradients
            logits = model(
                batch_features
            )  # special constructor that then calls our .forward()
            loss = cross_entropy(logits, batch_labels)
            loss.backward()  # Calcs the gradients for each weight
            optimiser.step()  # Updates the weights all at oncek

            train_epoch_loss += loss.item() * len(batch_labels)
        avg_epoch_loss = train_epoch_loss / len(train)

        writer.add_scalar(f"Loss/partiton_{counter}", avg_epoch_loss, epoch)

    return model


def eval_loop(model, test):
    test_classes = test["Class"].map(classes_name_dict).values
    test_features = test.drop(columns="Class").values

    test_dataset = SeedDataset(test_features, test_classes)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # Eval loop
    predicted = []
    actual = []
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in test_data_loader:
            logits = model(
                batch_features
            )  # special constructor that then calls our .forward()
            predicted.extend(torch.argmax(logits, dim=1).tolist())
            actual.extend(batch_labels.tolist())
            # multiply by the batch size so can scale our mean properly
            total_loss += cross_entropy(logits, batch_labels).item() * len(batch_labels)

    accuracy, balanced_accuracy = calc_accuracies(predicted, actual)
    loss = total_loss / len(test)

    return accuracy, balanced_accuracy, loss


def cross_validate(partitions_split, writer):
    total_accuracy = 0
    total_balanced_accuracy = 0
    total_loss = 0
    counter = 0
    for test, train in partitions_split:
        model = train_loop(train, writer, counter)
        accuracy, balanced_accuracy, loss = eval_loop(model, test)

        writer.add_scalar("Accuracy", accuracy, counter)
        writer.add_scalar("Loss", loss, counter)

        total_accuracy += accuracy
        total_balanced_accuracy += balanced_accuracy
        total_loss += loss
        counter += 1

    avg_accuracy = total_accuracy / len(partitions_split)
    avg_balanced_accuracy = total_balanced_accuracy / len(partitions_split)
    avg_loss = total_loss / len(partitions_split)
    return (avg_accuracy, avg_balanced_accuracy, avg_loss)


def main():
    train_df = pd.read_csv("dry_bean_train.csv")
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)

    partitions_split = partition_datafram(train_df, 5)

    global classes_name_dict
    classes_name_dict = {}
    class_names = sorted(train_df["Class"].unique())

    counter = 0
    for class_name in class_names:
        classes_name_dict[class_name] = counter
        counter += 1

    writer = SummaryWriter(log_dir="beans")

    avg_accuracy, avg_balanced_accuracy, avg_loss = cross_validate(
        partitions_split, writer
    )

    writer.close()

    # train final model using good parameters


if __name__ == "__main__":
    main()
