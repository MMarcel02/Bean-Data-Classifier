import math
import pandas as pd
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score

PARAMETER_GRID = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 3, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"],
}


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


def cross_validate(partitions_split, parameters):
    total_accuracy = 0
    total_balanced_accuracy = 0
    for test, train in partitions_split:
        dTree = fit(train, parameters)
        accuracy, balanced_accuracy = score(dTree, test)
        total_accuracy += accuracy
        total_balanced_accuracy += balanced_accuracy

    avg_accuracy = total_accuracy / len(partitions_split)
    avg_balanced_accuracy = total_balanced_accuracy / len(partitions_split)
    return (avg_accuracy, avg_balanced_accuracy)


def fit(train, parameters):
    train_feature_cols = train.drop(columns=["Class"])
    train_class_col = train["Class"]

    dTree = tree.DecisionTreeClassifier(**parameters)
    dTree.fit(train_feature_cols, train_class_col)
    return dTree


def score(dTree, test):
    test_feature_cols = test.drop(columns=["Class"])
    test_class_col = test["Class"]

    predicted = dTree.predict(test_feature_cols)

    # wrote my own version as an exercise
    balanced_accuracy = calc_balanced_accuracy(
        predicted.tolist(), test_class_col.tolist()
    )

    # Just to compare against the actual version
    # sk_learn_balanced_accuracy = balanced_accuracy_score(test_class_col, predicted)
    # is_matching = math.isclose(
    #     balanced_accuracy, sk_learn_balanced_accuracy, rel_tol=1e-9
    # )
    # print(
    #     f"Match? {is_matching} | Yours: {balanced_accuracy:.4f} | Official:"
    #     f" {sk_learn_balanced_accuracy:.4f}"
    # )

    accuracy = dTree.score(test_feature_cols, test_class_col)

    return (accuracy, balanced_accuracy)


def calc_balanced_accuracy(predicted, actual):
    count_by_classes = dict.fromkeys(actual, (0, 0))

    for i in range(len(predicted)):
        guessed_correct, actually_correct = count_by_classes[actual[i]]
        actually_correct += 1
        if predicted[i] == actual[i]:
            guessed_correct += 1
        count_by_classes[actual[i]] = (guessed_correct, actually_correct)

    total = 0
    for guessed_correct, actually_correct in count_by_classes.values():
        total += guessed_correct / actually_correct

    return total / len(count_by_classes)


def main():
    test_df = pd.read_csv("dry_bean_test.csv")
    train_df = pd.read_csv("dry_bean_train.csv")
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)

    partitions_split = partition_datafram(train_df, 5)

    configs_list = generate_parameter_configs(PARAMETER_GRID)

    best_config = None

    best_accuracy = 0
    best_balanced_accuracy = 0.5

    for config in configs_list:
        avg_accuracy, avg_balanced_accuracy = cross_validate(partitions_split, config)
        if best_balanced_accuracy < avg_balanced_accuracy:
            best_accuracy = avg_accuracy
            best_balanced_accuracy = avg_balanced_accuracy
            best_config = config

    print(
        f"Accuracy: {best_accuracy} \nBalanced Accuracy: {best_balanced_accuracy}\nConfig: {best_config}"
    )

    dTree = fit(train_df, best_config)

    test_df["Predicted"] = dTree.predict(test_df)
    test_df.to_csv("single_tree_predictions.csv", index=False)


if __name__ == "__main__":
    main()
