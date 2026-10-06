import pandas as pd
import numpy as np
from sklearn import tree

# Tree:
#  Best Config: {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': None}
#  CV Accuracy: 0.9092
#  CV Balanced Accuracy: 0.9205844960302689
#
# Forest:
# Best Config: {'forest_size': 50, 'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2'}
# CV Accuracy: 0.9260999999999999
# CV Balanced Accuracy: 0.9349031405515305

PARAMETER_GRID_TREE = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [None, "sqrt", "log2"],
}

PARAMETER_GRID_FOREST = {
    "forest_size": [50, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# HELPER FUNCTIONS


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


# TREE FUNCTIONS


def fit_tree(train, parameters):
    train_feature_cols = train.drop(columns=["Class"])
    train_class_col = train["Class"]

    dTree = tree.DecisionTreeClassifier(**parameters)
    dTree.fit(train_feature_cols, train_class_col)
    return dTree


def score_tree(dTree, test):
    test_feature_cols = test.drop(columns=["Class"])
    test_class_col = test["Class"]

    predicted = dTree.predict(test_feature_cols)

    # wrote my own version as an exercise
    accuracy, balanced_accuracy = calc_accuracies(
        predicted.tolist(), test_class_col.tolist()
    )

    return (accuracy, balanced_accuracy)


def cross_validate_tree(partitions_split, parameters):
    total_accuracy = 0
    total_balanced_accuracy = 0
    for test, train in partitions_split:
        dTree = fit_tree(train, parameters)
        accuracy, balanced_accuracy = score_tree(dTree, test)
        total_accuracy += accuracy
        total_balanced_accuracy += balanced_accuracy

    avg_accuracy = total_accuracy / len(partitions_split)
    avg_balanced_accuracy = total_balanced_accuracy / len(partitions_split)
    return (avg_accuracy, avg_balanced_accuracy)


def get_best_tree_config(configs_list, partitions_split):
    best_config = None
    best_accuracy = 0
    best_balanced_accuracy = 0

    for config in configs_list:
        avg_accuracy, avg_balanced_accuracy = cross_validate_tree(
            partitions_split, config
        )
        if best_balanced_accuracy < avg_balanced_accuracy:
            best_accuracy = avg_accuracy
            best_balanced_accuracy = avg_balanced_accuracy
            best_config = config

    return (best_config, best_accuracy, best_balanced_accuracy)


# FOREST FUNCTIONS


def fit_forest(train, config):
    forest = []
    tree_parameters = config.copy()
    forest_size = tree_parameters.pop("forest_size")
    for i in range(forest_size):
        sampled_train = train.sample(frac=1.0, replace=True)
        dTree = fit_tree(sampled_train, tree_parameters)
        forest.append(dTree)

    return forest


def predict_forest(forest, test_features):
    all_predictions = []
    for dTree in forest:
        predicted = dTree.predict(test_features)
        all_predictions.append(predicted)

    all_predictions_by_row = np.column_stack(all_predictions)

    final_predictions = []
    for row_prediction in all_predictions_by_row:
        predictions_counter = {}
        for predicted_class in row_prediction:
            predictions_counter[predicted_class] = (
                predictions_counter.get(predicted_class, 0) + 1
            )

        best_class = None
        highest_votes = 0
        for b_class, count in predictions_counter.items():
            if count > highest_votes:
                best_class = b_class
                highest_votes = count

        final_predictions.append(best_class)

    return final_predictions


# Could honestly just refactor into one score function for forest and tree
def score_forest(forest, test):
    test_feature_cols = test.drop(columns=["Class"])
    test_class_col = test["Class"]

    predicted = predict_forest(forest, test_feature_cols)
    accuracy, balanced_accuracy = calc_accuracies(predicted, test_class_col.tolist())

    return (accuracy, balanced_accuracy)


def cross_validate_forest(partitions_split, config):
    total_accuracy = 0
    total_balanced_accuracy = 0

    for test, train in partitions_split:
        forest = fit_forest(train, config)
        accuracy, balanced_accuracy = score_forest(forest, test)
        total_accuracy += accuracy
        total_balanced_accuracy += balanced_accuracy

    avg_accuracy = total_accuracy / len(partitions_split)
    avg_balanced_accuracy = total_balanced_accuracy / len(partitions_split)
    return (avg_accuracy, avg_balanced_accuracy)


def get_best_forest_config(configs_list, partitions_split):
    best_config = None
    best_accuracy = 0
    best_balanced_accuracy = 0

    for config in configs_list:
        avg_accuracy, avg_balanced_accuracy = cross_validate_forest(
            partitions_split, config
        )
        if best_balanced_accuracy < avg_balanced_accuracy:
            best_accuracy = avg_accuracy
            best_balanced_accuracy = avg_balanced_accuracy
            best_config = config

    return (best_config, best_accuracy, best_balanced_accuracy)


# MAIN


def main():
    test_df = pd.read_csv("dry_bean_test.csv")
    train_df = pd.read_csv("dry_bean_train.csv")
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # classes = train_df["Class"]
    # print(set(classes))

    partitions_split = partition_datafram(train_df, 5)

    # SINGLE TREE
    # configs_list = generate_parameter_configs(PARAMETER_GRID_TREE)
    # best_config, best_accuracy, best_balanced_accuracy = get_best_tree_config(
    #     configs_list, partitions_split
    # )
    # dTree = fit_tree(train_df, best_config)
    # test_df["Target"] = dTree.predict(test_df)
    # test_df.to_csv("tree.csv", index=False)
    # test_df.drop(columns=["Target"], errors="ignore")
    # print(
    #     f"Tree:\n Best Config: {best_config}\n CV Accuracy: {best_accuracy}\n CV Balanced Accuracy: {best_balanced_accuracy}\n"
    # )

    # FOREST
    configs_list = generate_parameter_configs(PARAMETER_GRID_FOREST)
    best_config, best_accuracy, best_balanced_accuracy = get_best_forest_config(
        configs_list, partitions_split
    )
    forest = fit_forest(train_df, best_config)
    test_df["Target"] = predict_forest(forest, test_df)
    test_df.to_csv("forest.csv", index=False)

    print(
        f"Forest:\n Best Config: {best_config}\n CV Accuracy: {best_accuracy}\n CV Balanced Accuracy: {best_balanced_accuracy}\n"
    )


if __name__ == "__main__":
    main()
