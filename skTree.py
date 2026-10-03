import pandas as pd
from sklearn import tree

train_df = pd.read_csv("dry_bean_train.csv")

train_feature_cols = train_df.drop(columns=["Class"])
train_class_col = train_df["Class"]

dTree = tree.DecisionTreeClassifier(criterion="entropy", splitter="best")
dTree.fit(train_feature_cols, train_class_col)

feautre_names = list(train_feature_cols.columns)
learned_rules = tree.export_text(dTree, feature_names=feautre_names)
print(learned_rules)
