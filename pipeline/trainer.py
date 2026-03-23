"""
Этап 4: Обучение моделей
"""
import os, pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def train(X_train, y_train, batch_idx):
    os.makedirs("models", exist_ok=True)
    models = {}

    # Decision Tree
    tree = DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
    )
    tree.fit(X_train, y_train)
    path = f"models/tree_batch_{batch_idx:04d}.pkl"
    with open(path, "wb") as f:
        pickle.dump(tree, f)
    models["tree"] = tree
    print(f"  DecisionTree обучен, глубина={tree.get_depth()}")

    # MLP
    prev_mlp_path = f"models/mlp_batch_{batch_idx-1:04d}.pkl"
    if batch_idx > 0 and os.path.exists(prev_mlp_path):
        with open(prev_mlp_path, "rb") as f:
            mlp = pickle.load(f)
        mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
        print(f"  MLP дообучен (partial_fit)")
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        mlp.fit(X_train, y_train)
        print(f"  MLP обучен с нуля")

    path = f"models/mlp_batch_{batch_idx:04d}.pkl"
    with open(path, "wb") as f:
        pickle.dump(mlp, f)
    models["mlp"] = mlp

    return models
