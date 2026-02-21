import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from tensorflow.keras.preprocessing import image


def plot_history(history, name: str):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title(f"{name} — Acc")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"{name} — Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_ds, class_names):
    # Collect predictions
    y_true, y_pred = [], []

    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(yb.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(
        cm, display_labels=class_names
    ).plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(
        classification_report(y_true, y_pred, target_names=class_names)
    )


def per_class_accuracy(model, dataset, class_names):
    correct = np.zeros(len(class_names), dtype=int)
    total = np.zeros(len(class_names), dtype=int)

    for xb, yb in dataset:
        probs = model.predict(xb, verbose=0)
        pred = np.argmax(probs, axis=1)
        true = np.argmax(yb.numpy(), axis=1)
        for t, p in zip(true, pred):
            total[t] += 1
            if t == p:
                correct[t] += 1

    accs = correct / np.maximum(total, 1)

    import pandas as pd

    return (
        pd.DataFrame(
            {"class": class_names, "samples": total, "acc": accs}
        )
        .sort_values("acc")
        .reset_index(drop=True)
    )


def predict_with_topk(model, img_path, img_size, class_names, k=3):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    topk = np.argsort(probs)[::-1][:k]
    topk_labels = [(class_names[i], float(probs[i])) for i in topk]

    return img, probs, topk_labels
