# train.py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import train_ds, val_ds, CLASS_NAMES, EPOCHS, IMG_SIZE
from models import (
    build_base_cnn,
    build_stn_cnn,
    build_attention_cnn,
)
from utils import (
    plot_history,
    evaluate_model,
    per_class_accuracy,
    predict_with_topk,
)

sns.set(style="whitegrid")


def train_and_eval(model_fn, name: str):
    print(f"\n🔹 Training {name}...")
    model = model_fn()

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )
    t1 = time.time()

    plot_history(history, name)
    evaluate_model(model, val_ds, CLASS_NAMES)

    loss, acc = model.evaluate(val_ds, verbose=0)

    # Save (so you can reload next time without training)
    fname = name.lower().replace(" ", "_").replace("+", "plus")
    model.save(f"{fname}.keras")

    metrics = {
        "Model": name,
        "Val Accuracy": round(acc, 4),
        "Val Loss": round(loss, 4),
        "Train Time (s)": round(t1 - t0, 2),
    }
    return model, metrics


if __name__ == "__main__":
    # ==== Train all three models ====
    models = []
    metrics = []

    m_base, r_base = train_and_eval(build_base_cnn, "Base CNN")
    models.append(("Base CNN", m_base))
    metrics.append(r_base)

    m_stn, r_stn = train_and_eval(build_stn_cnn, "CNN + STN")
    models.append(("CNN + STN", m_stn))
    metrics.append(r_stn)

    m_attn, r_attn = train_and_eval(
        build_attention_cnn, "CNN + Attention"
    )
    models.append(("CNN + Attention", m_attn))
    metrics.append(r_attn)

    # ==== Comparison table + bar chart ====
    df = pd.DataFrame(metrics)
    print("\nModel Comparison Summary:")
    print(df[["Model", "Val Accuracy", "Val Loss", "Train Time (s)"]])

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="Val Accuracy")
    plt.ylim(0.5, 0.85)
    plt.title("EuroSAT — Base vs STN vs Attention")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    # ==== Per-class accuracy ====
    for name, model in models:
        print(f"\n== Per-class accuracy: {name} ==")
        print(per_class_accuracy(model, val_ds, CLASS_NAMES))

    # ==== Optional: external image checks (example paths) ====
    # img, probs, top3 = predict_with_topk(m_base, "forest.jpg",
    #                                      IMG_SIZE, CLASS_NAMES, k=3)
    # print("\nBase top-3 on forest.jpg:")
    # for c, p in top3:
    #     print(f"{c:20s} {p:.3f}")
    # plt.imshow(img)
    # plt.axis("off")
    # plt.title("Forest image")
    # plt.show()
