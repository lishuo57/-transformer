import matplotlib.pyplot as plt
import pandas as pd

# 目前掌握的训练日志（epoch, train_loss, valid_loss）
# 如果后续拿到更多 epoch 的数据，可在此列表中补充
LOSS_HISTORY = [
    (4, 5.3938, 5.2112),
    (5, 5.1232, 4.9721),
    (6, 4.9044, 4.7787),
    (7, 4.7271, 4.6315),
    (8, 4.5817, 4.5141),
    (9, 4.4424, 4.4235),
    (10, 4.3627, 4.3288),
    (11, 4.2415, 4.2749),
    (12, 4.1562, 4.2446),
    (13, 4.0860, 4.2037),
]


def main() -> None:
    if not LOSS_HISTORY:
        raise SystemExit("LOSS_HISTORY 为空，请先填充训练日志数据。")

    df = pd.DataFrame(LOSS_HISTORY, columns=["epoch", "train_loss", "valid_loss"])
    df.sort_values("epoch", inplace=True)
    df.to_csv("loss_curve.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    plt.plot(df["epoch"], df["valid_loss"], marker="s", label="valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("已生成 loss_curve.csv 和 loss_curve.png")


if __name__ == "__main__":
    main()

