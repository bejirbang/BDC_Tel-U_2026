from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "2_processed"
    / "transcripts"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis"
    / "modeling"
)

REPORT_DIR = OUTPUT_DIR / "reports"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrix"


RANDOM_STATE = 42
TEST_SIZE = 0.2


STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "ada", "untuk", "dengan", "atau", "pada", "dalam",
    "jadi", "akan", "sudah", "udah", "telah", "bisa",
    "dapat", "tidak", "gak", "ga", "nggak", "enggak",
    "ya", "iya", "saya", "aku", "kami", "kita",
    "kamu", "lu", "lo", "gue", "dia", "mereka",
    "apa", "siapa", "mana", "bagaimana", "kenapa",
    "kalau", "kalo", "karena", "tapi", "namun",
    "juga", "lagi", "lebih", "sangat", "banget",
    "sama", "seperti", "kayak", "gitu", "begitu",
    "tuh", "nih", "kan", "kok", "sih", "dong",
    "deh", "lah", "pun", "saja", "aja",
    "masih", "belum", "pernah",
    "harus", "mau", "ingin", "boleh",
    "buat", "pakai", "pake", "terus",
    "dulu", "tadi", "nanti", "sekarang",
    "ketika", "saat", "setiap", "sebuah",
    "seorang", "orang", "hal", "nya",
}


def load_transcripts() -> pd.DataFrame:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Directory transcript tidak ditemukan: {INPUT_DIR}"
        )

    records = []

    for emotion_dir in sorted(INPUT_DIR.iterdir()):

        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name

        for transcript_file in sorted(
            emotion_dir.glob("*_processed.txt")
        ):

            try:
                text = transcript_file.read_text(
                    encoding="utf-8"
                ).strip()

                if not text:
                    continue

                records.append(
                    {
                        "emotion": emotion,
                        "file_name": transcript_file.name,
                        "text": text,
                    }
                )

            except Exception as exc:
                print(
                    f"[ERROR] "
                    f"{emotion}/{transcript_file.name}: "
                    f"{exc}"
                )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "Tidak ada transcript processed yang ditemukan."
        )

    return df


def remove_stopwords(text: str) -> str:
    words = text.split()

    return " ".join(
        word
        for word in words
        if word not in STOPWORDS
    )


def prepare_text(
    df: pd.DataFrame,
    remove_sw: bool,
) -> pd.Series:

    texts = df["text"].copy()

    if remove_sw:
        texts = texts.apply(remove_stopwords)

    return texts


def split_dataset(
    texts: pd.Series,
    labels: pd.Series,
):
    rare_classes = (
        labels.value_counts()
        [labels.value_counts() < 2]
        .index
        .tolist()
    )

    if rare_classes:
        print(
            "\nWARNING:"
        )

        print(
            "Kelas berikut hanya memiliki "
            "1 sample:"
        )

        for emotion in rare_classes:
            print(f"  - {emotion}")

        print(
            "\nStratified split tidak dapat digunakan."
        )

        print(
            "Train/test split dilakukan tanpa stratify."
        )

    return train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )


def build_vectorizer(
    ngram_range: tuple[int, int],
) -> TfidfVectorizer:

    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )


def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer: TfidfVectorizer,
):

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    print(
        f"Jumlah feature: "
        f"{len(vectorizer.get_feature_names_out())}"
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_tfidf,
        y_train,
    )

    predictions = model.predict(
        X_test_tfidf
    )

    return model, predictions


def evaluate_model(
    y_test,
    predictions,
    experiment_name: str,
):

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        REPORT_DIR
        / f"classification_report_{experiment_name}.csv",
        encoding="utf-8",
    )

    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def save_confusion_matrix(
    y_test,
    predictions,
    experiment_name: str,
):

    labels = sorted(
        set(y_test) | set(predictions)
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(cm)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Emotion",
        ylabel="True Emotion",
        title=(
            f"Confusion Matrix - "
            f"{experiment_name}"
        ),
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    fig.savefig(
        CONFUSION_DIR
        / f"confusion_matrix_{experiment_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    ngram_range: tuple[int, int],
    remove_sw: bool,
):
    print("\n" + "=" * 70)

    print(
        f"EXPERIMENT: {experiment_name}"
    )

    print("=" * 70)

    texts = prepare_text(
        df,
        remove_sw=remove_sw,
    )

    labels = df["emotion"]

    X_train, X_test, y_train, y_test = (
        split_dataset(
            texts,
            labels,
        )
    )

    print(
        f"\nTrain samples: {len(X_train)}"
    )

    print(
        f"Test samples : {len(X_test)}"
    )

    print(
        "\nTrain distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
    )

    print(
        "\nTest distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
    )

    vectorizer = build_vectorizer(
        ngram_range
    )

    model, predictions = train_model(
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer,
    )

    metrics = evaluate_model(
        y_test,
        predictions,
        experiment_name,
    )

    save_confusion_matrix(
        y_test,
        predictions,
        experiment_name,
    )

    print(
        "\nRESULT:"
    )

    print(
        f"Accuracy       : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{metrics['weighted_f1']:.4f}"
    )

    return metrics


def main():

    print("=" * 70)
    print("EMOTION CLASSIFICATION MODELING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFUSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_transcripts()

    print(
        f"\nTotal transcript: {len(df)}"
    )

    print(
        "\nDistribusi emotion:"
    )

    print(
        df["emotion"]
        .value_counts()
        .sort_index()
    )

    experiments = [
        {
            "name": "unigram",
            "ngram_range": (1, 1),
            "remove_sw": False,
        },
        {
            "name": "stopwords",
            "ngram_range": (1, 1),
            "remove_sw": True,
        },
        {
            "name": "bigram",
            "ngram_range": (1, 2),
            "remove_sw": True,
        },
        {
            "name": "trigram",
            "ngram_range": (1, 3),
            "remove_sw": True,
        },
    ]

    results = []

    for experiment in experiments:

        metrics = run_experiment(
            df=df,
            experiment_name=experiment["name"],
            ngram_range=experiment[
                "ngram_range"
            ],
            remove_sw=experiment[
                "remove_sw"
            ],
        )

        results.append(metrics)

    comparison_df = pd.DataFrame(
        results
    )

    comparison_df = comparison_df.sort_values(
        "macro_f1",
        ascending=False,
    )

    comparison_df.to_csv(
        OUTPUT_DIR
        / "model_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(
        "\nModeling selesai."
    )

    print(
        f"Hasil disimpan di: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()