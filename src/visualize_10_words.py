from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "word2vec.model"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_IMG = OUTPUTS_DIR / "word_vectors.png"


DEFAULT_WORDS = [
    "king",
    "queen",
    "man",
    "woman",
    "doctor",
    "teacher",
    "apple",
    "orange",
    "cat",
    "dog",
]


def load_keyed_vectors(model_path: Path = MODEL_PATH) -> KeyedVectors:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please run training first."
        )
    model = Word2Vec.load(str(model_path))
    return model.wv


def choose_words(wv: KeyedVectors, desired: List[str], num_words: int = 10) -> List[str]:
    words = [w for w in desired if w in wv.key_to_index]
    if len(words) < num_words:
        for candidate in wv.index_to_key:
            if candidate not in words:
                words.append(candidate)
            if len(words) == num_words:
                break
    if len(words) < num_words:
        raise ValueError(f"Vocabulary size is less than {num_words}.")
    return words[:num_words]


def plot_10_words() -> Path:
    wv = load_keyed_vectors()
    words = choose_words(wv, DEFAULT_WORDS, num_words=10)

    vectors = np.array([wv[word] for word in words])
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c="steelblue", s=70)

    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=10)

    plt.title("2D Distribution of 10 Word Vectors (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    plt.close()

    print(f"Visualization saved to: {OUTPUT_IMG}")
    return OUTPUT_IMG


if __name__ == "__main__":
    plot_10_words()
