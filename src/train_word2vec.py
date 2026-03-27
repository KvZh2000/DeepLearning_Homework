from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from gensim.models import Word2Vec


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "corpus.txt"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "word2vec.model"
KV_PATH = MODELS_DIR / "word2vec.kv"


def tokenize_line(line: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", line.lower())
    return tokens


def load_sentences(corpus_path: Path = DATA_PATH) -> List[List[str]]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    sentences: List[List[str]] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            tokens = tokenize_line(line)
            if tokens:
                sentences.append(tokens)

    if not sentences:
        raise ValueError("No valid sentences found in corpus.")
    return sentences


def train_model(sentences: Iterable[List[str]]) -> Word2Vec:
    model = Word2Vec(
        sentences=list(sentences),
        vector_size=50,
        window=4,
        min_count=1,
        workers=1,
        sg=1,
        epochs=200,
        seed=42,
    )
    return model


def save_model(model: Word2Vec, model_path: Path = MODEL_PATH, kv_path: Path = KV_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    model.wv.save(str(kv_path))


def run_training() -> Word2Vec:
    print("[1/3] Loading corpus...")
    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentences.")

    print("[2/3] Training Word2Vec model...")
    model = train_model(sentences)

    print("[3/3] Saving model...")
    save_model(model)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"KeyedVectors saved to: {KV_PATH}")
    return model


if __name__ == "__main__":
    run_training()
