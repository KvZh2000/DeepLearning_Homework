from src.train_word2vec import run_training
from src.visualize_10_words import plot_10_words


def main() -> None:
    run_training()
    plot_10_words()
    print("Done: model trained and 10-word vector plot generated.")


if __name__ == "__main__":
    main()
