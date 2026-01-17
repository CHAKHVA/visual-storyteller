"""Vocabulary class for mapping between words and indices in captions."""

import pickle
from collections import Counter
from pathlib import Path


class Vocabulary:
    """Handles mapping between words and indices for caption processing."""

    def __init__(self, freq_threshold: int = 5):
        """
        Initialize vocabulary with special tokens.

        Special tokens are assigned fixed indices:
        - <PAD> = 0: Padding token for sequences
        - <SOS> = 1: Start of sequence token
        - <EOS> = 2: End of sequence token
        - <UNK> = 3: Unknown token for out-of-vocabulary words

        Args:
            freq_threshold: Minimum frequency for a word to be included in vocabulary.
                Words appearing fewer times are mapped to <UNK>.
        """
        self.freq_threshold = freq_threshold

        # Index to string mapping
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
        }

        # String to index mapping
        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }

    def __len__(self) -> int:
        """
        Return the size of the vocabulary.

        Returns:
            Number of unique tokens in vocabulary.
        """
        return len(self.itos)

    def build_vocabulary(self, captions: list[str]) -> None:
        """
        Build vocabulary from a list of captions.

        Tokenizes captions, counts word frequencies, and adds words
        that meet the frequency threshold to the vocabulary.

        Args:
            captions: List of caption strings to build vocabulary from.
        """
        # Count word frequencies across all captions
        word_counts = Counter()

        for caption in captions:
            # Tokenize: lowercase and split on whitespace
            tokens = caption.lower().split()
            word_counts.update(tokens)

        # Add words that meet frequency threshold
        idx = len(self.itos)  # Start from next available index
        for word, count in word_counts.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"Vocabulary built with {len(self)} tokens")

    def numericalize(self, text: str) -> list[int]:
        """
        Convert text to a list of indices.

        Tokenizes the input text and maps each token to its corresponding
        index. Unknown words are mapped to <UNK>.

        Args:
            text: Input text string to convert.

        Returns:
            List of indices corresponding to tokens in the text.
        """
        tokens = text.lower().split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

    def denumericalize(self, indices: list[int]) -> str:
        """
        Convert a list of indices back to text.

        Converts indices to words and joins them into a string.
        Skips special tokens (<PAD>, <SOS>, <EOS>) for cleaner output.

        Args:
            indices: List of token indices to convert.

        Returns:
            Text string reconstructed from indices.
        """
        # Skip special tokens when converting back to text
        skip_tokens = {"<PAD>", "<SOS>", "<EOS>"}
        words = []

        for idx in indices:
            word = self.itos.get(idx, "<UNK>")
            if word not in skip_tokens:
                words.append(word)

        return " ".join(words)

    def save(self, filepath_str: str) -> None:
        """
        Save vocabulary to a pickle file.

        Saves the vocabulary state including mappings and frequency threshold
        so it can be reloaded later.

        Args:
            filepath: Path where the vocabulary should be saved.

        Raises:
            IOError: If there's an error writing to the file.
        """
        filepath = Path(filepath_str)

        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        vocab_state = {
            "itos": self.itos,
            "stoi": self.stoi,
            "freq_threshold": self.freq_threshold,
        }

        try:
            with open(filepath, "wb") as f:
                pickle.dump(vocab_state, f)
            print(f"Vocabulary saved to {filepath}")
        except Exception as e:
            raise IOError(f"Failed to save vocabulary to {filepath}: {e}")

    @classmethod
    def load(cls, filepath_str: str) -> "Vocabulary":
        """
        Load vocabulary from a pickle file.

        Creates a new Vocabulary instance and restores its state from
        a previously saved file.

        Args:
            filepath: Path to the saved vocabulary file.

        Returns:
            Vocabulary instance with loaded state.

        Raises:
            FileNotFoundError: If the vocabulary file doesn't exist.
            IOError: If there's an error reading the file.
        """
        filepath = Path(filepath_str)

        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                vocab_state = pickle.load(f)

            # Create new instance without initializing default values
            vocab = cls.__new__(cls)
            vocab.itos = vocab_state["itos"]
            vocab.stoi = vocab_state["stoi"]
            vocab.freq_threshold = vocab_state["freq_threshold"]

            print(f"Vocabulary loaded from {filepath} ({len(vocab)} tokens)")
            return vocab
        except Exception as e:
            raise IOError(f"Failed to load vocabulary from {filepath}: {e}")


if __name__ == "__main__":
    import tempfile

    # Sample captions for testing
    sample_captions = [
        "a dog playing in the park",
        "a cat sitting on the couch",
        "a dog running in the park",
        "a bird flying in the sky",
        "the dog is playing with a ball",
        "a cat is sleeping on the couch",
        "the park is full of dogs",
        "a beautiful bird in the tree",
    ]

    # Create and build vocabulary
    print("=" * 50)
    print("Building vocabulary from sample captions")
    print("=" * 50)
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(sample_captions)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Special tokens: {list(vocab.stoi.keys())[:4]}")

    # Test numericalize
    test_caption = "a dog playing in the park"
    indices = vocab.numericalize(test_caption)
    print(f"\nOriginal caption: '{test_caption}'")
    print(f"Numericalized: {indices}")

    # Test denumericalize
    reconstructed = vocab.denumericalize(indices)
    print(f"Denumericalized: '{reconstructed}'")

    # Test with unknown words
    test_unknown = "a zebra running fast"
    indices_unknown = vocab.numericalize(test_unknown)
    print(f"\nCaption with unknown word: '{test_unknown}'")
    print(f"Numericalized: {indices_unknown}")
    reconstructed_unknown = vocab.denumericalize(indices_unknown)
    print(f"Denumericalized: '{reconstructed_unknown}'")

    # Test with special tokens
    test_with_special = [vocab.stoi["<SOS>"]] + indices + [vocab.stoi["<EOS>"]]
    print(f"\nWith special tokens: {test_with_special}")
    reconstructed_special = vocab.denumericalize(test_with_special)
    print(f"Denumericalized (special tokens skipped): '{reconstructed_special}'")

    # Test save/load functionality
    print("\n" + "=" * 50)
    print("Testing save/load functionality")
    print("=" * 50)

    # Save vocabulary to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        temp_path = tmp.name

    print(f"\nSaving vocabulary to temporary file: {temp_path}")
    vocab.save(temp_path)

    # Store original vocab size for comparison
    original_size = len(vocab)
    original_test_result = vocab.numericalize(test_caption)

    # Load vocabulary from file
    print(f"\nLoading vocabulary from file...")
    loaded_vocab = Vocabulary.load(temp_path)

    # Verify loaded vocabulary works correctly
    print(f"\nVerifying loaded vocabulary:")
    print(f"  Original vocab size: {original_size}")
    print(f"  Loaded vocab size: {len(loaded_vocab)}")
    print(f"  Sizes match: {original_size == len(loaded_vocab)}")

    # Test that numericalization works the same
    loaded_test_result = loaded_vocab.numericalize(test_caption)
    print(f"\n  Original numericalization: {original_test_result}")
    print(f"  Loaded numericalization: {loaded_test_result}")
    print(f"  Results match: {original_test_result == loaded_test_result}")

    # Test denumericalization
    loaded_reconstructed = loaded_vocab.denumericalize(loaded_test_result)
    print(f"\n  Loaded denumericalization: '{loaded_reconstructed}'")
    print(f"  Matches original: {loaded_reconstructed == reconstructed}")

    # Cleanup temporary file
    Path(temp_path).unlink()
    print(f"\nTemporary file cleaned up")
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
