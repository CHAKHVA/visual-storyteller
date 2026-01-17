"""Vocabulary class for mapping between words and indices in captions."""

from collections import Counter


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


if __name__ == "__main__":
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
