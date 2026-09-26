import random


DRAWING_WORDS = [
    "sun",
    "house",
    "tree",
    "car",
    "boat",
    "fish",
    "cat",
    "dog",
    "flower",
    "apple",
    "cloud",
    "mountain",
    "rocket",
    "airplane",
    "bicycle",
    "cup",
    "ice cream",
    "pizza",
    "umbrella",
]


def get_random_word(previous_word=None):
    """
    Returns a random word.

    If possible, the new word will be different from the previous word.
    """

    available_words = DRAWING_WORDS.copy()

    if previous_word in available_words and len(available_words) > 1:
        available_words.remove(previous_word)

    return random.choice(available_words)