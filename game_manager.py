import time

from words import get_random_word


class GameState:

    DRAWING = "DRAWING"
    REVEAL = "REVEAL"


class GameManager:

    def __init__(self, round_duration=30):

        self.round_duration = round_duration

        self.state = GameState.DRAWING

        self.current_word = None
        self.previous_word = None

        self.start_time = None
        self.final_drawing = None

        self.start_new_round()

    def start_new_round(self):

        self.previous_word = self.current_word

        self.current_word = get_random_word(
            self.previous_word
        )

        self.start_time = time.monotonic()

        self.state = GameState.DRAWING

        self.final_drawing = None

        print(
            f"[GAME] New round started. "
            f"Word: {self.current_word}"
        )

    def get_remaining_seconds(self):

        if self.state != GameState.DRAWING:
            return 0

        elapsed = time.monotonic() - self.start_time

        remaining = self.round_duration - elapsed

        return max(
            0,
            int(remaining + 0.999)
        )

    def is_time_up(self):

        if self.state != GameState.DRAWING:
            return False

        elapsed = time.monotonic() - self.start_time

        return elapsed >= self.round_duration

    def finish_round(self, drawing):

        self.final_drawing = drawing.copy()

        self.state = GameState.REVEAL

        print(
            f"[GAME] Time is up! "
            f"Drawing was for: {self.current_word}"
        )

    def is_drawing(self):

        return self.state == GameState.DRAWING

    def is_reveal(self):

        return self.state == GameState.REVEAL

    def get_word(self):

        return self.current_word