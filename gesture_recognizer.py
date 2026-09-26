import math


def distance(point_a, point_b):
    return math.hypot(
        point_a.x - point_b.x,
        point_a.y - point_b.y
    )


def is_hand_closed(hand_landmarks):
    """
    Detects a fist.

    A finger is considered folded when its fingertip is
    relatively close to the wrist compared to its MCP joint.
    """

    wrist = hand_landmarks.landmark[0]

    tip_indices = [8, 12, 16, 20]
    mcp_indices = [5, 9, 13, 17]

    folded_fingers = 0

    for tip_idx, mcp_idx in zip(tip_indices, mcp_indices):

        tip = hand_landmarks.landmark[tip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]

        tip_distance = distance(tip, wrist)
        mcp_distance = distance(mcp, wrist)

        if tip_distance < mcp_distance * 1.15:
            folded_fingers += 1

    return folded_fingers >= 3


def count_extended_fingers(hand_landmarks):
    """
    Counts extended fingers using fingertip vs PIP positions.
    """

    tip_indices = [8, 12, 16, 20]
    pip_indices = [6, 10, 14, 18]

    extended = 0

    for tip_idx, pip_idx in zip(
        tip_indices,
        pip_indices
    ):

        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]

        if tip.y < pip.y:
            extended += 1

    return extended


def get_gesture(hand_landmarks):
    """
    Gesture mapping:

        Fist          -> DRAW
        Exactly 2     -> ERASE
        Everything else -> PAUSE
    """

    # Check fist first.
    if is_hand_closed(hand_landmarks):
        return "DRAW"

    # Count extended fingers.
    extended_fingers = count_extended_fingers(
        hand_landmarks
    )

    # ONLY exactly two fingers should erase.
    if extended_fingers == 2:
        return "ERASE"

    # Open hand, one finger, three fingers, etc.
    return "PAUSE"