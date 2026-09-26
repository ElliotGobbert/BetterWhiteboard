import math

def is_hand_closed(hand_landmarks):
    """Returns True if the hand is closed (fist), False if open."""
    wrist = hand_landmarks.landmark[0]

    # Fingertip indices: Index (8), Middle (12), Ring (16), Pinky (20)
    tip_indices = [8, 12, 16, 20]
    # Corresponding base/MCP joint indices: Index (5), Middle (9), Ring (13), Pinky (17)
    mcp_indices = [5, 9, 13, 17]

    folded_fingers = 0
    for tip_idx, mcp_idx in zip(tip_indices, mcp_indices):
        tip = hand_landmarks.landmark[tip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]

        dist_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_mcp_wrist = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)

        # A folded tip sits near the wrist relative to its base joint. The
        # multiplier allows for small landmark-detection inaccuracies.
        if dist_tip_wrist < dist_mcp_wrist * 1.15:
            folded_fingers += 1

    # Treat three folded non-thumb fingers as a fist, allowing a little noise.
    return folded_fingers >= 3

def count_extended_fingers(hand_landmarks):
    """Returns the number of extended fingers (excluding thumb for simplicity)."""
    tip_indices = [8, 12, 16, 20]
    mcp_indices = [5, 9, 13, 17]
    
    extended = 0
    for tip_idx, mcp_idx in zip(tip_indices, mcp_indices):
        # Image coordinates increase downward, so an extended tip has a smaller
        # y value than its base (MCP) joint.
        if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[mcp_idx].y:
            extended += 1
            
    return extended
