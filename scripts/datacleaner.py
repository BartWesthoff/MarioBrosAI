import os
import json
import re

# Paths relative to the script's directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
movements_path = os.path.join(data_dir, "movements.json")
screenshots_dir = os.path.join(data_dir, "screenshots")


# Define valid composite actions
def agent_action(filtered_keys):
    sprint = filtered_keys.get("B", False)
    move_right = filtered_keys.get("Right", False)
    move_left = filtered_keys.get("Left", False)
    jump = filtered_keys.get("A", False)
    crouch = filtered_keys.get("Down", False)
    airborne = filtered_keys.get("One", False)

    sprint_left = move_left and sprint
    sprint_right = move_right and sprint
    jump_left = (move_left or sprint_left) and jump
    jump_right = (move_right or sprint_right) and jump
    stand_still = not any([
        move_right, move_left, jump, crouch, airborne,
        sprint_left, sprint_right, jump_left, jump_right
    ])

    if any([crouch, airborne, sprint_left, jump]):
        stand_still = True
        crouch = airborne = sprint_left = jump = False

    return {
        "sprint_right": sprint_right,
        "jump_left": jump_left,
        "jump_right": jump_right,
        "move_right": move_right,
        "move_left": move_left,
        "none": stand_still,
    }

# Frame parsing
def extract_frame_number(key):
    match = re.search(r"frame_(\d+)", key)
    return int(match.group(1)) if match else -1

# Main cleaning logic
def clean_dataset():
    with open(movements_path, "r") as f:
        data = json.load(f)

    cleaned = {}
    removed = []
    
    for key, val in data.items():
        frame = extract_frame_number(key)
        if frame < 761 or frame > 6995:
            removed.append(key)
            continue

        inputs = {k: v for k, v in val.items() if k != "state"}
        actions = agent_action(inputs)
        if any(actions.get(k, False) for k in actions if k != "none"):
            cleaned[key] = val
        else:
            removed.append(key)

    # Save cleaned movement data
    with open(movements_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    # Delete corresponding screenshots
    deleted_screens = 0
    for key in removed:
        frame = extract_frame_number(key)
        for fname in os.listdir(screenshots_dir):
            if f"frame_{frame}.png" in fname:
                os.remove(os.path.join(screenshots_dir, fname))
                deleted_screens += 1

    print(f"Removed {len(removed)} invalid movement entries.")
    print(f"Deleted {deleted_screens} screenshots.")

# Run
if __name__ == "__main__":
    clean_dataset()
