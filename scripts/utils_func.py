from dolphin import event, gui, memory, controller, savestate
import sys
import os

# Define colors
RED = 0xffff0000
BLACK = 0xff000000
GREEN = 0xff00ff00
CYAN = 0xff00ffff

# Try to import pygetwindow, but provide a fallback if it's not available
try:
    import pygetwindow as gw
    GW_AVAILABLE = True
    print("pygetwindow imported successfully in utils_func.py")
except ImportError as e:
    print(f"pygetwindow import error in utils_func.py: {e}")
    print("pygetwindow not available, window resizing will be disabled")
    GW_AVAILABLE = False

def set_window_size(window_title, width, height):
    """Set the size of a window with the given title.

    If pygetwindow is not available, this function will do nothing.
    """
    if not GW_AVAILABLE:
        print(f"Cannot resize window: pygetwindow module not available")
        return

    try:
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            print(f"Window '{window_title}' not found!")
            return
        window = windows[0]
        if window.isMinimized:
            window.restore()
        window.resizeTo(width, height)
        print(f"Window resized to {width}x{height}")
    except Exception as e:
        print(f"Error resizing window: {e}")


def compute_reward(data, previous_lives, previous_mario_form, previous_checkpoint_idx, checkpoints):
    reward = 0.0
    new_checkpoint_idx = previous_checkpoint_idx

    if previous_lives is not None and data['lives'] < previous_lives:
        reward -= 10

    if previous_mario_form is not None:
        if data['mario_form'] < previous_mario_form:
            reward -= 10
        elif data['mario_form'] > previous_mario_form:
            reward += 10

    reward += data['speed'] / 3.0

    # Only progress if entering a *higher* checkpoint
    for idx, (start_x, end_x) in enumerate(checkpoints):
        if start_x <= data['cur_x'] <= end_x:
            if idx > previous_checkpoint_idx:
                reward += 10
                if idx == len(checkpoints) - 1:
                    reward += 90000  # Big bonus at final checkpoint
                new_checkpoint_idx = idx
            break

    reward -= 0.05
    return round(reward, 2), new_checkpoint_idx


def generate_checkpoints(level_start, level_end, num_checkpoints):
    """Generate num_checkpoints - 1 evenly spaced checkpoints + 1 near the end."""
    if num_checkpoints < 1:
        return []

    # Always reserve the last checkpoint at end - small margin
    final_checkpoint = (level_end - 8, level_end + 8)  # ±8 units around 6692
    if num_checkpoints == 1:
        return [final_checkpoint]

    # Spread the other checkpoints across the course
    checkpoints = []
    spacing = (level_end - level_start) / (num_checkpoints)

    for i in range(num_checkpoints - 1):
        start = level_start + i * spacing
        end = start + spacing
        checkpoints.append((start, end))

    checkpoints.append(final_checkpoint)
    return checkpoints


def read_game_memory():
    return {
        "cur_x": memory.read_f32(0x815E425C),
        "cur_y": memory.read_f32(0x815E38E4),
        "lives": memory.read_u8(0x80355193),
        "mario_form": memory.read_u8(0x811FDFF6),
        "second_x": memory.read_f32(0x815E4290),
        "termin": memory.read_f64(0x8043CA78),
        "current_time": memory.read_u32(0x81547900),
        "speed": memory.read_f32(0x8154B8C8),
    }

def  is_game_in_state(data):
    return data['cur_x'] != 0.0 and data['cur_y'] != 0.0 and data['current_time'] > 0 and data['cur_x'] < 6700

def detect_freeze(freeze_threshold, current_time, previous_time, frozen_frame_count, termin, cur_x, cur_y):
    if previous_time is not None and current_time == previous_time:
        frozen_frame_count += 1
    else:
        frozen_frame_count = 0
    is_frozen = frozen_frame_count >= freeze_threshold and cur_x != 0.0 and cur_y != 0.0 and termin == 1
    return frozen_frame_count, is_frozen

def agent_action(filtered_keys):
    sprint = filtered_keys.get("B", False)
    move_right = filtered_keys.get("Right", False)
    move_left = filtered_keys.get("Left", False)

    jump = filtered_keys.get("A", False)
    crouch = filtered_keys.get("Down", False)
    airbone = filtered_keys.get("One", False)
    sprint_left = move_left and sprint
    sprint_right = move_right and sprint
    jump_left = move_left and jump
    jump_right = move_right and jump
    stand_still = not any([move_right, move_left, jump, crouch, airbone, sprint_left, sprint_right, jump_left, jump_right])
    return {

        "jump": jump,
        "crouch": crouch,
        "airbone": airbone,
        "sprint_left": sprint_left,
        "sprint_right": sprint_right,
        "jump_left": jump_left,
        "jump_right": jump_right,
        "move_right": move_right,
        "move_left": move_left,
        "none": stand_still,
    }