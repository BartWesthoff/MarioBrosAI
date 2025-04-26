from dolphin import event, gui, memory, controller,savestate
import sys
from collections import deque
sys.path.append("C:\\Users\\bartw\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
import os
import json
from datetime import datetime
from PIL import Image
import pygetwindow as gw
# Colors
RED = 0xffff0000
BLACK = 0xff000000
GREEN = 0xff00ff00
CYAN = 0xff00ffff

# Globals
frame_counter = 0
previous_time = None
frozen_frame_count = 0
save_per_frames = 4
freeze_threshold = 60
pending_movements = {}
images_to_save = []
recent_rewards = deque(maxlen=300)  # Store last 5 seconds of rewards at 60 FPS
date = datetime.now().strftime("%Y-%m-%d_%H-%M")
mario_form_dict = {
    0: "Small", 1: "Large", 2: "Fire Flower", 3: "Mini",
    4: "Propeller Suit", 5: "Penguin Suit", 6: "Ice Flower"
}

# Screen Settings
pos_start_screen = (200, 100-20)
pos_end_screen = (860-50, 500-50)
middle_x = (pos_start_screen[0] + pos_end_screen[0]) // 2
middle_y = (pos_start_screen[1] + pos_end_screen[1]) // 2

# Level Settings
checkpoint_width = 100
level1_start_x = 760
level1_last_cp = 6692
num_checkpoints = 10
s_cp_box = 10
checkpoints = []
death_display_timer = 0
previous_lives = None
previous_mario_form = None
previous_checkpoint_idx = 0
stored_lives = None
stored_mario_form = None
stored_checkpoint_idx = 0
image_size = (224,224)

def set_window_size(window_title, width, height):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"{window_title} window not found!")
    window = windows[0]
    if window.isMinimized:
        window.restore()
    window.resizeTo(width, height)

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

def draw_debug_info(data, reward, mean_reward, is_frozen, in_game, death_display_timer, filtered_keys,curr_checkpoint_idx):
    action = agent_action(filtered_keys)
    gui.draw_text((10, 10), RED, f"Frame: {frame_counter}")
    gui.draw_text((10, 30), RED, f"X Coordinate: {data['cur_x']}")
    # gui.draw_text((10, 50), RED, f"X Coordinate 2: {data['second_x']}")
    gui.draw_text((10, 50), RED, f"Reward: {reward}")
    gui.draw_text((140, 50), RED, f"Mean Reward (5s): {mean_reward:.2f}")
    gui.draw_text((140, 60), RED, f"Checkpoint: {curr_checkpoint_idx+1}/{len(checkpoints)}")
    gui.draw_text((10, 70), RED, f"Is Large: {data['mario_form'] > 0}")
    gui.draw_text((10, 90), RED, f"Y Coordinate: {data['cur_y']}")
    gui.draw_text((10, 110), RED, f"Lives: {data['lives']}")
    gui.draw_text((10, 130), RED, f"Mario Form: {mario_form_dict.get(data['mario_form'], 'Unknown')}")
    gui.draw_text((10, 150), RED, f"Speed: {data['speed']}")
    gui.draw_text((10, 170), RED, f"Is In Game: {in_game}")
    chosen_action = [key for key, value in action.items() if value][0]
    gui.draw_text((10, 190), RED, f"Agent Action: {chosen_action}")
    gui.draw_text((50, 270), RED, f"Terminator: {data['termin']}")
    gui.draw_text((50, 290), RED, f"Time: {data['current_time']}")
    # draw box
    gui.draw_rect(
        (pos_start_screen[0], pos_start_screen[1]),
        (pos_end_screen[0], pos_end_screen[1]),
        RED, 1
    )

    if previous_time is not None:
        gui.draw_text((50, 310), RED, f"Previous Time: {previous_time}")
        gui.draw_text((50, 330), RED, f"Frozen Frames: {frozen_frame_count}")
    if is_frozen:
        gui.draw_text((50, 350), GREEN, "TIMER FROZEN!")

    if death_display_timer > 0:
        gui.draw_text((50, 370), CYAN, "DIED!")
        if is_frozen and is_in_game and data['cur_x'] > level1_start_x and data['termin'] == 1:
            savestate.load_from_slot(1)
     
    

def save_screenshots_and_movements(small_screenshot=False, crop_to_subscreen=False):
    if images_to_save:
        for img_data, frame in images_to_save:
            width, height, rgba_bytes = img_data
            image = Image.frombytes("RGBA", (width, height), rgba_bytes)

            if crop_to_subscreen:
                left, top = pos_start_screen
                right, bottom = pos_end_screen
                image = image.crop((left, top, right, bottom))

            if small_screenshot:
                image = image.convert("L").resize(image_size)
            
            image.save(f"screenshots\\d_{date}_frame_{frame}.png")
        images_to_save.clear()

    if pending_movements:
        last_frame = max(pending_movements.keys())
        pending_movements.pop(last_frame, None)
        movements_path = "movements.json"
        if os.path.exists(movements_path):
            with open(movements_path, "r") as f:
                all_movements = json.load(f)
        else:
            all_movements = {}
        all_movements.update(pending_movements)
        with open(movements_path, "w") as f:
            json.dump(all_movements, f, indent=4)
        pending_movements.clear()

def  is_game_in_state(data):
    return data['cur_x'] != 0.0 and data['cur_y'] != 0.0 and data['current_time'] > 0 and data['cur_x'] < 6700

def detect_freeze(current_time, previous_time, frozen_frame_count, termin, cur_x, cur_y):
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

checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)
set_window_size("Dolphin scripting-preview2-4802-dirty |", 860, 500)
auto_save = True
while True:
    await event.frameadvance()
    data = read_game_memory()
    reward, new_checkpoint_idx = compute_reward(
    data, previous_lives, previous_mario_form, previous_checkpoint_idx, checkpoints
    )
    recent_rewards.append(reward)
    mean_reward = sum(recent_rewards) / len(recent_rewards)

    b_is_pressed2 = controller.get_wiimote_buttons(0)
    keys_of_interest = ["A", "B", "One", "Left", "Right", "Down"]
    filtered_keys = {k: b_is_pressed2[k] for k in keys_of_interest if k in b_is_pressed2}

    frozen_frame_count, is_frozen = detect_freeze(
        data['current_time'], previous_time, frozen_frame_count, data['termin'], data['cur_x'], data['cur_y']
    )
    is_in_game = is_game_in_state(data)

    # Draw pressed keys
    y_offset = pos_start_screen[1]
    for key, pressed in filtered_keys.items():
        gui.draw_text((pos_end_screen[0] - 200, y_offset), RED, f"{key}: {'Pressed' if pressed else 'Released'}")
        y_offset += 20

    # Draw checkpoint box if player inside a checkpoint
    for start_x, end_x in checkpoints:
        if start_x <= data['cur_x'] <= end_x:
            gui.draw_rect_filled(
                (middle_x - s_cp_box, middle_y - s_cp_box),
                (middle_x + s_cp_box, middle_y + s_cp_box),
                CYAN, 1
            )
            break

    if previous_lives is not None:
            if data['lives'] < previous_lives:
                death_display_timer = 120  # Show for ~2 seconds (assuming 60 fps)



    draw_debug_info(data, reward, mean_reward, is_frozen, is_in_game, death_display_timer, filtered_keys, new_checkpoint_idx)


    if death_display_timer > 0:
        death_display_timer -= 1
    

    if frame_counter % save_per_frames == 0 and is_in_game and not is_frozen:
        width, height, rgba_bytes = await event.framedrawn()
        images_to_save.append(((width, height, rgba_bytes), frame_counter))
        pending_movements[f"{date}_frame_{frame_counter}"] = pending_movements[f"{date}_frame_{frame_counter}"] = {
    **filtered_keys,
    "state": {
        "reward": reward,
        "mean_reward": mean_reward,
        "cur_x": data["cur_x"],
        "cur_y": data["cur_y"],
        "lives": data["lives"],
        "mario_form": data["mario_form"],
        "speed": data["speed"],
        "checkpoint_idx": new_checkpoint_idx
    }
}

    if b_is_pressed2.get("Home") or (auto_save and data['cur_x'] > level1_last_cp+50):
        save_screenshots_and_movements(small_screenshot=True,crop_to_subscreen=True)


    if frame_counter % 10 == 0:
        stored_lives = data['lives']
        stored_mario_form = data['mario_form']
        stored_checkpoint_idx = new_checkpoint_idx

    # Always update previous_* immediately
    previous_lives = data['lives']
    previous_mario_form = data['mario_form']
    previous_checkpoint_idx = new_checkpoint_idx
    previous_time = data['current_time']
    frame_counter += 1