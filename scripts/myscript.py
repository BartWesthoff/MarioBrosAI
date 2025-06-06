from dolphin import event, gui, controller, savestate,memory
import sys
import os

sys.path.append(os.path.join(os.getenv('LOCALAPPDATA'), 'Programs', 'Python', 'Python311', 'Lib', 'site-packages'))
sys.path.append(os.path.join(os.getenv('APPDATA'), 'Python', 'Python311', 'site-packages'))

from collections import deque
import json
from datetime import datetime
cwd = os.getcwd()
data_dir = os.path.abspath(os.path.join(cwd , "data2"))
scripts_dir = os.path.abspath(os.path.join(cwd, "scripts"))
screenshots_dir = os.path.join(data_dir, "screenshots")
movements_path = os.path.join(data_dir, "movements.json")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

if not os.path.exists(movements_path):
    with open(movements_path, "w") as f:
        json.dump({}, f, indent=4)

if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
print(f"Added {scripts_dir} to sys.path")
# Import utils_func
from scripts.utils_func import (
    set_window_size,
    generate_checkpoints,
    compute_reward,
    read_game_memory,
    is_game_in_state,
    detect_freeze,
    agent_action,
)
from PIL import Image

# Colors
RED = 0xffff0000
BLACK = 0xff000000
GREEN = 0xff00ff00
CYAN = 0xff00ffff

# Globals
frame_counter = 0
previous_time = None
frozen_frame_count = 0
save_per_frames = 1
freeze_threshold = 60
pending_movements = {}
images_to_save = []
recent_rewards = deque(maxlen=save_per_frames)  # Store last 5 seconds of rewards at 60 FPS
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
checkpoint4 = 2500
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
previous_x = None
previous_clock = None
stored_checkpoint_idx = 0
image_size = (140,114)


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
    gui.draw_text((10, 170), GREEN if in_game else RED, f"Is In Game: {in_game}")
    chosen_action = [key for key, value in action.items() if value]
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
            savestate.load_from_slot(3)
    if data['cur_x'] > checkpoint4 and data['termin'] == 1:
        gui.draw_text((50, 390), CYAN, "Checkpoint 4 reached!")
        if is_in_game:
            savestate.load_from_slot(3)

def save_screenshots_and_movements(small_screenshot=False, crop_to_subscreen=False):
    
    gui.draw_text((pos_start_screen[0], pos_start_screen[1]), RED, "Saving screenshots and movements...")
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

            # Save screenshots in the data/screenshots directory
            screenshots_dir = os.path.join(data_dir, "screenshots")
            if not os.path.exists(screenshots_dir):
                os.makedirs(screenshots_dir)
            image.save(os.path.join(screenshots_dir, f"d_{date}_frame_{frame}.png"))
        images_to_save.clear()
    if pending_movements:
        last_frame = max(pending_movements.keys())
        pending_movements.pop(last_frame, None)
        # Save movements in the Experiment/data directory
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        movements_path = os.path.join(data_dir, "movements.json")

        if os.path.exists(movements_path):
            with open(movements_path, "r") as f:
                all_movements = json.load(f)
        else:
            all_movements = {}
        all_movements.update(pending_movements)
        with open(movements_path, "w") as f:
            json.dump(all_movements, f, indent=4)
        pending_movements.clear()

checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)
set_window_size("Dolphin scripting-preview2-4802-dirty |", 860, 500)
auto_save = True
while True:
    await event.frameadvance()
    data = read_game_memory()
    reward, new_checkpoint_idx = reward, new_checkpoint_idx = compute_reward(
    data, previous_lives, previous_mario_form, previous_checkpoint_idx, checkpoints,
    previous_x, previous_clock
    )

    recent_rewards.append(reward)
    mean_reward = sum(recent_rewards) / len(recent_rewards)

    b_is_pressed2 = controller.get_wiimote_buttons(0)
    keys_of_interest = ["A", "B", "One", "Left", "Right", "Down"]
    filtered_keys = {k: b_is_pressed2[k] for k in keys_of_interest if k in b_is_pressed2}

    frozen_frame_count, is_frozen = detect_freeze(
        freeze_threshold, data['current_time'], previous_time, frozen_frame_count, data['termin'], data['cur_x'], data['cur_y']
    )
    is_in_game = is_game_in_state(data)

    # Draw pressed keys
    y_offset = pos_start_screen[1]
    for key, pressed in filtered_keys.items():
        gui.draw_text((pos_end_screen[0] - 200, y_offset), RED, f"{key}: {'Pressed' if pressed else 'Released'}")
        y_offset += 20

    if previous_lives is not None:
            if data['lives'] < previous_lives:
                death_display_timer = 120  # Show for ~2 seconds (assuming 60 fps)

    draw_debug_info(data, reward, mean_reward, is_frozen, is_in_game, death_display_timer, filtered_keys, new_checkpoint_idx)


    if death_display_timer > 0:
        death_display_timer -= 1


    if frame_counter % save_per_frames == 0 and is_in_game and not is_frozen:
        # Capture screenshot if PIL is available
        width, height, rgba_bytes = await event.framedrawn()
        images_to_save.append(((width, height, rgba_bytes), frame_counter))
      
        # Save movement data (doesn't require PIL)
        pending_movements[f"{date}_frame_{frame_counter}"] = {
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

    if b_is_pressed2.get("Home") or (auto_save and data['cur_x'] > checkpoint4+50):
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
    previous_x = data['cur_x']
    previous_clock = data['current_time']