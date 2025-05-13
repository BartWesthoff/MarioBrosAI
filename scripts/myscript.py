from dolphin import event, gui, memory, controller, savestate
from dolphin import event, gui, controller, savestate
import sys
sys.path.append("C:\\Users\\maxja\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import deque
import os
import json
from datetime import datetime

# Add scripts directory to sys.path
cwd = os.getcwd()
if os.path.basename(cwd) == "dolphin" and "Experiment" in cwd:
    # If we're in Experiment/dolphin, go up two levels and then to scripts
    detect_freeze,
    agent_action,


# Check for optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available, screenshots will be disabled")

try:
    import pygetwindow as gw
    GW_AVAILABLE = True
except ImportError:
    GW_AVAILABLE = False
    print("pygetwindow not available, window resizing will be disabled")
from PIL import Image
import pygetwindow as gw

# Colors
RED = 0xffff0000
if is_frozen and is_in_game and data['cur_x'] > level1_start_x and data['termin'] == 1:
            savestate.load_from_slot(1)



def save_screenshots_and_movements(small_screenshot=False, crop_to_subscreen=False):
    # Save screenshots if PIL is available
    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
    if images_to_save:
        if PIL_AVAILABLE:
            for img_data, frame in images_to_save:
                width, height, rgba_bytes = img_data
                image = Image.frombytes("RGBA", (width, height), rgba_bytes)

                if crop_to_subscreen:
                    left, top = pos_start_screen
                    right, bottom = pos_end_screen
                    image = image.crop((left, top, right, bottom))

                if small_screenshot:
                    image = image.convert("L").resize(image_size)

                # Save screenshots in the Experiment/screenshots directory
                screenshots_dir = os.path.join(os.path.dirname(os.getcwd()), "screenshots")
                if not os.path.exists(screenshots_dir):
                    os.makedirs(screenshots_dir)
                image.save(os.path.join(screenshots_dir, f"d_{date}_frame_{frame}.png"))
        else:
            print("Screenshots disabled because PIL is not available")
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

    # Save movements data (doesn't require PIL)
    if pending_movements:
        last_frame = max(pending_movements.keys())
        pending_movements.pop(last_frame, None)
        # Save movements in the Experiment/data directory
        data_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        movements_path = os.path.join(data_dir, "movements.json")

checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)

# Set window size if pygetwindow is available
if GW_AVAILABLE:
    try:
        set_window_size("Dolphin scripting-preview2-4802-dirty |", 860, 500)
        print("Window size set successfully")
    except Exception as e:
        print(f"Error setting window size: {e}")
else:
    print("Window resizing disabled because pygetwindow is not available")

set_window_size("Dolphin scripting-preview2-4802-dirty |", 860, 500)
auto_save = True
while True:
    await event.frameadvance()

    if frame_counter % save_per_frames == 0 and is_in_game and not is_frozen:
        # Capture screenshot if PIL is available
        if PIL_AVAILABLE:
            try:
                width, height, rgba_bytes = await event.framedrawn()
                images_to_save.append(((width, height, rgba_bytes), frame_counter))
            except Exception as e:
                print(f"Error capturing screenshot: {e}")
         
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