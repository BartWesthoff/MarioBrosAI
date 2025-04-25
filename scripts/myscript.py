from dolphin import event, gui, memory, controller
import sys
import os
import json
from datetime import datetime
from PIL import Image
import pygetwindow as gw

# Setup
sys.path.append("C:\\Users\\bartw\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages")

# Colors
RED = 0xffff0000
BLACK = 0xff000000
GREEN = 0xff00ff00
CYAN = 0xff00ffff

# Globals
frame_counter = 0
previous_time = None
frozen_frame_count = 0
freeze_threshold = 60
pending_movements = {}
images_to_save = []

date = datetime.now().strftime("%Y-%m-%d_%H-%M")
mario_form_dict = {
    0: "Small", 1: "Large", 2: "Fire Flower", 3: "Mini",
    4: "Propeller Suit", 5: "Penguin Suit", 6: "Ice Flower"
}

# Screen Settings
pos_start_screen = (105, 51)
pos_end_screen = (830, 456)
middle_x = (pos_start_screen[0] + pos_end_screen[0]) // 2
middle_y = (pos_start_screen[1] + pos_end_screen[1]) // 2

# Level Settings
checkpoint_width = 100
level1_start_x = 760
level1_last_cp = 6709  # 6704 + 5 margin
num_checkpoints = 10
s_cp_box = 10
checkpoints = []
death_display_timer = 0  # Counts how long to show the death message
previous_lives = None


def set_window_size(window_title, width, height):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"{window_title} window not found!")
    window = windows[0]
    if window.isMinimized:
        window.restore()
    window.resizeTo(width, height)

def generate_checkpoints(level_start, level_end, num_checkpoints, width):
    spacing = (level_end - level_start - width) / num_checkpoints
    return [(level_start + i * spacing, level_start + i * spacing + width) for i in range(1, num_checkpoints + 1)]

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

def draw_debug_info(data, is_frozen,in_game, death_display_timer):
    gui.draw_text((10, 10), RED, f"Frame: {frame_counter}")
    gui.draw_text((10, 30), RED, f"X Coordinate: {data['cur_x']}")
    gui.draw_text((10, 50), RED, f"X Coordinate 2: {data['second_x']}")
    gui.draw_text((10, 70), RED, f"Is Large: {data['mario_form'] > 0}")
    gui.draw_text((10, 90), RED, f"Y Coordinate: {data['cur_y']}")
    gui.draw_text((10, 110), RED, f"Lives: {data['lives']}")
    gui.draw_text((10, 130), RED, f"Mario Form: {mario_form_dict.get(data['mario_form'], 'Unknown')}")
    gui.draw_text((10, 170), RED, f"Speed: {data['speed']}")
    gui.draw_text((10, 190), RED, f"Is In Game: {in_game}")
    gui.draw_text((50, 270), RED, f"Terminator: {data['termin']}")
    gui.draw_text((50, 290), RED, f"Time: {data['current_time']}")
    if previous_time is not None:
        gui.draw_text((50, 310), RED, f"Previous Time: {previous_time}")
        gui.draw_text((50, 330), RED, f"Frozen Frames: {frozen_frame_count}")
    if is_frozen:
        gui.draw_text((50, 350), GREEN, "TIMER FROZEN!")

    if death_display_timer > 0:
        gui.draw_text((50, 370), CYAN, "DIED!")

def save_screenshots_and_movements():
    if images_to_save:
        for img_data, frame in images_to_save:
            width, height, rgba_bytes = img_data
            image = Image.frombytes("RGBA", (width, height), rgba_bytes)
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

def is_game_in_state(data):
    return data['cur_x'] != 0.0 and data['cur_y'] != 0.0 and data['current_time'] > 0

def detect_freeze(current_time, previous_time, frozen_frame_count, termin, cur_x, cur_y):
    if previous_time is not None and current_time == previous_time:
        frozen_frame_count += 1
    else:
        frozen_frame_count = 0
    is_frozen = frozen_frame_count >= freeze_threshold and cur_x != 0.0 and cur_y != 0.0 and termin == 1
    return frozen_frame_count, is_frozen

# Initialize checkpoints
checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints, checkpoint_width)

# Main loop
while True:
    await event.frameadvance()
    data = read_game_memory()
    b_is_pressed2 = controller.get_wiimote_buttons(0)
    keys_of_interest = ["A", "B", "One", "Two", "Left", "Right", "Down"]
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

    previous_lives = data['lives']

    draw_debug_info(data, is_frozen, is_in_game, death_display_timer)
    previous_time = data['current_time']
    frame_counter += 1

    if death_display_timer > 0:
        death_display_timer -= 1
    

    if frame_counter % 60 == 0 and is_in_game and not is_frozen:
        width, height, rgba_bytes = await event.framedrawn()
        images_to_save.append(((width, height, rgba_bytes), frame_counter))
        pending_movements[f"{date}_frame_{frame_counter}"] = filtered_keys

    if b_is_pressed2.get("Home"):
        save_screenshots_and_movements()
