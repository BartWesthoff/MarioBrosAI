from dolphin import event, gui
from dolphin import memory

# Colors
red = 0xffff0000
black = 0xff000000
green = 0xff00ff00
orange = 0xffffa500
cyan = 0xff00ffff
frame_counter = 0
previous_time = None
frozen_frame_count = 0
freeze_threshold = 60  # Number of frames to consider it "frozen"
def generate_checkpoints(level_start: float, level_end: float, num_checkpoints: int, width: float):
    """
    Returns a list of (start_x, end_x) checkpoint tuples.
    - width: width of each checkpoint box (world units)
    """
    checkpoints = []
    spacing = (level_end - level_start - width) / (num_checkpoints)
    for i in range(num_checkpoints+1):
        start = level_start + i * spacing
        checkpoints.append((start, start + width))
    return checkpoints[1:]

while True:
    await event.frameadvance()  # Wait for the next frame

    # Read game memory
    cur_x = memory.read_f32(0x815E425C)
    cur_y = memory.read_f32(0x815E38E4)
    lives = memory.read_u8(0x80355193)
    has_prop_1 = memory.read_u8(0x803FCCCB)
    has_prop_2A = memory.read_u8(0x803AF000)
    is_large = memory.read_u8(0x8154C8C1)
    second_x = memory.read_f32(0x815E4290)
    termin = memory.read_f64(0x8043CA78)
    current_time = memory.read_u32(0x81547900)

    # Freeze detection
    if previous_time is not None:
        if current_time == previous_time:
            frozen_frame_count += 1
        else:
            frozen_frame_count = 0  # Reset counter on change

    is_frozen = frozen_frame_count >= freeze_threshold and cur_x != 0.0 and cur_y != 0.0 and termin == 1 and current_time >0
    is_in_game = cur_x != 0.0 and cur_y != 0.0 and current_time >0
    # Draw GUI
    pos_start_screen = (105, 51)
    pos_end_screen = (830, 456)
    middle_x  = (pos_start_screen[0] + pos_end_screen[0]) // 2
    middle_y  = (pos_start_screen[1] + pos_end_screen[1]) // 2
    gui.draw_rect(pos_start_screen, pos_end_screen, red, 0, 1)
    checkpoint_width = 100
    checkpoint1_start = 3030
    level1_last_cp = 6704 + 5 # margin
    level1_start_x = 760
    # draw a box in the middle of the screen
    s_cp_box = 10
    num_checkpoints = 10
    checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints, checkpoint_width)

    # Inside loop
    s_cp_box = 10
    for start_x, end_x in checkpoints:
        if start_x <= cur_x <= end_x:
            gui.draw_rect_filled(
                (middle_x - s_cp_box, middle_y - s_cp_box),
                (middle_x + s_cp_box, middle_y + s_cp_box),
                cyan, 1
            )
            break
    # draw completion bar using cur_x and level1_start_x
    completion_bar_width = 200
    completion_bar_height = 10
    completion_bar_x = pos_start_screen[0] + (pos_end_screen[0] - pos_start_screen[0]) // 2 - completion_bar_width // 2
    completion_bar_y = pos_start_screen[1] - 20
    completion_percentage = (cur_x - level1_start_x) / (level1_last_cp - level1_start_x)
    completion_bar_filled_width = int(completion_bar_width * completion_percentage)

    # Draw the black box around the completion bar
    if is_in_game:
        gui.draw_rect((completion_bar_x, completion_bar_y),
                    (completion_bar_x + completion_bar_width, completion_bar_y + completion_bar_height), black, 1)

        # Draw the filled portion of the completion bar
        gui.draw_rect_filled((completion_bar_x, completion_bar_y),
                            (completion_bar_x + completion_bar_filled_width, completion_bar_y + completion_bar_height), green, 1)
    


        gui.draw_text((10, 10), red, f"Frame: {frame_counter}")
        gui.draw_text((10, 30), red, f"X Coordinate: {cur_x}")
        gui.draw_text((10, 50), red, f"X Coordinate 2: {second_x}")
        gui.draw_text((10, 70), red, f"Is Large: {is_large}")
        gui.draw_text((10, 90), red, f"Y Coordinate: {cur_y}")
        gui.draw_text((10, 110), red, f"Lives: {lives}")
        gui.draw_text((10, 130), red, f"Has Prop 1: {has_prop_1}")
        gui.draw_text((10, 150), red, f"Has Prop 2A: {has_prop_2A}")
        gui.draw_text((50, 270), red, f"Terminator: {termin}")
        gui.draw_text((50, 290), red, f"Time: {current_time}")
            # Update for next frame
        previous_time = current_time
        frame_counter += 1
    if previous_time is not None:
        gui.draw_text((50, 310), red, f"Previous Time: {previous_time}")
        gui.draw_text((50, 330), red, f"Frozen Frames: {frozen_frame_count}")

    if is_frozen:
        gui.draw_text((50, 350), green, f"TIMER FROZEN!")

    # Screenshot save (optional)
    if frame_counter % 1000 == 1 and is_in_game:
        width, height, rgba_bytes = await event.framedrawn()
        rgb_bytes = bytearray()
        for i in range(0, len(rgba_bytes), 4):
            rgb_bytes.extend(rgba_bytes[i:i+3])
        filename = f"screenshots/frame_{frame_counter:05d}.ppm"
        with open(filename, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.write(rgb_bytes)


