from dolphin import gui, event

frame_counter = 0

while True:
    await event.frameadvance()
    gui.draw_text((50, 50), 0xff00ff00, f"Hello from script! Frame {frame_counter}")
    frame_counter += 1
