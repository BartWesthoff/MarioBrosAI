import os
import sys
sys.path.append(os.path.join(os.getenv('LOCALAPPDATA'), 'Programs', 'Python', 'Python311', 'Lib', 'site-packages'))
sys.path.append(os.path.join(os.getenv('APPDATA'), 'Python', 'Python311', 'site-packages'))
from PIL import Image
import mss
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from scripts.model import BTRNetwork
from scripts.utils_func import set_window_size, generate_checkpoints

import threading
import queue


from dolphin import event, gui, memory, controller,savestate
import pygetwindow as gw
import random
import time
from collections import deque

level1_start_x = 760
level1_last_cp = 6692
num_checkpoints = 10

height = 84
width = 84

action_mapping = {
    0: "A",
    1: "B",
    2: "One",
    3: "Two",
    4: "Up",
    5: "Down",
    6: "Left",
    7: "Right",
}

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        if len(self.buffer) >= self.buffer.maxlen:
            self.buffer.popleft()
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def prepareWindow(title="Dolphin scripting-preview2-4802-dirty |"):
    try:
        set_window_size(title,  860, 500)
        window = gw.getWindowsWithTitle(title)[0]
        if window.isMinimized:
            window.restore()
    except Exception as e:
        print(f"Couldn't find window: {e}")
        return None
    print("Window found and resized")
    return window



def get_frame(window):
    with mss.mss() as sct:
        monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
        img = sct.grab(monitor)
        img = Image.frombytes("RGB", img.size, img.rgb)
        img = img.convert("L") # to grayscale

        return img

def preprocess_frame(frame):
    img = frame.resize((width, height))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
    
def get_action(frames):
    print(f"Entered get action at {time.time()} with {len(frames)} frames")
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    # TODO: [:4] shouldnt be needed per se but just to be sure for now
    stacked_frames = np.stack(preprocessed_frames[:4], axis=0)
    # shape = (4,1,84,84) need to remove 1
    stacked_frames = np.squeeze(stacked_frames, axis=1)
    # now shape = (4,84,84)
    tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # now shape = (1,4,84,84)
    q_values = model(tensor)
    print(f"Shape of q_values: {q_values.shape}")
    q_value_max = q_values.max(dim=1)[0]
    action = q_value_max.argmax().item()

    return action


def do_action(action):
    controller.set_wiimote_buttons(0, {action_mapping[action]: True})


def capture_frames(window):
    frame_skip = 5
    frame_skip_counter = 0
    while True:
        if frame_skip_counter == frame_skip:
            frame = get_frame(window)  # Capture frame
            if frame_queue.qsize() < 4:
                frame_queue.put(frame)  # Push to queue
            frame_skip_counter = 0
        else:
            frame_skip_counter += 1

def action_loop():
    window = None
    frames = []
    while True:
        if not window:
            window = prepareWindow()

        if frame_queue.qsize() >= 4:
            # Pop frames from queue
            frames = [frame_queue.get() for _ in range(4)]

            # Get action
            action = get_action(frames)
            do_action(action)

checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)
model = BTRNetwork(num_actions=len(action_mapping))
frame_queue = queue.Queue(maxsize=4)

if __name__ == "__main__":

    window = None
    while window is None:
        window = prepareWindow()
        if window is None:
            print("Window not found, retrying in 5 seconds...")
            time.sleep(5)
    print("Window found, starting main loop...")

    frame_capture_thread = threading.Thread(target=capture_frames, args=(window,))
    frame_capture_thread.daemon = True
    frame_capture_thread.start()

    action_loop_thread = threading.Thread(target=action_loop)
    action_loop_thread.daemon = True
    action_loop_thread.start()

    while True:
        await event.frameadvance()
