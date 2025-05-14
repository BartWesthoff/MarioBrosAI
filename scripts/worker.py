import os
import sys
sys.path.append(os.path.join(os.getenv('LOCALAPPDATA'), 'Programs', 'Python', 'Python311', 'Lib', 'site-packages'))
sys.path.append(os.path.join(os.getenv('APPDATA'), 'Python', 'Python311', 'site-packages'))
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

import time
import threading
import pickle
import socket
from scripts.utils_func import generate_checkpoints, compute_reward, read_game_memory, is_game_in_state, detect_freeze, agent_action
import queue
import torch
import time

from dolphin import event, gui, memory, controller, savestate

import cv2
import numpy as np


def connect_to_socket():
    worker_id = int(os.environ['WORKER_ID'])
    host = 'localhost'
    port = 6000 + worker_id

    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            print(f"[SOCKET] Connected to manager at {host}:{port}")
            return s
        except Exception as e:
            print(f"[SOCKET] Waiting for manager... {e}")
            time.sleep(1)


send_queue = queue.Queue()
action_queue = queue.Queue()

def threaded_comm(conn):
    while True:
        try:
            experience = send_queue.get(block=True)  # blocks until available
            data = pickle.dumps(experience)
            size = len(data).to_bytes(4, "little")
            conn.sendall(size + data)

            # Wait for response
            size_bytes = conn.recv(4)
            size = int.from_bytes(size_bytes, "little")
            data = b""
            while len(data) < size:
                data += conn.recv(size - len(data))

            action = pickle.loads(data)
            action_queue.put(action)

        except Exception as e:
            print(f"[THREAD] Error in socket communication: {e}")
            action_queue.put(5)


def do_action(action):
    action_string = ACTION_KEYS[action]
    print(f"\t\t\tAction: {action_string}")
    button_map = {
        "sprint_right": {"Right": True, "B": True},
        "jump_left": {"Left": True, "A": True},
        "jump_right": {"Right": True, "A": True},
        "move_right": {"Right": True},
        "move_left": {"Left": True},
        "none": {}  # release all buttons
    }
    controller.set_wiimote_buttons(0, button_map.get(action_string, {}))

preprocess_queue = queue.Queue()
processed_queue = queue.Queue()

def threaded_preprocessor():
    while True:
        try:
            frames_batch, action, reward = preprocess_queue.get()

            # Process in background
            state = preprocess_frames_cv2(frames_batch[:4])
            next_state = preprocess_frames_cv2(frames_batch[1:])
            processed_queue.put((state, action, reward, next_state))

        except Exception as e:
            print(f"[PREPROCESSOR] Error: {e}")


def preprocess_frames_cv2(frames, target_size=(140, 114), save=False):
    img_arrays_list = []
    if len(frames) != 4:
        raise ValueError(f"Expected 4 frames, got {len(frames)}")

    for (width, height, raw_bytes) in frames:
        img_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 4))
        img_cropped = img_array[
            pos_start_screen[1]:pos_end_screen[1],
            pos_start_screen[0]:pos_end_screen[0]
        ]
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGBA2GRAY)
        img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32)
        img_arrays_list.append(img_float)


    img_arrays = np.stack(img_arrays_list)
    #img_array_width_height = img_arrays.transpose(0, 2, 1) # IMPORTANT: our model expects width height, our images are height width so we need to transpose

    return img_arrays


memory_queue = queue.Queue()
read_signal = threading.Event()
shutdown_signal = threading.Event()
def persistent_memory_reader():
    while not shutdown_signal.is_set():
        read_signal.wait()  # Block until signaled
        read_signal.clear()  # Reset signal for next use

        try:
            data = read_game_memory()
            if memory_queue.full():
                memory_queue.get_nowait()
            memory_queue.put(data)
        except Exception as e:
            print(f"[MEMORY_THREAD] Error: {e}")
            memory_queue.put(None)



# needed for reward calculation
previous_lives = None
previous_clock = None
previous_x = None
previous_mario_form = None
previous_checkpoint_idx = 0
stored_lives = None
stored_mario_form = None
stored_checkpoint_idx = 0
level1_start_x = 760
level1_last_cp = 6692
num_checkpoints = 10

# for cropping the input image
pos_start_screen = (200, 100-20)
pos_end_screen = (860-50, 500-50)
image_dim = (140, 114)


# action mappings
ACTION_KEYS = [
                "sprint_right",
                "jump_right",
                "jump_left",
                "move_right",
                "move_left", 
                "none"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_KEYS)}
NUM_ACTIONS = len(ACTION_KEYS)


class frameList:
    def __init__(self):
        self.length = 5
        self.frames = []
    
    def __len__(self):
        return len(self.frames)
    
    def getFrames(self):
        return self.frames

    def append(self, frame):
        if len(self.frames) >= self.length:
            self.frames.pop(0)
        self.frames.append(frame)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("------------------Starting worker script...--------------------")
    frame_count = 0
    frameStorage = frameList()
    rewards = []
    experience_list = [] # experiences (state, action, reward, next_state)
    checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)
    action = 5 # "none", default action for first frames
    data = None
    socket_conn = connect_to_socket()
    threading.Thread(target=threaded_comm, args=(socket_conn,), daemon=True).start()
    threading.Thread(target=threaded_preprocessor, daemon=True).start()
    threading.Thread(target=persistent_memory_reader, daemon=True).start()


    action_repeat = 4
    repeat_counter = 0

    # move past initial frames, to skip initial loading of game
    for i in range(10):
        await event.framedrawn()
        
    
    while True:
        print(f"Start loop, frameCount: {frame_count}")
        
        frameStorage.append(await event.framedrawn())           # frame 0
        do_action(action)
        print("preSet")
        await event.frameadvance()
        read_signal.set()
        print("postSet")

        frameStorage.append(await event.framedrawn())           # frame 1
        print("postSetDraw")
        do_action(action)
        
        old_data = data
        try:
            data = memory_queue.get_nowait()
            print("Data aquired")
        except queue.Empty:
            print("\nWARNING: data was not available yet, reusing old data")

       
        frameStorage.append(await event.framedrawn())           # frame 2
        do_action(action)
        print("Image aquired")
        reward, new_checkpoint_idx = reward, new_checkpoint_idx = compute_reward(
        data, previous_lives, previous_mario_form, previous_checkpoint_idx, checkpoints,
        previous_x, previous_clock
        )
        print("Reward computed")
        frameStorage.append(await event.framedrawn())           # frame 3
        do_action(action)
        
        if len(rewards) == 5:
            rewards.pop(0)
        rewards.append(reward)

        print(f"Rewards: {rewards}\tMean reward: {np.mean(rewards)}")

        if len(frameStorage) == 5:
            if repeat_counter == 0:
                preprocess_queue.put((frameStorage.getFrames(), action, reward))
                time.sleep(0.01)
                try:
                    state, action, reward, next_state = processed_queue.get_nowait()
                    experience = (state, action, reward, next_state)
                    experience_list.append(experience)
                    send_queue.put(experience)
                except queue.Empty:
                    print("\nWARNING:No processed data yet, skipping this frame\n")

                action = None
                for i in range(3):
                    try:
                        action = action_queue.get_nowait()
                        break
                    except queue.Empty:
                        print(f"[WORKER] No action received yet, retrying ({i+1}/3)...")
                        time.sleep(0.05)
                if action is None:
                    print("\nWARNING:No action received, defaulting to action 5: None\n")
                    action = 5

                repeat_counter = action_repeat

            repeat_counter -= 1

        frameStorage.append(await event.framedrawn())                                # frame 4
        do_action(action)
        #print(f"Post action")

        if previous_lives is not None and previous_lives > data['lives']:
            print("-------------------")
            print("Died, resetting state. Reward: ", reward)
            print("-------------------")
            await event.framedrawn()
            savestate.load_from_slot(1)
            await event.framedrawn()
        
        previous_lives = data['lives']
        previous_mario_form = data['mario_form']
        previous_checkpoint_idx = new_checkpoint_idx
        previous_time = data['current_time']
        frame_count += 1
        previous_x = data['cur_x']
        previous_clock = data['current_time']
        
        #frameStorage.append(await event.framedrawn())                                # frame 5
        #do_action(action)
        #print(f"End loop, frameCount: {frame_count}")
