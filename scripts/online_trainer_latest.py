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
from PIL import Image
from model import BTRNetwork

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def threaded_socket_sender(conn):
    while True:
        
        try:
            batch = send_queue.get(block=False)
        except Exception:
            time.sleep(0.5)
            continue
        try:
            serialized = pickle.dumps(batch)
            size = len(serialized).to_bytes(4, byteorder="little")
            conn.sendall(size + serialized)

            print(f"[SOCKET] Sent {len(batch)} compressed experiences.")
        except Exception as e:
            print(f"[SOCKET ERROR] Could not send to manager: {e}")

def listen_for_reload(model, model_path='latest_model.pth', port_range=7000):
    port = port_range + int(os.environ['WORKER_ID'])
    def handler():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("localhost", port))
        s.listen(1)
        print(f"[WORKER] Reload listener running on port {port}")
        while True:
            conn, _ = s.accept()
            data = conn.recv(1024)
            if b"reload" in data:
                with reloading_lock:
                    try:
                        model.load_state_dict(torch.load(model_path))
                        model.eval()
                        print("[WORKER] Reloaded updated model.")
                    except Exception as e:
                        print(f"[WORKER] Failed to reload model: {e}")
            conn.close()
    threading.Thread(target=handler, daemon=True).start()


def do_action(action):
    action_string = ACTION_KEYS[action]
    print(f"\t\t\tAction: {action_string}") # the \t's are for separating from rest of logs, so we can see it better
                                            # To view logs, open dolphin, view tab->Show log, can even split it off into separate window
    match action_string:
        case "jump":
            controller.set_wiimote_buttons(0, {"A": True})
        case "crouch":
            controller.set_wiimote_buttons(0, {"Down": True})
        case "airborne":
            controller.set_wiimote_buttons(0, {"One": True})
        case "sprint_left":
            controller.set_wiimote_buttons(0, {"Left": True, "B": True})
        case "sprint_right":
            controller.set_wiimote_buttons(0, {"Right": True, "B": True})
        case "jump_left":
            controller.set_wiimote_buttons(0, {"Left": True, "A": True})
        case "jump_right":
            controller.set_wiimote_buttons(0, {"Right": True, "A": True})
        case "move_right":
            controller.set_wiimote_buttons(0, {"Right": True})
        case "move_left":
            controller.set_wiimote_buttons(0, {"Left": True})
        case "none":
            # TODO: do we set all to none or just do nothing?
            pass

# only used for getAction, and not for sending over socket. Because this is not pickle-able
def process_frames(frames, target_size=(140, 114), frame_window=4):
    imgs = []
    if len(frames) != frame_window:
        raise ValueError(f"Expected {frame_window} frames, got {len(frames)}")
    for (width,height,data) in frames:
        img = Image.frombytes("RGBA", (width, height), data)
        img = img.crop((pos_start_screen[0], pos_start_screen[1], pos_end_screen[0], pos_end_screen[1]))

        # resize if needed
        img = img.convert("L")
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_tensor = torch.tensor(img_array).unsqueeze(0)  # Add channel dimension (1, H, W)
        if img_tensor.shape != (1, target_size[1], target_size[0]):
            raise ValueError(f"Expected image shape (1, {target_size[1]}, {target_size[0]}), got {img_tensor.shape}")
        imgs.append(img_tensor)

    state = torch.cat(imgs, dim=0)
    state = state.unsqueeze(0)

    return state


def getAction(frames):
    if len(frames) != 4:
        raise ValueError(f"Expected 4 frames, got {len(frames)}")
    state = process_frames(frames).to(device)

    for i in range(3):
        if reloading_lock.locked():
            print("Model is reloading, waiting...")
            time.sleep(0.1)
     
    if reloading_lock.locked():
        print(r"Model is still reloading, returning default action: 'none'.")
        return ACTION_TO_INDEX["none"]
    
    q_values = model(state)
    q_value_max = q_values.max(dim=1)[0]
    action = q_value_max.argmax().item()

    return action


def threaded_get_action(frames):
    try:
        action = getAction(frames)
        action_queue.put(action)
    except Exception as e:
        print(f"Error in getAction thread: {e}")
        action_queue.put(None)
    
# for pickable data, we keep it unprocessed so we save cpu time. We could also use a separate thread for processing
# this data and then send it over via socket. This might also allow for larger send_batch_size because 
# at least for me, copy() of send_batch_size > 16 (roughly havent tested limits) causes dolphin to freeze up.
def preprocess_frames(frames):
    if len(frames) != 4:
        raise ValueError(f"Expected 4 frames, got {len(frames)}")
    return [(width, height, np.frombuffer(img, dtype=np.uint8)) for (width, height, img) in frames]

# this version is outdated, but kept for reference. Scrapped it since it required more cpu time
# and it can freeze dolphin
def preprocess_frames2(frames):
    img_arrays_list = []
    if len(frames) != 4:
        raise ValueError(f"Expected 4 frames, got {len(frames)}")
    for (width, height, img) in frames:
        img = Image.frombytes("RGBA", (width, height), img)
        img = img.crop((pos_start_screen[0], pos_start_screen[1], pos_end_screen[0], pos_end_screen[1]))

        img = img.convert("L")
        img = img.resize((140, 114), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_arrays_list.append(img_array)
    
    return img_arrays_list




# needed for reward calculation
previous_lives = None
previous_time = None
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

# queue storage for threading and lock for reloading the updated model from manager
action_queue = queue.Queue()
send_queue = queue.Queue()
reloading_lock = threading.Lock()

# action mappings
ACTION_KEYS = ["jump", "crouch", "airborne", "sprint_left", "sprint_right",
               "jump_left", "jump_right", "move_right", "move_left", "none"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_KEYS)}



# loading in the base model
model = BTRNetwork(len(ACTION_KEYS)).to(device)
target_model = BTRNetwork(len(ACTION_KEYS)).to(device)

model.load_state_dict(torch.load("btr_model.pth"))
target_model.load_state_dict(torch.load("btr_model.pth"))

print("Model loaded")


# code for threaded experience sending and model reloading
send_batch_size = 10
# -- below can be combined into 1, like listen_for_reload
socket_conn = connect_to_socket()
threading.Thread(target=threaded_socket_sender, args=(socket_conn,), daemon=True).start()
# --
threading.Thread(target=listen_for_reload, args=(model,), daemon=True).start()




# In the main loop I try to place frame drawn after cpu intensive tasks, so that the game doesnt freeze or
# slow its fps too much. I might have placed too many, so we should keep this in mind when we consider bugs or 
# weird behavior. Not sure if this explains the agent learning action "none" though.
if __name__ == "__main__":

    frame_count = 0
    frames = []
    rewards = []
    experience_list = [] # experiences (state, action, reward, next_state)
    checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)
    action = 9 # "none", default action for first frames
    
    # move past initial frames, to skip initial loading of game
    for i in range(10):
        (width,height,img_data) = await event.framedrawn()
    
    while True:
        print(f"Start loop, frameCount: {frame_count}")
        data = read_game_memory()

        (width,height,img_data) = await event.framedrawn() # reading game_memo takes long so we place frame after it
        reward, new_checkpoint_idx = compute_reward(
        data, previous_lives, previous_mario_form, previous_checkpoint_idx, checkpoints
        )

        await event.framedrawn()
        frames.append((width, height, img_data)) # the frame data will be share over socket raw, this saves cpu time
        rewards.append(reward)

        print(f"Rewards: {rewards}\tMean reward: {np.mean(rewards)}")

        if len(frames) == 5:
            # # only use first 4, the last 4 is for next_state
            threading.Thread(
            target=threaded_get_action,
                args=([frames[:4]]),
                daemon=True
            ).start()
            action = action_queue.get()
            while action is None:
                print("Waiting for action...") # doesnt appear to happen, maybe its fast enough
                time.sleep(0.05)
                action = action_queue.get()
            
            
            state = preprocess_frames(frames[:4])
            await event.framedrawn()

            next_state = preprocess_frames(frames[1:])
            experience = (state, action, reward, next_state)
            experience_list.append(experience)

            frames.pop(0)
            rewards.pop(0)
        
        do_action(action)
        await event.framedrawn()

        
        if previous_lives is not None and previous_lives > data['lives']:
            print("-------------------")
            print("Died, resetting state. Reward: ", reward)
            print("-------------------")
            savestate.load_from_slot(1)
        
        # draw a second frame to increase fps
        frame_count += 1
        previous_lives = data['lives']
        previous_mario_form = data['mario_form']
        previous_checkpoint_idx = new_checkpoint_idx
        previous_time = data['current_time']
        await event.framedrawn()
        # if len of data is above some threshold write to manager process via pipe
        if len(experience_list) >= send_batch_size:
            send_queue.put(experience_list.copy())
            experience_list.clear()
        await event.framedrawn()
        await event.frameadvance()      # <-- idk why, but this stops it from freezing for me. Without it, it freezes after like 7 seconds
                                        # I think its because of readmemory taking a long time. My suspicion is that somehow this 
                                        # double frame allows dolphin to catch up which in turns makes it so that readmemory doesnt freeze dolphin 
                                        # also it appears to need to be frameadvance not framedrawn, otherwise it still freezes