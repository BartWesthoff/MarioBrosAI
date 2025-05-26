import os
import sys
# Original paths
sys.path.append(os.path.join(os.getenv('LOCALAPPDATA'), 'Programs', 'Python', 'Python311', 'Lib', 'site-packages'))
sys.path.append(os.path.join(os.getenv('APPDATA'), 'Python', 'Python311', 'site-packages'))
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

# Add virtual environment path
venv_path = os.path.join(os.getcwd(), 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.append(venv_path)
    print(f"[WORKER] Added venv path: {venv_path}")

from utils_func import agent_action
import time
import threading
import pickle
import socket
from scripts.utils_func import generate_checkpoints, compute_reward, read_game_memory, is_game_in_state, detect_freeze, agent_action
import queue
import time
from dolphin import event, gui, memory, controller, savestate
from collections import deque
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


send_queue = queue.Queue(maxsize=50) # shouldnt have a small limit, this 50 is just to prevent exploding memory in case of mistiming
action_queue = queue.Queue(maxsize=1) # to prevent action lagging
reward_send_queue = queue.Queue()

def reward_sender_thread(conn):
    while True:
        try:
            avg_cumul_reward = reward_send_queue.get(block=True)  # blocking wait
            data = pickle.dumps({"type": "avg_cumul_reward", "value": avg_cumul_reward["value"], "bucket_index": avg_cumul_reward["bucket_index"]})
            size = len(data).to_bytes(4, "little")
            conn.sendall(size + data)
            print(f"[REWARD_THREAD] Sent avg cumulative reward (bucket {avg_cumul_reward['bucket_index']}): {avg_cumul_reward['value']:.3f}")

        except Exception as e:
            print(f"[REWARD_THREAD] Error sending reward data: {e}")


def threaded_comm(conn):
    while True:
        try:
            experience = send_queue.get(block=True)  # blocks until available
            data = pickle.dumps(experience)
            size = len(data).to_bytes(4, "little")
            conn.sendall(size + data)
            sent_time = time.time()
            print(f"[WORKER_THREAD] Sent experience at {sent_time}")

            # wait for response from manager
            size_bytes = conn.recv(4)
            size = int.from_bytes(size_bytes, "little")
            data = b""
            while len(data) < size:
                data += conn.recv(size - len(data))

            action = pickle.loads(data)
            action_queue.put(action)
            action_time = time.time()
            print(f"[WORKER_THREAD] Obtained action at {action_time}")
            print(f"Cycle time: {action_time - sent_time:.4f} seconds")

        except Exception as e:
            print(f"[THREAD] Error in socket communication: {e}")
            action_queue.put(5)


def do_action(action):
    action_string = ACTION_KEYS[action]
    print(f"\t\t\tAction: {action_string}")
    button_map = {
        "sprint_right": {"Right": True, "B": True},
        "jump_right": {"Right": True, "A": True, "B": True},
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
            assert len(frames_batch) == 8, f"Expected 8 frames, got {len(frames_batch)}"
            state = preprocess_frames_cv2(frames_batch[:4])
            next_state = preprocess_frames_cv2(frames_batch[4:])
            processed_queue.put((state, action, reward, next_state))

        except Exception as e:
            print(f"[PREPROCESSOR] Error: {e}")


def preprocess_frames_cv2(frames, target_size=(140, 114)):
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

    return img_arrays


memory_queue = queue.Queue()
read_signal = threading.Event()
shutdown_signal = threading.Event()
def persistent_memory_reader():
    while not shutdown_signal.is_set():
        read_signal.wait()  # blocks until set
        read_signal.clear()  # reset signal for next use

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
ACTION_KEYS = list(agent_action({}).keys())
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_KEYS)}
NUM_ACTIONS = len(ACTION_KEYS)


class frameList:
    def __init__(self, length=4):
        self.frames = deque(maxlen=length)

    def __len__(self):
        return len(self.frames)

    def getFrames(self):
        return list(self.frames)

    def append(self, frame):
        self.frames.append(frame)


if __name__ == "__main__":
    print("------------------Starting worker script...--------------------")
    episode_rewards = []
    
    cumul_reward_current_episode = 0.0
    frame_batch_size = 500 # every 500 frames we will send an average cumulative reward to the manager 
    frame_count = 0

    last_logged_frame_count = 0

    checkpoints = generate_checkpoints(level1_start_x, level1_last_cp, num_checkpoints)
    default_action = "none"
    action = ACTION_TO_INDEX[default_action]
    print(f"[WORKER] Using default action: {default_action} (index {action})")
    data = None
    socket_conn = connect_to_socket()
    threading.Thread(target=threaded_comm, args=(socket_conn,), daemon=True).start()
    threading.Thread(target=threaded_preprocessor, daemon=True).start()
    threading.Thread(target=persistent_memory_reader, daemon=True).start()
    threading.Thread(target=reward_sender_thread, args=(socket_conn,), daemon=True).start()


    # move past initial frames, 10 is arbitrary, this is just so the memory is intialized so we can read from it
    for i in range(10):
        await event.framedrawn()

    prev_chunk = None
    frameStorage = frameList(length=4)

    while True:
        print(f"Start loop, frameCount: {frame_count}")

        
        # C-captured frame
        # S-skipped frame
        # C-S-C-S-S-C-S-C, so we render 7 frames but only use 4 frames, also the last captured frame is only drawn after setting memory reader
        mask = [1,0,1,0,0,1,0]
        for i in range(7):
            if mask[i]:
                frameStorage.append(await event.framedrawn())
            else:
                await event.framedrawn()
            do_action(action)
        
        # rendered 6 frames now, then we read memory with the semi-latest frame, so we are off sync by 1, but we have to since
        # we need time to read memory
        read_signal.set()

        # render last frame and append it to frameStorage
        frameStorage.append(await event.framedrawn())
        do_action(action)

        old_data = data
        for i in range(2):
            try:
                data = memory_queue.get_nowait()
                print("Data aquired")
                break
            except queue.Empty:
                print(f"\nWARNING: data was not available yet, retrying ({i+1}/2)...")
                time.sleep(0.003)

        reward, new_checkpoint_idx = reward, new_checkpoint_idx = compute_reward(
        data, previous_lives, previous_mario_form, previous_checkpoint_idx, checkpoints,
        previous_x, previous_clock
        )

        cumul_reward_current_episode += reward
    
        done = False

        if data['cur_x'] >= level1_last_cp:
            done = True
            await event.framedrawn()
            await event.framedrawn()
            await event.framedrawn()
            savestate.load_from_slot(3)  # reset to start of level
            await event.framedrawn()
            await event.framedrawn()
            await event.framedrawn()


        if previous_lives is not None and previous_lives > data['lives']:
            print("-------------------")
            print("Died, resetting state. Reward: ", reward)
            print("-------------------")
            done = True
            await event.framedrawn()
            await event.framedrawn()
            await event.framedrawn()
            savestate.load_from_slot(3) # might be excessive with the frame draw calls, but it prevents dolphin from freezing
            await event.framedrawn()
            await event.framedrawn()
            await event.framedrawn()

        if done:
            episode_rewards.append(cumul_reward_current_episode)
            cumul_reward_current_episode = 0.0

            if frame_count - last_logged_frame_count >= frame_batch_size:
                avg_cumul_reward = np.mean(episode_rewards)
                batch_start_frame = last_logged_frame_count
                bucket_index = batch_start_frame // frame_batch_size

                reward_send_queue.put({
                    "type": "avg_cumul_reward",
                    "value": avg_cumul_reward,
                    "bucket_index": bucket_index
                })
                print(f"[WORKER] Sent avg reward for frames (bucket {bucket_index}): {avg_cumul_reward:.3f}")
                episode_rewards.clear()
                last_logged_frame_count = frame_count


        current_frames = frameStorage.getFrames()
        if prev_chunk:
            prev_frames, prev_action, prev_reward = prev_chunk

            preprocess_queue.put((prev_frames+current_frames, prev_action, prev_reward))
            try:
                state, _, _, next_state = processed_queue.get(timeout=.1)
                experience = (state, prev_action, prev_reward, next_state, done)
                send_queue.put(experience)

            except queue.Empty:
                print("[WORKER] Skipped frame: no processed result")

        for i in range(2):
            try:
                action = action_queue.get_nowait()
                break
            except queue.Empty:
                print(f"[WORKER] No action received yet, retrying ({i+1}/3)...")
                time.sleep(0.003)
                if action is None:
                    print("\nWARNING: No action received, defaulting to 'none'\n")
                    action = ACTION_TO_INDEX[default_action]

        prev_chunk = (current_frames, action, reward)
        previous_lives = data['lives']
        previous_mario_form = data['mario_form']
        previous_checkpoint_idx = new_checkpoint_idx
        previous_time = data['current_time']
        frame_count += 7
        previous_x = data['cur_x']
        previous_clock = data['current_time']