import os
import subprocess
import threading
import torch
import pickle
import time
import socket
import random
import sys
import traceback
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import queue
from PIL import Image
from btr.Agent import Agent
torch.autograd.set_detect_anomaly(True)

# ---- setup ----
def project_root():
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # Stop at the root directory
        if all(os.path.exists(os.path.join(current_dir, file)) for file in ['requirements.txt', 'README.md']):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Project root with 'requirements.txt' and 'README.md' not found.")
root = project_root()



# ------ COMMUNICATION -----
conn_map = {}
def handle_worker_connection(worker_id):
    global agent

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 6000 + worker_id))
    server.listen(1)
    print(f"[MANAGER] Listening for worker {worker_id} on port {6000 + worker_id}")
    conn, addr = server.accept()
    conn_map[worker_id] = conn
    print(f"[MANAGER] Worker {worker_id} connected.")

    while True:
        try:
            # Receive experience
            size_bytes = conn.recv(4)
            size = int.from_bytes(size_bytes, "little")
            data = b""
            while len(data) < size:
                data += conn.recv(size - len(data))

            experience = pickle.loads(data)


            (state, action, reward, next_state) = experience

            # Get action from model
            next_state_batched = np.expand_dims(next_state, axis=0)
            action_tensor = agent.choose_action(next_state_batched)
            action_discrete = action_tensor.item()


            agent.store_transition(state, action, reward, next_state, False, 0, True)

            #print(f"[MANAGER] Received experience from worker {worker_id}")


            # Send back action
            response = pickle.dumps(action_discrete)
            response_size = len(response).to_bytes(4, "little")
            conn.sendall(response_size + response)
            print(f"[MANAGER] Sent action {ACTION_KEYS[action_discrete]} to worker {worker_id}")



        except Exception as e:
            print(f"[MANAGER] Error with worker {worker_id}: {e}")
            traceback.print_exc()
            break


# ----- CODE FOR MODEL AND TRAINING -----
ACTION_KEYS = [
                "sprint_right",
                "jump_right",
                "jump_left",
                "move_right",
                "move_left",
                "none"]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_KEYS)}
NUM_ACTIONS = len(ACTION_KEYS)


num_workers = 1   # multi-worker requires --user and --user_folder in launch_workers and disables logging unless you copy user_dir into worker dir
experience_buffers = [[] for _ in range(num_workers)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MANAGER] Using device: {device}")
agent = Agent(
    n_actions=NUM_ACTIONS,
    input_dims=(4, 140, 114), # 4 x width x height
    device=device,
    num_envs=1, # havent tried this with 2, so keep this at 1, even when using multiple workers
    agent_name="online_agent",
    total_frames=300_000, # set too high to avoid too fast epsilon decay
    max_mem_size=10_000, # maybe too high
    imagex=114, # IMPORTANT: This is switched because their memory expects height x width, while the rest is width x height
    imagey=140,
    #eps_steps=10_000,
    testing=False
)




def launch_workers():
    try:
        for i in range(num_workers):
            env = os.environ.copy()
            env["WORKER_ID"] = str(i)

            # Create per-worker user folders like DolphinUser_0, DolphinUser_1, etc.
            user_folder = os.path.abspath(f"DolphinUser_{i}")
            os.makedirs(user_folder, exist_ok=True)
            threading.Thread(target=handle_worker_connection, args=(i,), daemon=True).start()

            # Print debug information
            print(f"[MANAGER] Launching Dolphin with the following parameters:")
            print(f"[MANAGER] Dolphin path: {dolphin_path}")
            print(f"[MANAGER] Game path: {game_path}")
            print(f"[MANAGER] Script path: {online_trainer_path}")
            print(f"[MANAGER] Save state path: {game_save_dir}")
            print(f"[MANAGER] User folder: {user_folder}")

            # Launch Dolphin with the worker script
            subprocess.Popen([
                dolphin_path,
                game_path,
                "--script",
                online_trainer_path,
                "--save_state",
                game_save_dir,
                "--no-python-subinterpreters",
                "--user",
                user_folder
            ], env=env)

            print(f"[MANAGER] Dolphin launched with worker ID {i}")
    except Exception as e:
        print(f"[MANAGER] Error launching workers: {e}")
        traceback.print_exc()
        sys.exit(1)

import matplotlib.pyplot as plt

def load_model():
    global agent
    model_files = [
        os.path.join(model_path, f)
        for f in os.listdir(model_path)
        if f.endswith('.model')
    ]
    if model_files:
        print(f"[MANAGER] Found {model_files} model files.")
        latest_model = min(model_files, key=os.path.getctime)
        print(f"[MANAGER] Loading latest model: {latest_model}")
        agent.load_models(latest_model)
    else:
        print("[MANAGER] No model found. Starting training from scratch.")

def train():
    global agent
    save_every = 5

    while True:
        time.sleep(10)

        if agent.memory.size > 0:
            _, states, _, _, next_states, _, _ = agent.memory.sample(3)

            for idx, state in enumerate(states):
                for frame_idx in range(state.shape[0]):
                    frame = state[frame_idx].cpu().numpy()  # shape: (114, 140)
                    plt.imsave(f"manager_state_{idx}_frame_{frame_idx}.png", frame, cmap='gray')

        print(f"[MANAGER] Training agent...")
        try:
            agent.learn()
        except Exception as e:
            print(f"[MANAGER] Error during training: {e}")
            traceback.print_exc()
            continue

        if agent.grad_steps % save_every == 0:
            print(f"[MANAGER] Saving model at step {agent.grad_steps}...")
            agent.save_model()
            print(f"[MANAGER] Model saved.")

        print(f"Gradient steps: {agent.grad_steps}")
        print(f"[MANAGER] Training complete.")



dolphin_path = os.path.abspath(os.path.join(root, "../", "Assignment 3", "Experiment", "dolphin", "Dolphin.exe"))
game_path = os.path.join(root, "../", "Assignment 3", "Experiment", "dolphin", "rom", "New Super Mario Bros. Wii (Europe) (En,Fr,De,Es,It) (Rev 1).rvz")
game_save_dir = os.path.join(root, "StateSaves")
game_save_dir = os.path.join(game_save_dir, "SMNP01.s03")
scripts_dir = os.path.abspath(os.path.join(root, "scripts"))
online_trainer_path = os.path.join(scripts_dir, "worker.py")
model_path = os.path.join(root, "scripts")
print(f"[MANAGER] Using game path: {game_path}")
print(f"[MANAGER] Using game save dir: {game_save_dir}")
print(f"[MANAGER] Using online trainer path: {online_trainer_path}")

if __name__ == "__main__":
    load_model()
    launch_workers()
    train()