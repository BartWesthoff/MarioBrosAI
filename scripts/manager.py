import os
import sys
# ---- setup ----
def project_root():
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # Stop at the root directory
        if all(os.path.exists(os.path.join(current_dir, file)) for file in ['requirements.txt', 'README.md']):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Project root with 'requirements.txt' and 'README.md' not found.")
root = project_root()

import subprocess
import threading
import queue
import torch
import pickle
import time
import socket

import traceback
import numpy as np
from threading import Lock
from btr.Agent import Agent
sys.path.append(os.path.join(root, "scripts"))

from utils_func import agent_action


from datetime import datetime, timedelta

worker_last_seen = {}
worker_processes = {}



memory_lock = Lock()
log_queue = queue.Queue()




def logging_thread():
    # creat logs directory if it does not exist
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[MANAGER] Logging to {os.path.join(log_dir, 'manager.log')}")
    with open(os.path.join(root, "logs", "manager.log"), "a") as log_file:
        while True:
            try:
                msg = log_queue.get()
                if msg is None:
                    break
                log_file.write(msg + "\n")
                log_file.flush()
            except Exception as e:
                print(f"[MANAGER] Logging thread error: {e}")
                traceback.print_exc()
                break            


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
            size_bytes = conn.recv(4)
            size = int.from_bytes(size_bytes, "little")
            data = b""
            while len(data) < size:
                data += conn.recv(size - len(data))


            experience = pickle.loads(data)
            worker_last_seen[worker_id] = datetime.now()


            # we are collecting average cumulative reward and experiences over same socket, so we check
            # in case it is actually a reward message instead of an experience
            if isinstance(experience, dict) and experience.get("type") == "avg_cumul_reward":
                avg_reward = experience.get("value")
                bucket_index = experience.get("bucket_index")
                log_queue.put(f"{bucket_index}, {avg_reward}, {worker_id}")
                print(f"[MANAGER] Received avg cumulative reward {avg_reward} for bucket {bucket_index} from worker {worker_id}")
                continue



            (state, action, reward, next_state, done) = experience

            next_state_batched = np.expand_dims(next_state, axis=0)
            action_tensor = agent.choose_action(next_state_batched)
            action_discrete = action_tensor.item()

            with memory_lock:
                agent.store_transition(state, action, reward, next_state, done, worker_id, True)

            # send back action
            response = pickle.dumps(action_discrete)
            response_size = len(response).to_bytes(4, "little")
            conn.sendall(response_size + response)
            print(f"[MANAGER] Sent action {ACTION_KEYS[action_discrete]} to worker {worker_id}")

        except Exception as e:
            print(f"[MANAGER] Error with worker {worker_id}: {e}")
            traceback.print_exc()
            break


def monitor_workers(timeout_seconds=60):
    while True:
        time.sleep(5)
        now = datetime.now()
        for worker_id, last_seen in list(worker_last_seen.items()):
            if now - last_seen > timedelta(seconds=timeout_seconds):
                print(f"[MANAGER] Worker {worker_id} has timed out. Restarting...")
                restart_worker(worker_id)

def restart_worker(worker_id):
    try:
        # kill the process if it is still running
        proc = worker_processes.get(worker_id)
        if proc and proc.poll() is None:
            proc.kill()
            print(f"[MANAGER] Killed unresponsive worker {worker_id}")

        # reset last seen
        worker_last_seen[worker_id] = datetime.now()

        # close old connection if it exists
        old_conn = conn_map.get(worker_id)
        if old_conn:
            try:
                old_conn.close()
            except:
                pass

        # relaunch socket
        threading.Thread(target=handle_worker_connection, args=(worker_id,), daemon=True).start()

        # restart worker
        env = os.environ.copy()
        env["WORKER_ID"] = str(worker_id)
        user_folder = os.path.abspath(f"DolphinUser_{worker_id}")
        os.makedirs(user_folder, exist_ok=True)

        proc = subprocess.Popen([
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

        worker_processes[worker_id] = proc
        print(f"[MANAGER] Relaunched Dolphin for worker {worker_id}")
    except Exception as e:
        print(f"[MANAGER] Failed to restart worker {worker_id}: {e}")
        traceback.print_exc()



# ----- CODE FOR MODEL AND TRAINING -----
ACTION_KEYS = list(agent_action({}).keys())
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTION_KEYS)}
NUM_ACTIONS = len(ACTION_KEYS)


num_workers = 2  # multi-worker requires --user and --user_folder in launch_workers and disables logging unless you copy user_dir into worker dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MANAGER] Using device: {device}")
agent = Agent(
    n_actions=NUM_ACTIONS,
    input_dims=(4, 140, 114), # 4 x width x height
    device=device,
    num_envs=2,
    agent_name="online_agent",
    total_frames=10_000_000, # set too high to avoid too fast epsilon decay
    max_mem_size=20_000, # maybe too high
    imagex=114, # IMPORTANT: This is switched because their memory expects height x width, while the rest is width x height
    imagey=140,
    target_replace=100,
    batch_size=128, # lowered batch size to 128 to compensate for higher replay ratio than before
    eps_steps=300_000,
    testing=False,
    n=4, # frame step size, essentially (framestack - n) = overlap, per default its 3, so (4-3) = 1 overlap but we use (4-4) = 0 overlap
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
            PID = subprocess.Popen([
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
            worker_processes[i] = PID

            worker_last_seen[i] = datetime.now()

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
        latest_model = max(model_files, key=os.path.getctime)
        print(f"[MANAGER] Loading latest model: {latest_model}")
        agent.load_models(latest_model)
    else:
        print("[MANAGER] No model found. Starting training from scratch.")

    #  not sure if this is needed again after loading in weights, but just in case
    agent.net.train()
    agent.tgt_net.train()



def train():
    global agent
    save_every = 5
    replay_ratio = 1/64

    # create debug directory to save images
    debug_dir = os.path.join(root, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    print(f"[MANAGER] Debug images will be saved to {debug_dir}")

    while True:
        time.sleep(1)
        # with memory_lock:
        #     if agent.env_steps > 5:
        #         idxs, states, actions, rewards, next_states, dones, weights = agent.memory.sample(1)

        #         #print(f"[MANAGER] Sampled experience: {idxs}, {states.shape}, {actions}, {rewards}, {next_states.shape}, {dones}")

        #         for i in range(states.shape[0]):
        #             state = states[i]
        #             action = actions[i]
        #             reward = rewards[i]
        #             next_state = next_states[i]
        #             done = dones[i]

        #             # Save state's frames to .png for visual debugging
        #             for frame_idx in range(state.shape[0]):
        #                frame = state[frame_idx].cpu().numpy()
        #                plt.imsave(os.path.join(debug_dir, f"Y_manager_sampled_{frame_idx}.png"), frame, cmap='gray')
        #             for frame_idx in range(next_state.shape[0]):
        #                frame = next_state[frame_idx].cpu().numpy()
        #                plt.imsave(os.path.join(debug_dir, f"Z_manager_next_sampled_{frame_idx}.png"), frame, cmap='gray')

        
        try:
            if agent.grad_steps < (agent.env_steps / agent.num_envs) * replay_ratio:
                print(f"[MANAGER] Training agent...")
                agent.learn()
                print(f"[MANAGER] Training complete.")
                print(f"Gradient steps: {agent.grad_steps}")
        except Exception as e:
            print(f"[MANAGER] Error during training: {e}")
            traceback.print_exc()
            continue

        if agent.grad_steps % save_every == 0 and agent.grad_steps > 0:
            print(f"[MANAGER] Saving model at step {agent.grad_steps}...")
            agent.save_model()
            print(f"[MANAGER] Model saved.")




dolphin_path = os.path.abspath(os.path.join(root, "dolphin", "Dolphin.exe"))
game_path = os.path.join(root, "NSMB.rvz")
game_save_dir = os.path.join(root, "StateSaves")
game_save_dir = os.path.join(game_save_dir, "SMNP01.s03")
scripts_dir = os.path.abspath(os.path.join(root, "scripts"))
online_trainer_path = os.path.join(scripts_dir, "worker.py")
model_path = os.path.join(root, "scripts")
print(f"[MANAGER] Using game path: {game_path}")
print(f"[MANAGER] Using game save dir: {game_save_dir}")
print(f"[MANAGER] Using online trainer path: {online_trainer_path}")

logging_thread = threading.Thread(target=logging_thread, args=(), daemon=True)
logging_thread.start()

if __name__ == "__main__":
    load_model()
    launch_workers()
    threading.Thread(target=monitor_workers, daemon=True).start()
    train()