import os
import subprocess
import threading
import torch
import pickle
import time
import socket
import random
import sys

import torch.nn.functional as F
import numpy as np

from collections import deque
from PIL import Image
from model import BTRNetwork


# I just ran it via 'python .\scripts\trainerManager.py', which matches the os.getcwd()
# if you wanna run it from another directory, just adjust or add to path

# General notes and TODOs:
# The target network is wasted? Since workers have their own model which gets synced every once in a while
# I think we might need to centralize actions in this manager script as well. So the workers send their state (images) and manager returns the action
# Though this might be too slow. Not sure what to do about it yet.
# - We need to add priority ER, instead of just sampling from the replay buffer
# - Since we have opencv in requirements, we should replace some PIL with opencv, since it is faster. This can speedup worker script. If we combine this with central model it might be fast enough
# - upon level complete we need to reload states, this would be done in worker script, but we dont have to think about this yet, since we dont get even clsoe
# - We need to add a way to relaunch workers, if they crash. This is not implemented yet, but we have a relaunch_worker function that should work, but we need to test it first
# - Check the training loop, and try to fix this behaviour, the model learns to stand still? Outputing "none"
# - We likely also need to add epsilon greedy exploration and reduce it over time, either in model.py or in getAction() of worker script
# - Another thing, the model doesnt seem to output "airborne", which it should sometimes do, noticed this during testing, but havent yet looked into it. ("airborne" should be the spin action)

sys.path.append(os.path.join(os.getcwd(), 'scripts'))

def project_root():
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # Stop at the root directory
        if all(os.path.exists(os.path.join(current_dir, file)) for file in ['requirements.txt', 'README.md']):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Project root with 'requirements.txt' and 'README.md' not found.")
root = project_root()

def receive_experience(worker_id):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 6000 + worker_id))
    server.listen(1)
    print(f"[MANAGER] Listening for worker {worker_id} on port {6000 + worker_id}")
    conn, addr = server.accept()
    print(f"[MANAGER] Worker {worker_id} connected.")

    while True:
        size_bytes = conn.recv(4)
        size = int.from_bytes(size_bytes, "little")
        data = b""
        while len(data) < size:
            data += conn.recv(size - len(data))

        #decompressed = zlib.decompress(data)
        #experiences = pickle.loads(decompressed)

        experiences = pickle.loads(data)

        print(f"[MANAGER] Got {len(experiences)} from worker {worker_id}")


        experience_buffers[worker_id].extend(experiences)

def broadcast_model_update(worker_ids):
    for worker_id in worker_ids:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("localhost", 7000 + worker_id))
            s.sendall(b"reload_model")
            s.close()
            print(f"[MANAGER] Signaled model_update to worker {worker_id}")
        except Exception as e:
            print(f"[MANAGER] Error signaling worker {worker_id}: {e}")

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def process_frame_group(self, frame_group):
        imgs = [Image.frombytes("RGBA", (w, h), data) for (w, h, data) in frame_group]
        imgs = [img.crop((pos_start_screen[0], pos_start_screen[1], pos_end_screen[0], pos_end_screen[1])) for img in imgs]
        imgs = [img.convert("L").resize((140, 114), Image.Resampling.LANCZOS) for img in imgs]
        return [np.array(img, dtype=np.float32) for img in imgs]

    def add(self, experience_list):
        for (state, action, reward, next_state) in experience_list:
            s = self.process_frame_group(state)
            ns = self.process_frame_group(next_state)
            self.buffer.append((s, action, reward, ns))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

pos_start_screen = (200, 100-20)
pos_end_screen = (860-50, 500-50)

def train():
    global step, replay_buffer, model, experience_buffers, target_model

    # TODO: these parameters need to be adjusted to match the papers
    batch_size = 96
    learning_rate = 0.0001
    num_epochs = 5
    target_network_update_freq = 1000
    save_name = 'latest_model.pth'


    while  True:
        all_unprocessed = sum(experience_buffers, [])
        replay_buffer.add(all_unprocessed)
        experience_buffers = [[] for _ in range(num_workers)]

        if len(replay_buffer.buffer) < batch_size:
            print(f"[MANAGER] Not enough data to sample, waiting...")
            time.sleep(5)
            continue
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        model.train()
        target_model.to(device)

        # TODO: do we need to adjust the training loop? now we do mini-batches, each 1 step
        for epoch in range(num_epochs):

            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states = zip(*batch)
            
            # converting to array for faster tensor conversion, otherwise we get a warning
            states_arr = np.array(states, dtype=np.float32)
            next_states_arr = np.array(next_states, dtype=np.float32)

            states = torch.from_numpy(states_arr).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            next_states = torch.from_numpy(next_states_arr).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

            #print(f"States shape: {states.shape} \n Actions shape: {actions.shape} \n Rewards shape: {rewards.shape} \n Next states shape: {next_states.shape}")

            states = states.view(batch_size, 4, 114, 140)
            next_states = next_states.view(batch_size, 4, 114, 140)


            # These values need to be adjusted to match the papers
            # ----
            alpha = .9
            tau = 0.03
            clip_min = -1.0
            # ----

            q_values = model(states)
            q_values_mean = q_values.mean(dim=1)
            q_taken = q_values_mean.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = target_model(next_states)
                next_q_mean = next_q.mean(dim=1)
                next_probs = F.softmax(next_q_mean / tau, dim=1)
                next_v = (next_q_mean * next_probs).sum(dim=1)

                current_probs = F.softmax(q_values_mean / tau, dim=1)
                log_policy = torch.log(current_probs + 1e-8)
                log_pi_a = log_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
                munchausen_term = alpha * torch.clamp(log_pi_a, min=clip_min)

                target_q = rewards + munchausen_term + 0.99 * next_v

            loss = loss_fn(q_taken, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1


            if step % target_network_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
                print(f"[MANAGER] Updated target model at step {step}")
        

        torch.save(model.state_dict(), save_name)
        print(f"[MANAGER] Saved model at step {step}")
        
        # send a message to workers to reload their model
        broadcast_model_update(range(num_workers))
        time.sleep(30) # here so we dont train too fast, TODO: this needs adjusting

        # ----------------
        # Uncomment code below to see some of the input data saved as images, TODO: do we want uint8 or float32 for model input?
        # ----------------
        # experience = replay_buffer.sample(1)[0]
        # state, action, reward, next_state = experience
        # for idx, img_np_array in enumerate(state):
        #     img_np_array = (img_np_array - img_np_array.min()) / (img_np_array.max() - img_np_array.min()) * 255
        #     img_np_array = img_np_array.astype(np.uint8)

        #     img = Image.fromarray(img_np_array, mode='L')
        #     print(f"Frame {idx} shape: {img_np_array.shape}, dtype: {img_np_array.dtype}")
        #     img.save(f"state_{step}_{idx}.png")
        #     #img.save(f"state_{step}_{idx}.png")

        # print(f"[MANAGER] Saved test images for visualization of input data")
        
# Here it is important to note that if you want multiple workers to run in parallel
# you need to uncomment "--user" and "--user_folder" in the subprocess.Popen call
# This will also make it so you cant see logs anymore of workers, they dont appear to update
# so if you think something is going on with a worker, and you want to check the logs
# you need to run a single worker and comment out the "--user" and "--user_folder" lines
def launch_workers():
    for i in range(num_workers):
        env = os.environ.copy()
        env["WORKER_ID"] = str(i)

        # Create per-worker user folders like DolphinUser_0, DolphinUser_1, etc.
        user_folder = os.path.abspath(f"DolphinUser_{i}")
        os.makedirs(user_folder, exist_ok=True)
        threading.Thread(target=receive_experience, args=(i,), daemon=True).start()

        subprocess.Popen([
            dolphin_path,
            game_path,
            "--script",
            online_trainer_path,
            "--save_state",
            game_save_dir,
            "--no-python-subinterpreters",
            # "--user",
            # user_folder
        ], env=env)


# untested, not yet implemented
def relaunch_worker(worker_id):
    env = os.environ.copy()
    env["WORKER_ID"] = str(worker_id)

    # Create per-worker user folders like DolphinUser_0, DolphinUser_1, etc.
    user_folder = os.path.abspath(f"DolphinUser_{worker_id}")
    os.makedirs(user_folder, exist_ok=True)

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
    print(f"[MANAGER] Relaunched worker {worker_id}")

dolphin_path = os.path.abspath(os.path.join(root, "dolphin", "Dolphin.exe"))
game_path = os.path.join(root, "NSMB.rvz")
game_save_dir = os.path.join(root, "StateSaves")
game_save_dir = os.path.join(game_save_dir, "SMNP01.s01")
scripts_dir = os.path.abspath(os.path.join(root, "scripts"))
online_trainer_path = os.path.join(scripts_dir, "online_trainer_latest.py")
print(f"[MANAGER] Using game path: {game_path}")
print(f"[MANAGER] Using game save dir: {game_save_dir}")
print(f"[MANAGER] Using online trainer path: {online_trainer_path}")
num_workers = 1     # multi-worker requires --user and --user_folder in launch_workers and disables logging

experience_buffers = [[] for _ in range(num_workers)]
model = BTRNetwork(num_actions=10)
target_model = BTRNetwork(num_actions=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MANAGER] Using device: {device}")
step = 0

# we need a base model that is common between worker and manager, so experiences are actually from the same model's
# output. A base model can be generated with "offline_trainer.ipynb" and then saved as "btr_model.pth"
# or you can just train a model with the manager and save it as "btr_model.pth"
base_model_path = os.path.join(root, "btr_model.pth")
if os.path.exists(base_model_path):
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    print(f"[MANAGER] Loaded model from {base_model_path}")


# This needs to be upgraded to priority ER
replay_buffer = ReplayBuffer(max_size=10000) # not sure if its too much

if __name__ == "__main__":
    launch_workers() # separate thread that launches and receives experiences from workers
    train()          # main control loop that takes experiences, adds to replay buffer and trains the model

    # There is a relaunch_worker, not sure yet if it works. Havent tried to set this up yet
