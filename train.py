import math
import random
import torch
import torch.cuda
from torch._C import device
from torch.nn import parameter
import torch.optim as optim
import torch.nn as nn
from torch.cuda import memory
from utils import Transition, plot_durations, ReplayMemory
from agent_game_control import Control, GameState 
from game_env import GameEnv,show_image
from model import DQN
from itertools import count
import pickle
import gc


batch_size = 32
gama_discount = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(3).to(device)
target_net = DQN(3).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-4)

memory = ReplayMemory(10000)
game_evn = GameEnv()
agent = Control(game_evn)
game_state = GameState(agent)
checkpoint_name = "checkpoint.pth"

episode_durations = []

try:
	checkpoint = torch.load(checkpoint_name)
	policy_net.load_state_dict(checkpoint['policy_net'])
	target_net.load_state_dict(checkpoint['target_net'])
	optimizer.load_state_dict(checkpoint["optimizer"])
except:
	print("NO SAVED MODEL FOUND..........!!!!")

def optimize_model():
	if len(memory) < batch_size:
		return
	transitions = memory.sample(batch_size)
	batch = Transition(*zip(*transitions))
	non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	state_action_values = policy_net(state_batch.float()).gather(1, action_batch)

	next_state_values = torch.zeros(batch_size, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states.float()).max(1)[0].detach()

	expected_state_action_values = (next_state_values * gama_discount) + reward_batch

	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1,1)
	optimizer.step()

def train():
	global memory
	try:
		memory, ct, steps = pickle.load(open("cache.p", "rb"))
	except:
		print("Starting from scratch........!!!!!!")
		memory = ReplayMemory(10000)
		steps = 0
		ct = 0
	game_evn.jump()
	try:
		while True:
			score = 0
			current_screen = game_evn.capture_screen()/255
			current_screen_torch = torch.from_numpy(current_screen).unsqueeze(0).unsqueeze(0)
			state = current_screen_torch
			for t in count():
				sample = random.random()
				threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps / eps_decay)
				steps += 1
				if sample > threshold:
					with torch.no_grad():
						action =  policy_net(state.float()).max(1)[1].view(1, 1)
				else:
					action = torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
				
				current_screen, reward, is_gameover, score = game_state.get_state(action.item())
				reward = torch.tensor([reward], device=device)
				score += reward
				current_screen = game_evn.capture_screen()/255
				current_screen_torch = torch.from_numpy(current_screen).unsqueeze(0).unsqueeze(0)
				if not is_gameover:
					next_state = current_screen_torch
				else:
					next_state = None
				memory.push(state, action, next_state, reward)
				state = next_state
				optimize_model()
				if is_gameover:
					episode_durations.append(t+1)
					plot_durations(episode_durations)
					break
			if ct % 100 == 0:
				game_evn.pause_game()
				with open("cache.p", "wb") as cache:
					pickle.dump((memory, ct, steps), cache)
				target_net.load_state_dict(policy_net.state_dict())
				gc.collect()
				torch.save({"policy_net": policy_net.state_dict(), "target_net" : target_net.state_dict(), "optimizer" : optimizer.state_dict()}, checkpoint_name)
				game_evn.resume_game()
				print(f"{ct} running.....")

			ct += 1
	except KeyboardInterrupt:
		torch.save({"policy_net": policy_net.state_dict(), "target_net" : target_net.state_dict(), "optimizer" : optimizer.state_dict()}, checkpoint_name)
train()
print("complete")