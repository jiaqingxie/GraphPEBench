import torch
import subprocess

command = "srun --mem=25GB --gres=gpu:1 --exclude=tikgpu[06-10] --pty bash -i"
result = subprocess.run(command, shell=True, text=True, capture_output=True)
print(torch.cuda.is_available())