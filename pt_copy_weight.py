import copy
import torch

def is_same(a, b):
    return ((a - b).float().abs().sum() < 1e-6).item()

a = torch.nn.Conv2d(5, 5, 3)
b = a.weight
c = copy.deepcopy(a.weight)

torch.nn.init.xavier_uniform_(a.weight.data)

print("a == b: " + str(is_same(a.weight, b)))
print("a == c: " + str(is_same(a.weight, c)))