import torch

x = torch.rand(5, 3)

print(x)

print('-' * 10)

print(x[1:, 2])
# is actually x[row, column]
print(x[1:, 2:3])
print(x[1:, 2:3][0:, 0])

print('-' * 10)

print(x[0, :2])
# is actually x[row, column]
print(x[0:1, 0:2])
print(x[0:1, 0:2][0])
print(x[0:1, 0:2][0, 0:])
