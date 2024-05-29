import torch

import triton_ops


from xformers.triton import FusedLinear
from xformers.components import Activation

torch.manual_seed(2022)

class MLP_ref(torch.nn.Module):
    def __init__(self):
        super(MLP_ref, self).__init__()
        # self.linear_0 = torch.nn.Linear(in_features=31, out_features=64, bias=True)
        # self.linear_1 = torch.nn.Linear(in_features=64, out_features=64, bias=True)
        # self.linear_2 = torch.nn.Linear(in_features=64, out_features=2, bias=True)
        # self.linear_0 = FusedLinear(31, 64, activation=Activation.ReLU)
        self.linear_0 = torch.nn.Linear(in_features=64, out_features=64, bias=False)
        # self.linear_1 = FusedLinear(64, 64, activation=Activation.ReLU)
        # self.linear_2 = FusedLinear(64, 2, activation=None)

    def forward(self, x):
        x = self.linear_0(x)
        x = torch.relu(x)
        return x
        # x = torch.relu(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x
        


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.linear_0 = torch.nn.Linear(in_features=31, out_features=64, bias=True)
        # self.linear_1 = torch.nn.Linear(in_features=64, out_features=64, bias=True)
        # self.linear_2 = torch.nn.Linear(in_features=64, out_features=2, bias=True)
        # self.linear_0 = triton_ops.FusedLinear(31, 64, bias=True, activation=triton_ops.Activation.ReLU)
        # self.linear_0 = FusedLinear(31, 64, bias=True, activation=Activation.ReLU)
        self.linear_0 = triton_ops.FusedLinear(64, 64, bias=False, activation=triton_ops.Activation.ReLU)
        # self.linear_1 = triton_ops.FusedLinear(64, 64, activation=triton_ops.Activation.ReLU)
        # self.linear_2 = triton_ops.FusedLinear(64, 2, activation=None)

    def forward(self, x):
        x = self.linear_0(x)
        return x
        # x = torch.relu(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


mlp_ref = MLP_ref().cuda()

mlp = MLP().cuda()

mlp_ref.linear_0.weight.data = mlp.linear_0.weight.data
if mlp_ref.linear_0.bias is not None:
    # mlp.linear_0.bias.data.zero_()

    mlp_ref.linear_0.bias.data = mlp.linear_0.bias.data

# mlp_ref.linear_1.weight.data = mlp.linear_1.weight.data
# # mlp_ref.linear_1.bias.data = mlp.linear_1.bias.data

# mlp_ref.linear_2.weight.data = mlp.linear_2.weight.data
# # mlp_ref.linear_2.bias.data = mlp.linear_2.bias.data

 
mlp_ref = mlp_ref.train()

mlp.train()

#input_mat_ref = torch.rand(1048576, 31).cuda()
input_mat_ref = torch.rand(104, 64).cuda()
input_mat = input_mat_ref.clone().detach().requires_grad_()

input_mat_ref.requires_grad_()

# case 2: 32768, 31
# input_mat = torch.rand(32768, 31).cuda()

grad = torch.rand(104, 64).cuda()
#.bfloat16().cuda()

# with torch.autocast(device_type='cuda', dtype=torch.float16):
#     y_ref = mlp_ref(input_mat_ref)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    grad = grad.bfloat16()
    y = mlp(input_mat)
    y_ref = mlp_ref(input_mat_ref)
# y = mlp(input_mat)
# y_ref = mlp_ref(input_mat_ref)
#y = mlp(input_mat)

print("y_ref", y_ref.dtype)
print("y", y.dtype)

y_ref.backward(grad)

y.backward(grad)
import pdb
pdb.set_trace()
# pdb.set_trace()
print(torch.allclose(y_ref, y))
print((y_ref-y).abs().max())
print(torch.allclose(mlp_ref.linear_0.weight.grad, mlp.linear_0.weight.grad))

if mlp_ref.linear_0.bias is not None:
    print(torch.allclose(mlp_ref.linear_0.bias.grad, mlp.linear_0.bias.grad))
    print(torch.max((mlp_ref.linear_0.bias.grad-mlp.linear_0.bias.grad).abs()))

print(torch.allclose(input_mat_ref.grad, input_mat.grad))
print(torch.max((input_mat_ref.grad-input_mat.grad).abs()))

