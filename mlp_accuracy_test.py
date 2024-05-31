import torch

import triton_ops

torch.manual_seed(2022)

class MLP_ref(torch.nn.Module):
    def __init__(self):
        super(MLP_ref, self).__init__()
        # self.linear_0 = torch.nn.Linear(in_features=31, out_features=64, bias=True)
        # self.linear_1 = torch.nn.Linear(in_features=64, out_features=64, bias=True)
        # self.linear_2 = torch.nn.Linear(in_features=64, out_features=2, bias=True)

        self.linear_0 = torch.nn.Linear(in_features=64, out_features=64, bias=True)

    def forward(self, x):
        x = self.linear_0(x)
        #x = torch.relu(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.linear_0 = triton_ops.FusedLinear(31, 64, bias=True, activation=triton_ops.Activation.ReLU)
        self.linear_0 = triton_ops.FusedLinear(64, 64, bias=True)


    def forward(self, x):
        x = self.linear_0(x)
        return x
        return x


mlp_ref = MLP_ref().cuda()

mlp = MLP().cuda()

mlp_ref.linear_0.weight.data = mlp.linear_0.weight.data
if mlp_ref.linear_0.bias is not None:
    mlp_ref.linear_0.bias.data = mlp.linear_0.bias.data

 
mlp_ref = mlp_ref.train()
mlp.train()

input_mat_ref = torch.rand(64, 64).cuda()
input_mat = input_mat_ref.clone().detach().requires_grad_()

input_mat_ref.requires_grad_()

# case 2: 32768, 31
# input_mat = torch.rand(32768, 31).cuda()

grad = torch.rand(64, 64).cuda()

# with torch.autocast(device_type='cuda', dtype=torch.float16):
#     y_ref = mlp_ref(input_mat_ref)

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    grad = grad.bfloat16()
    y = mlp(input_mat)
    y_ref = mlp_ref(input_mat_ref)

print("y_ref", y_ref.dtype)
print("y_triton", y.dtype)

y_ref.backward(grad)
y.backward(grad)


print("compare forward output.................")
print(torch.allclose(y_ref, y))
print((y_ref-y).abs().max())
print("compare backward weight grad.................")
print(torch.allclose(mlp_ref.linear_0.weight.grad, mlp.linear_0.weight.grad))

if mlp_ref.linear_0.bias is not None:
    print("compare backward bias grad.................")
    print(torch.allclose(mlp_ref.linear_0.bias.grad, mlp.linear_0.bias.grad))
    print(torch.max((mlp_ref.linear_0.bias.grad-mlp.linear_0.bias.grad).abs()))

print("compare bacward input grad.................")
print(torch.allclose(input_mat_ref.grad, input_mat.grad))
print(torch.max((input_mat_ref.grad-input_mat.grad).abs()))
