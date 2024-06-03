
import torch
import time
import triton_ops

# self.geo_feat_dim 15 self.encoder_dir.n_output_dims 16 num_layers_reflectance 3 hidden_dim 64
# intensity_net input_mat torch.Size([1048576, 31])
# ray_drop_net input_mat torch.Size([1048576, 31])
geo_feat_dim = 15
n_output_dims = 16
num_layers_reflectance = 3
hidden_dim = 64


class MLP_Ref(torch.nn.Module):
    def __init__(self):
        super(MLP_Ref, self).__init__()
        self.linear_0 = torch.nn.Linear(in_features=31, out_features=64, bias=True)
        self.linear_1 = torch.nn.Linear(in_features=64, out_features=64, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=64, out_features=2, bias=True)

    def forward(self, x):
        x = self.linear_0(x)
        x = x.relu()
        x = self.linear_1(x)
        x = x.relu()
        x = self.linear_2(x) 
        return x
    

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_0 = triton_ops.FusedLinear(31, 64, bias=True, activation=triton_ops.Activation.ReLU)
        self.linear_1 = triton_ops.FusedLinear(64, 64, bias=True, activation=triton_ops.Activation.ReLU)
        self.linear_2 = triton_ops.FusedLinear(64, 2, bias=True, activation=triton_ops.Activation.ReLU)

    def forward(self, x):
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.linear_2(x) 
        return x

triton_model = MLP().cuda().train()

torch_model = MLP_Ref().cuda().train()

input_mat_triton = torch.rand(1048576, 31).cuda().requires_grad_()
#.bfloat16()
input_mat_torch = input_mat_triton.detach().clone().requires_grad_()
#.bfloat16()
# case 2: 32768, 31
#input_mat_triton = torch.rand(32768, 64).cuda()

grad_bfloat16 = torch.rand(1048576, 2).bfloat16().cuda()

for i in range(100):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = torch_model(input_mat_torch)
    y.backward(grad_bfloat16)

torch.cuda.synchronize()
t0 = time.time()
for i in range(100):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = torch_model(input_mat_torch)
    y.backward(grad_bfloat16)

torch.cuda.synchronize()
t1 = time.time()
print(f"torch mlp time: {(t1-t0)/100}")


for i in range(100):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = triton_model(input_mat_triton)
    y.backward(grad_bfloat16)

torch.cuda.synchronize()
t0 = time.time()
for i in range(100):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = triton_model(input_mat_triton)
    y.backward(grad_bfloat16)
torch.cuda.synchronize()
t1 = time.time()
print(f"triton mlp time: {(t1-t0)/100}")
