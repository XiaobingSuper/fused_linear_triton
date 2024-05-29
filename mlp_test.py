from nerfstudio.field_components import  mlp

import torch
import time
from xformers.triton import FusedLinear
from xformers.components import Activation
import triton_ops


# self.geo_feat_dim 15 self.encoder_dir.n_output_dims 16 num_layers_reflectance 3 hidden_dim 64
# intensity_net input_mat torch.Size([1048576, 31])
# ray_drop_net input_mat torch.Size([1048576, 31])
geo_feat_dim = 15
n_output_dims = 16
num_layers_reflectance = 3
hidden_dim = 64


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.linear_0 = torch.nn.Linear(in_features=31, out_features=64, bias=True)
        # self.linear_1 = torch.nn.Linear(in_features=64, out_features=64, bias=True)
        # self.linear_2 = torch.nn.Linear(in_features=64, out_features=2, bias=True)
        self.linear_0 = triton_ops.FusedLinear(31, 64, bias=True, activation=triton_ops.Activation.ReLU)
        self.linear_1 = triton_ops.FusedLinear(64, 64, bias=True, activation=triton_ops.Activation.ReLU)
        self.linear_2 = triton_ops.FusedLinear(64, 2, bias=True, activation=None)

    def forward(self, x):
        x = self.linear_0(x)
        # x = torch.relu(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        
        return x

model = MLP().cuda().train()
# intensity network
intensity_net = mlp.MLP(in_dim=geo_feat_dim + n_output_dims, num_layers=num_layers_reflectance, 
                                layer_width=hidden_dim, out_dim=2,  implementation = "tcnn")
# ray drop network
ray_drop_net = mlp.MLP(in_dim=geo_feat_dim + n_output_dims, num_layers=num_layers_reflectance, 
                                layer_width=hidden_dim, out_dim=1, implementation = "tcnn")


intensity_net_ref = mlp.MLP(in_dim=geo_feat_dim + n_output_dims, num_layers=num_layers_reflectance, 
                                layer_width=hidden_dim, out_dim=1,  implementation = "torch")

# copy intensity_net's weights to intensity_net_ref




input_mat = torch.rand(1048576, 31).cuda().requires_grad_()

input_mat_triton = torch.rand(1048576, 31).cuda().requires_grad_()
# case 2: 32768, 31
# input_mat = torch.rand(32768, 31).cuda()

grad = torch.rand(1048576, 2).half().cuda()

grad_bfloat16 = torch.rand(1048576, 2).bfloat16().cuda()

intensity_net_ref = intensity_net_ref.cuda().train()
intensity_net = intensity_net.cuda().train()
print(intensity_net)
for i in range(10):
    intensity = intensity_net(input_mat)
    intensity.backward(grad)
    
torch.cuda.synchronize()
t0 = time.time()

for i in range(100):
    intensity = intensity_net(input_mat)
    intensity.backward(grad)
    
torch.cuda.synchronize()
t1 = time.time()
print(intensity.dtype)
print(f"tcnn: {t1-t0}")


for i in range(10):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        intensity = model(input_mat_triton)
    intensity.backward(grad_bfloat16)

torch.cuda.synchronize()
t0 = time.time()

for i in range(100):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        intensity = model(input_mat_triton)
    intensity.backward(grad_bfloat16)
torch.cuda.synchronize()
t1 = time.time()
print(intensity.dtype)
print
print(f"torch: {t1-t0}")

# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                             aten::addmm         0.99%     441.000us         1.30%     576.000us      57.600us       9.510ms        27.64%       9.716ms     971.600us            10  
#                               _module_function_backward         0.41%     184.000us         0.67%     297.000us     148.500us       8.546ms        24.83%       8.763ms       4.381ms             2  
#                                   ampere_sgemm_64x64_tn         0.00%       0.000us         0.00%       0.000us       0.000us       7.539ms        21.91%       7.539ms       3.769ms             2  
# void tcnn::kernel_grid_backward<float, float, 3u, 2u...         0.00%       0.000us         0.00%       0.000us       0.000us       6.805ms        19.78%       6.805ms       6.805ms             1  
#                                                aten::mm         0.86%     382.000us         1.31%     583.000us      29.150us       4.474ms        13.00%       4.474ms     223.700us            20  
#                                        cudaLaunchKernel         6.73%       2.989ms         6.73%       2.989ms       5.189us       2.805ms         8.15%       2.805ms       4.870us           576  
#                                aten::threshold_backward         0.16%      70.000us         0.24%     108.000us      15.429us       2.012ms         5.85%       2.012ms     287.429us             7  
# void tcnn::transpose_gradients<float>(unsigned int, ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.711ms         4.97%       1.711ms       1.711ms             1  
#                                               aten::sum         1.40%     620.000us         2.41%       1.068ms      26.049us       1.597ms         4.64%       1.685ms      41.098us            41  
#                                         aten::clamp_min         0.17%      74.000us         0.26%     115.000us      16.429us       1.422ms         4.13%       1.471ms     210.143us             7  
#                                             aten::copy_         1.24%     549.000us        24.36%      10.813ms      81.301us       1.266ms         3.68%       1.722ms      12.947us           133  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.126ms         3.27%       1.126ms     225.200us             5  
#                                        _module_function         0.32%     141.000us         0.41%     181.000us      90.500us       1.039ms         3.02%       2.083ms       1.042ms             2  
# void at::native::reduce_kernel<128, 4, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     897.000us         2.61%     897.000us     149.500us             6  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     886.000us         2.57%     886.000us     443.000us             2  
#                                  ampere_sgemm_32x128_tn         0.00%       0.000us         0.00%       0.000us       0.000us     826.000us         2.40%     826.000us     826.000us             1  
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     813.000us         2.36%     813.000us     162.600us             5  
#                         ampere_sgemm_64x32_sliced1x4_nt         0.00%       0.000us         0.00%       0.000us       0.000us     772.000us         2.24%     772.000us     386.000us             2  
# void tcnn::kernel_grid<float, 3u, 2u, (tcnn::HashTyp...         0.00%       0.000us         0.00%       0.000us       0.000us     770.000us         2.24%     770.000us     770.000us             1  
#                                   ampere_sgemm_64x64_nn         0.00%       0.000us         0.00%       0.000us       0.000us     661.000us         1.92%     661.000us     330.500us             2  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
