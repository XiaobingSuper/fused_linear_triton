1. Prerequisites(optional if triton has been installed)
   ```
   pip install triton==2.1.0
   ```
2. install this package:
   ```
   pip install -e .
   ```
2. using fused Linear:
   ```
   import torch

   import triton_ops
   
   class MLP(torch.nn.Module):
       def __init__(self):
           super(MLP, self).__init__()
           self.linear = triton_ops.FusedLinear(31, 64, bias=False, activation=triton_ops.Activation.ReLU)
      
       def forward(self, x):
           x = self.linear(x)
           return x
    
   mlp = MLP().cuda()
   input_mat = torch.rand(104, 31).cuda()
    
   y = mlp(input_mat)
   ```
