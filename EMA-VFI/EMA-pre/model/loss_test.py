import torch

class Multi_Scale_loss(torch.nn.Module):
    def __init__(self):
        super(Multi_Scale_loss, self).__init__()

    def forward(self, af_loss, mf_loss):
        loss = 0
        loss += torch.nn.functional.l1_loss(af_loss[0], af_loss[1])
        loss += torch.nn.functional.l1_loss(af_loss[1], af_loss[2])
        loss += torch.nn.functional.l1_loss(af_loss[2], af_loss[0])
        loss += torch.nn.functional.l1_loss(mf_loss[0], mf_loss[1])
        loss += torch.nn.functional.l1_loss(mf_loss[1], mf_loss[2])
        loss += torch.nn.functional.l1_loss(mf_loss[2], mf_loss[0])
        return float(loss/6)

a = []
b = []

a.append(torch.randn(2, 64, 32, 32))
a.append(torch.randn(2, 64, 32, 32))
a.append(torch.randn(2, 64, 32, 32))
b.append(torch.randn(2, 64, 32, 32))
b.append(torch.randn(2, 64, 32, 32))
b.append(torch.randn(2, 64, 32, 32))

c = Multi_Scale_loss()
output = c(a, b)
print(output)