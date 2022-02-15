# POT Testing
import numpy as np
import torch
import ot

print('*'*10 + 'NumPy' + '*'*10)

aa = np.array([0.00991215, 0.50955325, 0.4805346 ])
print(aa)
print('aa sum:', aa.sum())

bb = np.array([9.359e-06, 9.341e-06, 9.999813e-01])
print(bb)
print('bb sum:', bb.sum())


MM = np.array([[0,1,2],[1,0,1],[2,1,0]])

out = ot.emd2(aa, bb, MM, log=True)

print(out)

print('*'*10 + 'Torch' + '*'*10)

aat = torch.Tensor(aa)
print(aat)
print('aat sum:', aat.sum())

bbt = torch.Tensor(bb)
print(bbt)
print('bbt sum:', bbt.sum())


MMt = torch.Tensor(MM)

outt = ot.emd2(aat, bbt, MMt, log=True)

print(outt)