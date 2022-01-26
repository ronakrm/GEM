import numpy as np

import ot

from demd.emd import greedy_primal_dual
from demd.emd_vanilla import demd_func, minimize, vectorize

from demd.sinkhorn_barycenters import barycenter
from demd.datagen import getData

np.random.seed(0)

n = 12  # nb bins
d = 3

# data, M = getData(n, d, 'uniform')
# data, M = getData(n, d, 'skewedGauss')

a1 = np.array([0.1, 0.2, 0.55, 0.15])
a2 = np.array([0.1, 0.25, 0.6, 0.05])
a3 = np.array([0.1, 0.2, 0.6, 0.1])
a4 = np.array([0.1, 0.2, 0.3, 0.4])
a5 = np.array([0.6, 0.2, 0.1, 0.1])
data = [a1, a2, a3, a4, a5]


d = len(data)
print(data)

log = greedy_primal_dual(data)
print('DEMD', log['primal objective'])

import pdb; pdb.set_trace()

# data = np.array(data)
# vecsize = n*d

# ot.tic()
# x = minimize(demd_func, data, d, n, vecsize,
#                  niters=10000, lr=1e-6)
# time = ot.toc('')

# print(x)
# print(sum(x[0]))
# print(sum(x[1]))
# print(sum(x[2]))
# print(time)

# import pdb; pdb.set_trace()


# device = 'cpu'
# imgs = torch.tensor(imgs_np, dtype=torch.float64, device=device,
#                     requires_grad=False)
# # dists = create_distribution_2d(imgs_np)
# imgs = imgs + 1e-10
# imgs /= imgs.sum((1, 2))[:, None, None]
# epsilon = 0.002

# grid = torch.arange(width).type(torch.float64)
# grid /= width
# M = (grid[:, None] - grid[None, :]) ** 2
# M_large = M[:, None, :, None] + M[None, :, None, :]
# M_large = M_large.reshape(n_features, n_features)
# M_large = M_large.to(device)

# K = torch.exp(- M / epsilon)
# K = K.to(device)
# # 
# # print("Doing IBP ...")
# # time_ibp = time.time()
# bar_ibp, log = barycenter(imgs, K, reference="uniform", return_log=True)
# # time_ibp = time.time() - time_ibp
# print('IBP', log['a'])
