# 2D test
import numpy as np
import ot
import torch
import matplotlib.pyplot as plt

from demd.emd_vanilla import vectorize

from demd.img_emd import greedy_primal_dual, img_demd_func, img_minimize

from demd.sinkhorn_barycenters import barycenter



def make_ellipse(width=100, mean=None, semimaj=0.3,
                 semimin=0.1, phi=np.pi / 3):
    """
    Generate ellipse.
    The function creates a 2D ellipse in polar coordinates then transforms
    to cartesian coordinates.

    semi_maj : float
        length of semimajor axis (always taken to be some phi (-90<phi<90 deg)
        from positive x-axis!)

    semi_min : float
        length of semiminor axis

    phi : float
        angle in radians of semimajor axis above positive x axis

    mean : array,
        coordinates of the center.

    n_samples : int
        Number of points to sample along ellipse from 0-2pi

    """
    if mean is None:
        mean = [width // 2, width // 2]
    semimaj *= width
    semimin *= width
    mean = np.asarray(mean)
    # Generate data for ellipse structure
    n_samples = 1e6
    theta = np.linspace(0, 2 * np.pi, int(n_samples))
    r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.array([x, y])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                 [np.sin(phi), np.cos(phi)]])
    T = np.dot(R, S)
    data = np.dot(T, data)
    data += mean[:, None]
    data = np.round(data).astype(int)
    data = np.clip(data, 0, width - 1)
    return data


def make_nested_ellipses(width, n_ellipses=1, centers=None, seed=None,
                         max_radius=0.3, smoothing=0.):
    """Creates array of random nested ellipses."""
    rng = np.random.RandomState(seed)
    if smoothing:
        grid = np.arange(width)
        kernel = np.exp(- (grid[:, None] - grid[None, :]) ** 2 / smoothing)
    ellipses = []
    if centers is None:
        centers = np.linspace(width // 3, 2 * width // 3, n_ellipses)
        centers = np.vstack([centers, centers]).T.astype(float)
    for ii in range(n_ellipses):
        mean = centers[ii]
        semimaj = rng.rand() * max_radius + 0.25
        semimin = rng.rand() * 0.15 + 0.05

        phi = rng.rand() * np.pi - np.pi / 2
        x1, y1 = make_ellipse(width, mean, semimaj, semimin, phi)
        mean = mean + rng.rand(2) * semimaj
        semimaj = rng.rand() * 0.05 + 0.1
        semimin = rng.rand() * 0.1 + 0.05
        phi = rng.rand() * np.pi - np.pi / 2
        x2, y2 = make_ellipse(width, mean, semimaj, semimin, phi)

        img = np.zeros((width, width))
        img[x1, y1] = 1.

        img[x2, y2] = 1.

        if smoothing:
            img = kernel.dot(kernel.dot(img).T).T
        ellipses.append(img)
    ellipses = np.array(ellipses)
    return ellipses

seed = 42
n_samples = 2
width = 20
n_features = width ** 2
imgs_np = make_nested_ellipses(width, n_samples, seed=seed)
imgs_np /= imgs_np.sum((1, 2))[:, None, None]

imgs_list = np.split(imgs_np, n_samples, axis=0)

flattened_imgs = []
for img in imgs_list:
    flattened_imgs.append(img.flatten())
    # flattened_imgs.append(imgs_list[0].flatten())


log = greedy_primal_dual(flattened_imgs)
print('DEMD', log['primal objective'])


vecsize = n_samples*n_features

data = flattened_imgs
data = np.array(data)
# data = vectorize(data, vecsize)

x = img_minimize(img_demd_func, data, n_samples, n_features, vecsize, width, niters=10000, lr=1e-7, print_rate=100)

f, axes = plt.subplots(2, n_samples, figsize=(12, 8))
for i, ax in enumerate(axes.ravel()):
    # time_value = times[i]
    # name = titles[i]
    # tt = " Ran in %s s" % np.round(time_value, 2)
    print(i)
    if i < n_samples:
        ax.imshow(np.squeeze(imgs_list[i]), cmap="hot_r")
    else:
        ax.imshow(np.squeeze(np.reshape(x[i-n_samples], [width, width])), cmap="hot_r")
    ax.set_xticks([])
    ax.set_yticks([])
plt.subplots_adjust(hspace=0.4)
plt.savefig("demd_test.pdf", bbox_inches="tight")



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

