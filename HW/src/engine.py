from options import LBEParams
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from tqdm import tqdm
from IPython.display import clear_output
import imageio
from os import listdir, getcwd, remove
from os.path import isfile, join
from natsort import natsorted


LATTICE_VELOCITIES = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]])
WEIGHTS = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])


@jit
def equilibrium(rho, u, v, t, nx, ny):
    usqr = u[0] ** 2 + u[1] ** 2
    feq = np.zeros((9, nx, ny))
    for i in range(9):
        cu = v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :]
        feq[i, :, :] = rho * t[i] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usqr)
    return feq


@jit
def macroscopic(fin, v, nx, ny):
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u


def inletVel(d, x, y):
    uLB, ly = LBEParams.uLB, LBEParams.ny - 1
#     delta = 1e-4 * np.sin(y / ly * 2 * np.pi)
    delta = 1e-4
    return (1 - d) * uLB * (1 + delta)


class Obstacle:
    def __init__(self):
        self.cx = LBEParams.cx
        self.cy = LBEParams.cy
        self.r = LBEParams.r
    
    def _isInner(self, x, y):
        return (x - self.cx)**2 + (y - self.cy)**2 < self.r**2

    def location(self):
        return np.fromfunction(self._isInner, (LBEParams.nx, LBEParams.ny))


class Solver:
    def __init__(self):
        self.obstacle = Obstacle()
        # plt.imshow(self.obstacle.location().T)
        self.pics_path = getcwd()[:-3] + 'pics\\'
        self.gif_path = getcwd()[:-3]

    def solve(self, steps=7700, n_pics=77, **kwargs):

        nx, ny, omega = LBEParams.nx, LBEParams.ny, LBEParams.omega

        col_0 = np.array([0, 1, 2])
        col_1 = np.array([3, 4, 5])
        col_2 = np.array([6, 7, 8])
        
        vel = np.fromfunction(inletVel, (2, nx, ny))
        v = LATTICE_VELOCITIES
        t = WEIGHTS
        fin = equilibrium(1, vel, v, t, nx, ny)
        
        for time in tqdm(range(steps)):
            
            fin[col_2, -1, :] = fin[col_2, -2, :]

            rho, u = macroscopic(fin, v, nx, ny)

            u[:, 0, :] = vel[:, 0, :]
            rho[0, :] = 1 / (1 - u[0, 0, :]) * (np.sum(fin[col_1, 0, :], axis=0) + 2*np.sum(fin[col_2, 0, :], axis=0))

            feq = equilibrium(rho, u, v, t, nx, ny)
            fin[col_0, 0, :] = feq[col_0, 0, :] + fin[col_2, 0, :] - feq[col_2, 0, :]

            fout = fin - omega * (fin - feq)

            for i in range(9):
                fout[i, self.obstacle.location()] = fin[8 - i, self.obstacle.location()]

            for i in range(9):
                fin[i, :, :] = np.roll(
                  np.roll(
                        fout[i, :, :], v[i, 0], axis=0
                       ),
                  v[i, 1], axis=1
                  )

            cmap = cm.Reds if 'cmap' not in kwargs else kwargs['cmap']

            uNorm = np.sqrt(u[0] ** 2 + u[1] ** 2)
            uNorm[self.obstacle.location()] = np.nan
            rho[self.obstacle.location()] = np.nan

            if time % (steps / n_pics) == 0 and time != 0:

                fig = plt.figure(figsize=(10, 10), dpi=100)

                plt.subplot2grid((11, 9), (0, 0), colspan=9, rowspan=3)
                ax = plt.gca()
                plt.title("time: {0:01d}".format(int(time*1)))
                im = plt.imshow(uNorm.T, cmap=cmap)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(im, cax=cax)
                cb.set_label('$|u|$')

                plt.subplot2grid((11, 9), (4, 0), colspan=9, rowspan=3)
                ax = plt.gca()
                im = plt.imshow(rho.T, cmap=cmap)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(im, cax=cax)
                cb.set_label(r'$\rho$')

                plt.subplot2grid((11, 9), (8, 0), colspan=9, rowspan=3)
                ax = plt.gca()
                im = plt.imshow(rho.T / 3, cmap=cmap)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb = plt.colorbar(im, cax=cax)
                cb.set_label('$p$')

                # plt.show()
                # clear_output(True)
                fig.savefig(self.pics_path + "time{0:01d}.png".format(time))
                plt.close()

    def makeGIF(self, remove_images=False):
        filenames = natsorted([f for f in listdir(self.pics_path) if isfile(join(self.pics_path, f))])
        with imageio.get_writer(self.gif_path + 'gif.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(self.pics_path + filename)
                writer.append_data(image)

        if remove_images:
            for filename in set(filenames):
                remove(self.pics_path + filename)
