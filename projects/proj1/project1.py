import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator


def init_plane_numpy(N):
    plane = np.zeros((2 ** N + 1, 2 ** N + 1))
    plane[0, 0] = np.random.randn() * 100
    plane[0, -1] = np.random.randn() * 100
    plane[-1, 0] = np.random.randn() * 100
    plane[-1, -1] = np.random.randn() * 100
    return plane

def init_plane_list(N):
    ind = 2 ** N + 1
    plane = []
    for _ in range(ind):
        row = []
        for _ in range(ind):
            row.append(0)
        plane.append(row)
    return plane

def create_map(plane, map_file, colormap):
    plt.figure(figsize=(10,10))
    plt.imshow(plane, cmap=colormap)
    if map_file is None:
        plt.show()
    else:
        plt.savefig(map_file, dpi=96)


def create_surface(plane, N, surf_file, colormap):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    # Make data.
    X = np.arange(2 ** N + 1)
    Y = np.arange(2 ** N + 1)
    X, Y = np.meshgrid(X, Y)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, plane, cmap=colormap,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.imshow(plane, cmap=colormap)
    if surf_file is None:
        plt.show()
    else:
        plt.savefig(surf_file, dpi=96)


def save_plane(matrix_file, plane):
    if matrix_file is not None:
        np.save(matrix_file, plane)


def create_centers(centers, k, step):
    new_centers = []
    for c in centers:
        new_centers.append([c[0] - step, c[1] - step])
        new_centers.append([c[0] - step, c[1] + step])
        new_centers.append([c[0] + step, c[1] - step])
        new_centers.append([c[0] + step, c[1] + step])
    if k == 1:
        return new_centers
    else:
        return create_centers(new_centers, k-1, step//2)



def gen(plane, centers, k, step, sigma, connect):
    new_step = step // 2
    for c in centers:
        plane[c[0]][c[1]] = (plane[c[0] - step][c[1] - step] + plane[c[0] + step][c[1] - step]
                             + plane[c[0] - step][c[1] + step] + plane[c[0] + step][c[1] + step]) / 4
        perturbation = 2 ** k * sigma * np.random.normal()
        plane[c[0]][c[1]] += perturbation
    new_centers = []
    for c in centers:
        plus = 0
        new_centers.append([c[0] - new_step, c[1] - new_step])
        plane[c[0]][c[1] - step] = plane[c[0]][c[1]] + plane[c[0] - step][c[1] - step] + plane[c[0] + step][c[1] - step]
        if 0 <= c[1] - 2 * step:
            plane[c[0]][c[1] - step] += plane[c[0]][c[1] - 2 * step]
            plus = 1
        elif connect:
            plane[c[0]][c[1] - step] += plane[c[0]][len(plane) - step]
            plus = 1
        plane[c[0]][c[1] - step] /= (3 + plus)
        perturbation = 2 ** (k - 1) * sigma * np.random.normal()
        plane[c[0]][c[1] - step] += perturbation

        plus = 0
        new_centers.append([c[0] - new_step, c[1] + new_step])
        plane[c[0] - step][c[1]] = plane[c[0]][c[1]] + plane[c[0] - step][c[1] - step] + plane[c[0] - step][c[1] + step]
        if 0 <= c[0] - 2 * step:
            plane[c[0] - step][c[1]] += plane[c[0] - 2 * step][c[1]]
            plus = 1
        elif connect:
            plane[c[0]][c[1] - step] += plane[c[0]][len(plane) - step]
            plus = 1
        plane[c[0] - step][c[1]] /= (3 + plus)
        perturbation = 2 ** (k - 1) * sigma * np.random.normal()
        plane[c[0] - step][c[1]] += perturbation

        plus = 0
        new_centers.append([c[0] + new_step, c[1] - new_step])
        if c[1] + step + 1 == len(plane):
            plane[c[0]][c[1] + step] = plane[c[0]][c[1]] + plane[c[0] - step][c[1] + step] + plane[c[0] + step][c[1] + step]
            if connect:
                plane[c[0]][c[1] + step] += plane[c[0]][step]
                plus = 1
            plane[c[0]][c[1] + step] /= (3 + plus)
            perturbation = 2 ** (k - 1) * sigma * np.random.normal()
            plane[c[0]][c[1] + step] += perturbation

        plus = 0
        new_centers.append([c[0] + new_step, c[1] + new_step])
        if c[0] + step + 1 == len(plane):
            plane[c[0] + step][c[1]] = plane[c[0]][c[1]] + plane[c[0] + step][c[1] - step] + plane[c[0] + step][c[1] + step]
            if connect:
                plane[c[0] + step][c[1]] += plane[step][c[1]]
                plus = 1
            plane[c[0] + step][c[1]] /= (3 + plus)
            perturbation = 2 ** (k - 1) * sigma * np.random.normal()
            plane[c[0] + step][c[1]] += perturbation

    if k == 1:
        return plane

    return gen(plane, new_centers, k-1, new_step, sigma, connect)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=10, help='rząd macierzy')
    parser.add_argument('--sigma', type=float, default=1.0, help='stopień “górzystości” terenu')
    parser.add_argument('--map_file', type=str, default=None, help=' nazwa pliku (pdf lub png) z mapą')
    parser.add_argument('--surf_file', type=str, default=None, help='nazwa pliku (pdf lub png), z wykresem płaszczyzny')
    parser.add_argument('--colormap', type=str, default='terrain', help='mapa kolorów z matplotlib')
    parser.add_argument('--matrix_file', type=str, default=None, help='zapis macierzy do pliku')

    args = parser.parse_args()
    plane = init_plane_numpy(args.N)
    center = len(plane) // 2
    init_centers = [[center, center]]
    gen(plane, init_centers, args.N, step=2**(args.N-1), sigma=args.sigma, connect=False)
    create_map(plane, args.map_file, args.colormap)
    create_surface(plane, args.N, args.surf_file, args.colormap)
    save_plane(args.matrix_file, plane)
