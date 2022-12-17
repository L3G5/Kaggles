import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np
import pandas as pd
from functools import *
from itertools import *
from pathlib import Path
from PIL import Image
from tqdm import tqdm, trange
import cv2
import numba as nb



# Functions to map between cartesian coordinates and array indexes
@nb.njit
def cartesian_to_array(x, y, shape):
    m, n = shape[:2]
    i = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i, j

@nb.njit
def array_to_cartesian(i, j, shape):
    m, n = shape[:2]
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    y = (n - 1) // 2 - i
    x = j - (n - 1) // 2
    return x, y


point = (1, 8)
shape = (9, 9, 3)
assert cartesian_to_array(*array_to_cartesian(*point, shape), shape) == point


# Functions to map an image between array and record formats
def image_to_dict(image):
    image = np.atleast_3d(image)
    kv_image = {}
    for i, j in product(range(len(image)), repeat=2):
        kv_image[array_to_cartesian(i, j, image.shape)] = tuple(image[i, j])
    return kv_image


def image_to_df(image):
    return pd.DataFrame(
        [(x, y, r, g, b) for (x, y), (r, g, b) in image_to_dict(image).items()],
        columns=['x', 'y', 'r', 'g', 'b']
    )


def df_to_image(df):
    side = int(len(df) ** 0.5)  # assumes a square image
    return df.set_index(['x', 'y']).to_numpy().reshape(side, side, -1)


def imread(path):
    if isinstance(path, Path):
        path = path.as_posix()
    return cv2.imread(path)[:, :, ::-1] / 255


@nb.njit
def get_position(config):
    return config.sum(0)

@nb.njit
def rotate_link(vector, direction):
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif y > x and y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif y >= x and y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return (x, y)

@nb.njit
def rotate(config, i, direction):
    config = config.copy()
    config[i] = rotate_link(config[i], direction)
    return config


# compress a path between two points
@nb.njit
def compress_path(path):
    n_joints = path.shape[1]
    r = np.zeros((n_joints, path.shape[0], 2), dtype=path.dtype)
    l = np.zeros(n_joints, dtype='int')
    for j in range(len(path)):
        for i in range(n_joints):
            if l[i] == 0 or (r[i][l[i] - 1] != path[j, i]).any():
                r[i, l[i]] = path[j, i]
                l[i] += 1
    r = r[:, :l.max()]

    for i in range(n_joints):
        for j in range(l[i], r.shape[1]):
            r[i, j] = r[i, j - 1]
    r = r.transpose(1, 0, 2)

    return r


@nb.njit
def get_direction(u, v):
    """Returns the sign of the angle from u to v."""
    # direction = np.sign(np.cross(u, v))
    direction = np.sign(u[0] * v[1] - u[1] * v[0])
    if direction == 0 and (u * v).sum() < 0:
        direction = 1
    return direction


@nb.njit
def get_radius(config):
    r = 0
    for link in config:
        r += np.abs(link).max()
    return r

@nb.njit
def get_radii(config):
    radii = np.cumsum(np.maximum(np.abs(config[:, 0]), np.abs(config[:, 1]))[::-1])[::-1]
    return np.append(radii, np.zeros(1, dtype='int'))


@nb.njit
def get_path_to_point(config, point):
    """Find a path of configurations to `point` starting at `config`."""
    config_start = config.copy()
    radii = get_radii(config)

    # Rotate each link, starting with the largest, until the point can
    # be reached by the remaining links. The last link must reach the
    # point itself.
    for i in range(len(config)):
        link = config[i]
        base = get_position(config[:i])
        relbase = point - base
        position = get_position(config[:i+1])
        relpos = point - position
        radius = radii[i + 1]
        # Special case when next-to-last link lands on point. 
        if radius == 1 and (relpos == 0).all():
            config = rotate(config, i, 1)
            if (get_position(config) == point).all():
                break
            else:
                continue
        while np.max(np.abs(relpos)) > radius:
            direction = get_direction(link, relbase)
            config = rotate(config, i, direction)
            link = config[i]
            base = get_position(config[:i])
            relbase = point - base
            position = get_position(config[:i+1])
            relpos = point - position
            radius = get_radius(config[i + 1:])

    assert (get_position(config) == point).all()
    path = get_path_to_configuration(config_start, config)

    return path


@nb.njit
def get_path_to_configuration(from_config, to_config):
    path = np.expand_dims(from_config, 0).copy()
    config = from_config.copy()
    while (config != to_config).any():
        for i in range(len(config)):
            config = rotate(config, i, get_direction(config[i], to_config[i]))
        path = np.append(path, np.expand_dims(config, 0), 0)
    assert (path[-1] == to_config).all()
    return path


# Functions to compute the cost function

# Cost of reconfiguring the robotic arm: the square root of the number of links rotated
@nb.njit
def reconfiguration_cost(from_config, to_config):
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    assert diffs.max() <= 1
    return np.sqrt(diffs.sum())


# Cost of moving from one color to another: the sum of the absolute change in color components
@nb.njit
def color_cost(from_position, to_position, image, color_scale=3.0):
    return np.abs(image[to_position] - image[from_position]).sum() * color_scale


# Total cost of one step: the reconfiguration cost plus the color cost
@nb.njit
def step_cost(from_config, to_config, image):
    pos_from = get_position(from_config)
    pos_to = get_position(to_config)
    from_position = cartesian_to_array(pos_from[0], pos_from[1], image.shape)
    to_position = cartesian_to_array(pos_to[0], pos_to[1], image.shape)
    return (
        reconfiguration_cost(from_config, to_config) +
        color_cost(from_position, to_position, image)
    )

# Compute total cost of path over image
@nb.njit
def total_cost(path, image):
    cost = 0
    for i in range(1, len(path)):
        cost += step_cost(path[i - 1], path[i], image)
    return cost


def get_origin(size):
    assert size % 2 == 1
    radius = size // 2
    p = [1]
    for power in range(0, 8):
        p.append(2**power)
        if sum(p) == radius:
            break
    else:
        assert False
    p = p[::-1]
    config = np.array([(p[0], 0)] + [(-pp, 0) for pp in p[1:]])
    return config


def points_to_path(points, size=257):
    origin = get_origin(size)

    visited = set()
    path = [origin]
    for p in points:
        config = path[-1]
        if tuple(p) not in visited:
            candy_cane_road = get_path_to_point(config, p)[1:]
            if len(candy_cane_road) > 0:
                visited |= set([tuple(get_position(r)) for r in candy_cane_road])
            path.extend(candy_cane_road)
    # Back to origin
    candy_cane_road = get_path_to_configuration(path[-1], origin)[1:]
    visited |= set([tuple(get_position(r)) for r in candy_cane_road])
    path.extend(candy_cane_road)
    
    assert len(visited) == size**2, f'Visited {len(visited)} points out of {size**2}'
    
    return np.array(path)


@nb.njit
def dir2idx(dx, dy):
    assert dx >= -1 and dx <= 1 and dy >= -1 and dy <= 1
    assert not (dx == 0 and dy == 0)
    idx = (dx + 1) * 3 + dy + 1
    if idx > 4:
        idx = idx - 1
    return idx

@nb.njit
def idx2dir(idx):
    assert idx >= 0 and idx < 8
    if idx >= 4:
        idx = idx + 1
    dx = idx // 3 - 1
    dy = idx % 3 - 1
    return dx, dy

@nb.njit
def idx2newij(i, j, idx):
    dx, dy = idx2dir(idx)
    new_i = i - dy
    new_j = j + dx
    return new_i, new_j

# test
for i in range(8):
    assert dir2idx(*idx2dir(i)) == i


def plot_traj(points, image, figsize=(20,20)):
    origin = np.array([0, 0])
    lines = []
    if not (origin == points[0]).all():
        lines.append([origin, points[0]])
    for i in range(1, len(points)):
        lines.append([points[i - 1], points[i]])
    if not (origin == points[1]).all():
        lines.append([points[-1], origin])

    colors = []
    for l in lines:
        dist = np.abs(l[0] - l[1]).max()
        if dist <= 2:
            colors.append('b')
        else:
            colors.append('r')

    lc = mc.LineCollection(lines, colors=colors)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.add_collection(lc)

    radius = image.shape[0] // 2
    ax.matshow(image * 0.8 + 0.2, extent=(-radius-0.5, radius+0.5, -radius-0.5, radius+0.5))
    ax.grid(None)

    ax.autoscale()
    fig.show()


def get_baseline(image):
    # Generate points
    points_baseline = []
    flag = True
    for split in range(2):
        for i in reversed(range(257)) if split%2==0 else range(257):
            if not flag:
                for j in range(128*split, 128+129*split):
                    points_baseline.append((j-128,i-128))
            else:
                for j in reversed(range(128*split, 128+129*split)):
                    points_baseline.append((j-128,i-128))
            flag = not flag
        flag = False
    points_baseline = np.array(points_baseline)

    # Make path
    path_baseline = points_to_path(points_baseline)

    # Compute cost
    score_baseline = total_cost(path_baseline, image)

    return points_baseline, path_baseline, score_baseline

def get_neighbors(config):
    config = np.array(config)
    nhbrs = (
        reduce(lambda x, y: rotate(x, *y), enumerate(directions), config)
        for directions in product((-1, 0, 1), repeat=len(config))
    )
    return list(filter(lambda c: c.tolist() != config.tolist(), nhbrs))

def plot_neighbors(config, figsize=(10,10)):
    radius = 2**(len(config)-1)
    config = np.array(config)
    points_costs =[(get_position(c), reconfiguration_cost(config, c)) for c in get_neighbors(config) if not \
        np.all(np.equal(get_position(c), get_position(np.array(config))))]
    points = np.unique(([x[0] for x in points_costs]), axis = 0)
    min_costs = [min([y[1] for y in points_costs if np.all(np.equal(y[0], x))]) for x in points]
    plt.figure(figsize=figsize, dpi=80)
    plt.scatter(*zip(*points), c = 'b')
    for i, txt in enumerate(min_costs):
        plt.annotate(f'{txt:.2f}', tuple(points[i]), xytext=(1., 1.), textcoords='offset points')
    plt.scatter(*zip(get_position(np.array(config))), c = 'r')
    plt.grid(None)
    plt.ylim(-radius-0.5, radius+0.5)
    plt.xlim(-radius-0.5, radius+0.5)
    plt.gca().set_aspect('equal', 'box')
    plt.title(str(list(map(tuple, config.tolist()))))

def get_square(link_length):
    link = (link_length, 0)
    coords = [link]
    for _ in range(8 * link_length - 1):
        link = rotate_link(link, direction=1)
        coords.append(link)
    return coords