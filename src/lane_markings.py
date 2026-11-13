"""Require a distinct, parallel companion before proposing a double white marking."""

import numpy as np


def sampled_line(points):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    lengths = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if lengths[-1] < 20:
        return None
    positions = np.linspace(0, lengths[-1], min(160, max(16, int(lengths[-1] / 2))))
    return np.column_stack([np.interp(positions, lengths, xy[:, axis]) for axis in (0, 1)])


def parallel_pair(first, second, scale):
    if first is None or second is None:
        return False
    delta_a, delta_b = first[-1] - first[0], second[-1] - second[0]
    length_a, length_b = np.linalg.norm(delta_a), np.linalg.norm(delta_b)
    if min(length_a, length_b) < 20 * scale or min(length_a, length_b) < 0.70 * max(length_a, length_b):
        return False
    direction = delta_a / length_a
    if abs(np.dot(direction, delta_b / length_b)) < 0.985:
        return False
    normal = np.array([-direction[1], direction[0]])
    offsets = second[None, :, :] - first[:, None, :]
    distances = np.linalg.norm(offsets, axis=2)
    for axis in (0, 1):
        nearest = distances.argmin(axis=axis)
        if axis == 1:
            matches = offsets[np.arange(len(first)), nearest]
        else:
            matches = offsets[nearest, np.arange(len(second))]
        lateral = matches @ normal / scale
        longitudinal = np.abs(matches @ direction) / scale
        # Two sides of a single detected stroke often differ by only a pixel or two.
        valid = (np.abs(lateral) >= 4) & (np.abs(lateral) <= 22) & (longitudinal <= 4)
        if max(np.mean(valid & (lateral > 0)), np.mean(valid & (lateral < 0))) < 0.80:
            return False
    return True


def filter_double_white(shapes, width):
    indices = [index for index, shape in enumerate(shapes)
               if shape["type"] == "polyline" and shape["label"] == "lane/double white"]
    if not indices:
        return shapes, 0
    samples = {index: sampled_line(shapes[index]["points"]) for index in indices}
    singles = [line for shape in shapes if shape["type"] == "polyline" and shape["label"] == "lane/single white"
               if (line := sampled_line(shape["points"])) is not None]
    scale = width / 1280
    ambiguous = set()
    for index, line in samples.items():
        if line is None:
            continue
        for single in singles:
            delta = line[:, None, :] - single[None, :, :]
            near = np.linalg.norm(delta, axis=2) <= 3 * scale
            if near.any(axis=1).mean() >= 0.65:
                ambiguous.add(index)
                break
    confirmed = set()
    for position, index in enumerate(indices):
        for other in indices[position + 1:]:
            if index not in ambiguous and other not in ambiguous and parallel_pair(samples[index], samples[other], scale):
                confirmed.update((index, other))
    rejected = set(indices) - confirmed
    return [shape for index, shape in enumerate(shapes) if index not in rejected], len(rejected)
