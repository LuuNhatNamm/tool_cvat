"""Conservative lane geometry for front-facing road images."""

import cv2
import numpy as np


AREA_LABELS = {"area/direct", "area/alternative"}
ROAD_PROMPT = "road surface"
BOUNDARY_PROMPTS = ("single white lane marking", "curb edge between the roadway and sidewalk")
DIVIDER_PROMPT = "yellow lane marking"


def driving_lanes(road, lines, ignored, white=None, divider_lines=()):
    height, width = road.shape
    x = width / 2
    blocked = np.flatnonzero(ignored[:, int(x)])
    bottom = int(blocked[0]) - 1 if len(blocked) else int(height * 0.90)
    if bottom < height * 0.57:
        return None, None

    left, right = [], []
    for points in lines:
        coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        y = coords[:, 1]
        span = float(np.ptp(y))
        if span < 0.09 * height or bottom - y.max() > 0.22 * height:
            continue
        slope, intercept = np.polyfit(y, coords[:, 0], 1)
        error = np.mean(np.abs(coords[:, 0] - (slope * y + intercept)))
        if error > 0.012 * width:
            continue
        at_bottom = slope * bottom + intercept
        if not -0.20 * width <= at_bottom <= 1.20 * width:
            continue
        if not 0.055 * width <= abs(at_bottom - x) <= 0.70 * width:
            continue
        at_far = slope * (height * 0.43) + intercept
        if not 0.12 * width <= at_far <= 0.88 * width:
            continue
        if at_bottom < x and slope < -0.25:
            left.append((at_bottom, slope, intercept, float(y.min())))
        elif at_bottom > x and slope > 0.25:
            right.append((at_bottom, slope, intercept, float(y.min())))

    if not left or not right:
        return None, None
    l0, ls, li, ly = max(left, key=lambda line: line[0])
    r0, rs, ri, ry = min(right, key=lambda line: line[0])
    if not 0.20 * width <= min(r0, width) - max(l0, 0) <= 0.90 * width:
        return None, None
    if white is not None:
        # Dashes nearer the horizon can extend the observed left boundary.
        ys, xs = np.nonzero(white & ~ignored)
        aligned = (ys >= height * 0.38) & (ys < ly) & (np.abs(xs - (ls * ys + li)) < 0.022 * width)
        if np.count_nonzero(aligned) >= 15:
            ly = int(ys[aligned].min())
    top = max(int(height * 0.38), round(max(ly, ry) - height * 0.13),
              int(np.ceil((li - ri + 0.025 * width) / (rs - ls))))
    if bottom - top < height * 0.10:
        return None, None
    corridor = np.zeros_like(road, dtype=np.uint8)
    points = [(round(ls * top + li), top), (round(rs * top + ri), top),
              (round(r0), bottom), (round(l0), bottom)]
    cv2.fillPoly(corridor, [np.asarray(points, dtype=np.int32)], 1)
    region = (corridor != 0) & road & ~ignored
    if np.count_nonzero(region) < 0.65 * np.count_nonzero(corridor):
        return None, None
    count, components, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=4)
    if count < 2:
        return None, None
    main = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    result = components == main
    # Require the visible lane to lead back to the camera, not an isolated distant patch.
    if not np.any(result[max(top, bottom - round(height * 0.08)):bottom + 1, int(x)]):
        return None, None
    alternative = adjacent_left(road & ~ignored, result, white, divider_lines, (ls, li), top, bottom)
    return result.astype(np.uint8), alternative


def adjacent_left(road, direct, white, divider_lines, shared, top, bottom):
    """Look for a dashed white separator inside the yellow center divider."""
    if white is None:
        return None
    height, width = road.shape
    slope, intercept = shared
    support = cv2.dilate(white.astype(np.uint8), np.ones((5, 7), np.uint8))
    rows = np.arange(top, bottom + 1)
    columns = np.rint(rows * slope + intercept).astype(int).clip(0, width - 1)
    runs = np.flatnonzero(np.diff(np.r_[0, support[rows, columns], 0]))
    dashes = [(start, end) for start, end in zip(runs[::2], runs[1::2]) if end - start >= 0.008 * height]
    if len(dashes) < 2 or not any(b[0] - a[1] >= 0.018 * height for a, b in zip(dashes, dashes[1:])):
        return None

    candidates = []
    for points in divider_lines:
        xy = np.asarray(points).reshape(-1, 2)
        if np.ptp(xy[:, 1]) < 0.045 * height:
            continue
        outer_slope, outer_intercept = np.polyfit(xy[:, 1], xy[:, 0], 1)
        if outer_slope >= slope or np.mean(np.abs(xy[:, 0] - (outer_slope * xy[:, 1] + outer_intercept))) > 0.012 * width:
            continue
        # Avoid extending a divider far outside its visible evidence.
        y0 = max(top, int(xy[:, 1].min()))
        if xy[:, 0].min() <= 0.02 * width and outer_slope * bottom + outer_intercept < 0:
            # The divider exits the image on the left; the image edge bounds the rest.
            y1 = bottom
        else:
            last_supported = int(xy[:, 1].max() + 0.16 * height)
            widest = int((0.90 * width - intercept + outer_intercept) / (slope - outer_slope))
            y1 = min(bottom, last_supported, widest)
        gap0 = (slope - outer_slope) * y0 + intercept - outer_intercept
        gap1 = (slope - outer_slope) * y1 + intercept - outer_intercept
        if y1 - y0 < 0.14 * height or not 0.04 * width <= gap0 < gap1:
            continue
        candidates.append((gap0, y0, y1, outer_slope, outer_intercept))
    if not candidates:
        return None
    _, y0, y1, outer_slope, outer_intercept = min(candidates)
    band = np.zeros_like(direct, dtype=np.uint8)
    corners = [(round(s * y + b), y) for s, b, y in
               ((outer_slope, outer_intercept, y0), (slope, intercept, y0),
                (slope, intercept, y1), (outer_slope, outer_intercept, y1))]
    cv2.fillPoly(band, [np.asarray(corners, dtype=np.int32)], 1)
    region = (band != 0) & road & ~direct
    if not region.any() or region.sum() < 0.65 * band.sum():
        return None
    _, components, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=4)
    main = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (components == main).astype(np.uint8)
