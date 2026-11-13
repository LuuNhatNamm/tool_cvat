"""Convert semantic class pixels to editable, hole-free CVAT polygons."""

import cv2
import numpy as np


# Continuous semantic surfaces can share an outline. Keep instance geometry separate.
MERGED_SURFACES = {"building", "road", "sidewalk", "sky", "terrain", "vegetation", "wall"}


def merged_surface_polygons(surfaces, protected):
    """Union a class before tracing it and bridge only small, unclaimed gaps."""
    height, width = protected.shape
    scale = min(width / 1280, height / 720)
    radius = max(1, min(4, round(2 * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    max_hole = max(4, min(192, round(64 * scale ** 2)))
    occupied = np.zeros(protected.shape, dtype=np.uint32)
    for surface in surfaces.values():
        occupied += surface
    claimed = protected.copy()
    for class_id, surface in surfaces.items():
        blocked = claimed | (occupied > surface)
        binary = surface.astype(np.uint8)
        additions = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel).astype(bool) & ~surface

        # Fill tiny enclosed voids, but preserve the entire hole if another label uses it.
        count, components, stats, _ = cv2.connectedComponentsWithStats((~surface).astype(np.uint8), connectivity=4)
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            if area > max_hole or x == 0 or y == 0 or x + w == width or y + h == height:
                continue
            hole = components[y:y+h, x:x+w] == index
            if not np.any(hole & blocked[y:y+h, x:x+w]):
                additions[y:y+h, x:x+w] |= hole
            else:
                additions[y:y+h, x:x+w] &= ~hole
        additions &= ~blocked
        claimed |= additions
        binary[additions] = 1
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        for index in range(1, count):
            x, y, w, h, _ = map(int, stats[index])
            region = (components[y:y+h, x:x+w] == index).astype(np.uint8)
            yield from ((class_id, points) for points in outlines(region, x, y))


def simplified_contour(contour, pixels):
    x, y, width, height = cv2.boundingRect(contour)
    crop = pixels[y:y + height, x:x + width]
    offset = np.array([[[x, y]]], dtype=np.int32)
    reference = np.zeros(crop.shape, dtype=np.uint8)
    cv2.fillPoly(reference, [contour - offset], 1)
    count = np.count_nonzero(reference)
    alternatives = [contour - offset]
    if count >= 400:
        inset = cv2.erode(reference, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        inset_contours, _ = cv2.findContours(inset, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(inset_contours) == 1:
            alternatives.append(inset_contours[0])
    best = contour - offset
    for outline in alternatives:
        for tolerance in (4.0, 3.0, 2.0, 1.0):
            candidate = cv2.approxPolyDP(outline, tolerance, True)
            if not 3 <= len(candidate) < len(best) or cv2.contourArea(candidate) <= 0:
                continue
            painted = np.zeros_like(reference)
            cv2.fillPoly(painted, [candidate], 1)
            if (not np.any(painted & (~crop | ~reference))
                    and np.count_nonzero(reference & ~painted) <= 0.02 * count):
                best = candidate
                break
    return best + offset


def outlines(region, left, top):
    pending = [(region, left, top)]
    while pending:
        part, x0, y0 = pending.pop()
        if not part.any():
            continue
        # Doubling preserves one-pixel-wide features as non-degenerate polygons.
        pixels = np.repeat(np.repeat(part, 2, axis=0), 2, axis=1)
        contours, hierarchy = cv2.findContours(pixels, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c, info in zip(contours, hierarchy[0]) if info[3] != -1]
        if holes:
            x, y, w, h = cv2.boundingRect(max(holes, key=cv2.contourArea))
            col, row = (x + w // 2) // 2, (y + h // 2) // 2
            if w >= h:
                pending.extend(((part[row:], x0, y0 + row), (part[:row], x0, y0)))
            else:
                pending.extend(((part[:, col:], x0 + col, y0), (part[:, :col], x0, y0)))
            continue
        for contour in contours:
            points = simplified_contour(contour, pixels).reshape(-1, 2)
            points = (points + 0.5) / 2 + (x0, y0)
            yield points.ravel().tolist()


def editable_polygons(ids, allowed):
    for label in sorted(allowed):
        binary = (ids == label).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        for i in range(1, count):
            x, y, width, height, _ = map(int, stats[i])
            region = (components[y:y + height, x:x + width] == i).astype(np.uint8)
            for points in outlines(region, x, y):
                yield label, points
