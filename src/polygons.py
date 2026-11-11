"""
"Spot the intruder" image datasets: four white polygons on a black background, arranged in a 2x2 grid.
Three of them are the same shape (up to rotation and scaling) and one, the intruder, is a different
shape of the same area.

Two kinds of base shape are supported:
  - "c4":        a random convex polygon with C4 (90-degree rotational) symmetry. With n_sides=4 this is a square.
  - "irregular": a random convex polygon with no particular symmetry.
The intruder is always a random irregular convex polygon with the same number of sides.

The polygon geometry was originally generated with the help of Gemini.
"""

import csv
import os
import re

import matplotlib

matplotlib.use("Agg")  # render to files only; works on headless machines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# Grid positions, indexed by intruder_label.
POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right"]
CENTERS = [(-2, 2), (2, 2), (-2, -2), (2, -2)]

QUESTION = (
    "This image shows four white shapes arranged in a 2x2 grid. Three of them are the same shape, "
    "up to rotation and scaling, and one of them is different. Which one is the odd one out? "
    "Answer with top-left, top-right, bottom-left or bottom-right."
)


def extract_position(answer):
    """
    The grid position a model answer settles on: the last of top-left / top-right / bottom-left / bottom-right
    (also "upper"/"lower", with or without a hyphen) mentioned in the text, or None if there is none.
    This is a heuristic; check answers by hand when it matters.
    """
    matches = re.findall(r"\b(top|upper|bottom|lower)[\s-]*(left|right)\b", str(answer).lower())
    if not matches:
        return None
    vertical, horizontal = matches[-1]
    vertical = {"upper": "top", "lower": "bottom"}.get(vertical, vertical)
    return f"{vertical}-{horizontal}"


def rotation_matrix(angle):
    """2D counterclockwise rotation matrix."""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])


def polygon_area(vertices):
    """Area of a polygon (shoelace formula)."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def match_area(source, target):
    """Scale `source` so that it has the same area as `target`."""
    return source * np.sqrt(polygon_area(target) / polygon_area(source))


def normalize_polygon(vertices):
    """Center the polygon and scale it so that its RMS radius is 1."""
    vertices = vertices - vertices.mean(axis=0)
    radius = np.sqrt(np.mean(np.sum(vertices ** 2, axis=1)))
    return vertices / radius


def transform_polygon(vertices, scale, rotation, center):
    """Scale, rotate, then translate a polygon."""
    vertices = (vertices * scale) @ rotation_matrix(rotation).T
    return vertices + np.asarray(center)


def generate_c4_polygon(rng, n_sides=4, radius_range=(0.70, 1.00)):
    """
    Random convex polygon with exact C4 symmetry and exactly n_sides vertices (n_sides must be a multiple of 4).
    Picks n_sides/4 points in the first quadrant and adds their rotations by 90, 180 and 270 degrees.
    """
    if n_sides % 4 != 0:
        raise ValueError("n_sides must be a multiple of 4 for C4 symmetry.")
    pts_per_quadrant = n_sides // 4

    while True:
        angles = np.linspace(0.02, np.pi / 2 - 0.02, pts_per_quadrant)
        angles = np.sort(np.clip(angles + rng.uniform(-0.02, 0.02, pts_per_quadrant), 0.01, np.pi / 2 - 0.01))
        radii = rng.uniform(*radius_range, pts_per_quadrant)
        quadrant = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        points = np.concatenate([quadrant @ rotation_matrix(j * np.pi / 2).T for j in range(4)])
        hull = ConvexHull(points)
        if len(hull.vertices) == n_sides:  # reject if some point is not a vertex of the hull
            vertices = points[hull.vertices]
            return vertices - vertices.mean(axis=0)


def generate_irregular_polygon(rng, n_sides=4, radius_range=(0.70, 1.00), angular_noise=0.5):
    """Random convex polygon with exactly n_sides vertices and no particular symmetry."""
    while True:
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        angles = np.sort(angles + rng.uniform(-angular_noise, angular_noise, n_sides))
        radii = rng.uniform(*radius_range, n_sides)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        hull = ConvexHull(points)
        if len(hull.vertices) == n_sides:
            vertices = points[hull.vertices]
            return vertices - vertices.mean(axis=0)


def generate_image(filename, rng, base="c4", n_sides=4):
    """Draw one 2x2 grid with one intruder and save it to `filename`. Returns the intruder's index in POSITIONS."""
    if base == "c4":
        base_polygon = generate_c4_polygon(rng, n_sides)
    elif base == "irregular":
        base_polygon = generate_irregular_polygon(rng, n_sides)
    else:
        raise ValueError(f"unknown base shape {base!r}")
    base_polygon = normalize_polygon(base_polygon)
    intruder = match_area(generate_irregular_polygon(rng, n_sides), base_polygon)

    scales = rng.uniform(0.5, 1.0, 4)
    rotations = rng.uniform(0, 2 * np.pi, 4)
    k = int(rng.integers(0, 4))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    for i in range(4):
        shape = intruder if i == k else base_polygon
        vertices = transform_polygon(shape, scales[i], rotations[i], CENTERS[i])
        ax.add_patch(Polygon(vertices, closed=True, facecolor="white", edgecolor="white"))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return k


def make_polygon_dataset(out_path, n=20, seed=42, base="c4", n_sides=4):
    """
    Generate n images (saved next to out_path) and a csv at out_path with columns
    question, image_path, intruder_label, intruder_position. Returns out_path.
    """
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n):
        image_path = os.path.join(out_dir, f"{stem}_{i}.png")
        k = generate_image(image_path, rng, base=base, n_sides=n_sides)
        rows.append([QUESTION, image_path, k, POSITIONS[k]])

    with open(out_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "image_path", "intruder_label", "intruder_position"])
        writer.writerows(rows)

    print(f"Saved {n} samples to {out_path}")
    return out_path
