"""Shape primitives and deterministic geometric detection.

No machine learning. Every shape is derived from contour geometry (vertices,
angles, circularity, rectangularity), so results are reproducible and
explainable. Each detected primitive records the measurements it was classified
from, which is what lets the engine say *why* it thinks something is a diamond.

Geometry conventions
--------------------
All coordinates are in source-image pixels, origin at the top-left, matching
OpenCV. Boxes are ``(x, y, width, height)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


class ShapeType(str, Enum):
    """Primitive geometry we can recognise. Deliberately coarse.

    ``UNKNOWN`` exists so that an unrecognised primitive is reported rather than
    silently coerced into the nearest common shape.
    """

    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    DIAMOND = "diamond"
    ELLIPSE = "ellipse"
    PARALLELOGRAM = "parallelogram"
    CYLINDER = "cylinder"
    TRIANGLE = "triangle"
    ARROWHEAD = "arrowhead"
    LINE = "line"
    CONTAINER = "container"
    UNKNOWN = "unknown"


@dataclass
class Shape:
    """One detected primitive."""

    id: str
    type: ShapeType
    box: Box
    area: float
    perimeter: float
    #: Geometric evidence, kept so a classification can be audited.
    measures: dict = field(default_factory=dict)
    confidence: float = 0.0
    #: Contour nesting depth: 0 = top level.
    depth: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    @property
    def aspect(self) -> float:
        _, _, w, h = self.box
        return w / h if h else float("inf")

    def corners(self) -> list[tuple[float, float]]:
        x, y, w, h = self.box
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def contains_point(self, px: float, py: float, margin: float = 0.0) -> bool:
        x, y, w, h = self.box
        return (x - margin) <= px <= (x + w + margin) and (y - margin) <= py <= (y + h + margin)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "box": list(self.box),
            "area": round(self.area, 2),
            "perimeter": round(self.perimeter, 2),
            "confidence": round(self.confidence, 3),
            "depth": self.depth,
            "measures": {k: _round(v) for k, v in self.measures.items()},
        }


@dataclass
class Segment:
    """A straight connector between two points."""

    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float = 1.0
    #: True when the stroke is broken into regularly spaced pieces. Detected, not
    #: assumed: Hough's maxLineGap is wide enough to bridge a dash pattern, so a
    #: dashed connector arrives here looking identical to a solid one.
    dashed: bool = False

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0]))

    @property
    def is_horizontal(self) -> bool:
        return abs(self.angle_deg) < 15 or abs(abs(self.angle_deg) - 180) < 15

    @property
    def is_vertical(self) -> bool:
        return abs(abs(self.angle_deg) - 90) < 15

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)

    def to_dict(self) -> dict:
        return {
            "start": [round(v, 2) for v in self.start],
            "end": [round(v, 2) for v in self.end],
            "length": round(self.length, 2),
            "angle_deg": round(self.angle_deg, 2),
        }


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def _round(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (np.floating, np.integer)):
        return round(float(value), 4)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_round(v) for v in value]
    return value


def polygon_area(points: np.ndarray) -> float:
    """Absolute shoelace area."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray, closed: bool = True) -> float:
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    total = float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))
    if closed and len(points) > 2:
        total += math.hypot(
            float(points[0][0] - points[-1][0]), float(points[0][1] - points[-1][1])
        )
    return total


def vertex_angles(points: np.ndarray) -> list[float]:
    """Interior angle at each vertex, in degrees."""
    n = len(points)
    if n < 3:
        return []
    angles: list[float] = []
    for i in range(n):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[(i + 1) % n]
        v1 = np.array([prev[0] - curr[0], prev[1] - curr[1]], dtype=np.float64)
        v2 = np.array([nxt[0] - curr[0], nxt[1] - curr[1]], dtype=np.float64)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            angles.append(180.0)
            continue
        cosine = float(np.dot(v1, v2) / (n1 * n2))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def min_area_rect(points: np.ndarray) -> tuple[float, float, float]:
    """``(area, width, height)`` of the rotated minimum-area rectangle."""
    cv2 = _import_cv2()
    rect = cv2.minAreaRect(points.astype(np.float32))
    (_, _), (w, h), _ = rect
    return float(w * h), float(w), float(h)


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    if not intersection:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _side_lengths(points: np.ndarray) -> list[float]:
    n = len(points)
    lengths = []
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        lengths.append(math.hypot(float(b[0] - a[0]), float(b[1] - a[1])))
    return lengths


def _opposite_sides_equal(lengths: Sequence[float], tolerance: float = 0.18) -> bool:
    """True when opposite sides have similar lengths (rectangle-like)."""
    if len(lengths) != 4:
        return False
    a, b, c, d = lengths
    return (
        abs(a - c) <= tolerance * max(a, c, 1.0)
        and abs(b - d) <= tolerance * max(b, d, 1.0)
        and min(lengths) > tolerance * max(lengths)
    )


def _opposite_sides_parallel_vectors(points: np.ndarray, tolerance: float = 0.20) -> bool:
    n = len(points)
    if n != 4:
        return False
    vectors = []
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        vectors.append(np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64))
    for i in (0, 1):
        v = vectors[i]
        w = vectors[i + 2]
        nv, nw = np.linalg.norm(v), np.linalg.norm(w)
        if nv == 0 or nw == 0:
            return False
        cosine = abs(float(np.dot(v, w)) / (nv * nw))
        if cosine < (1 - tolerance):
            return False
    return True


def _right_angles(angles: Sequence[float], tolerance: float = 22.0) -> bool:
    if len(angles) != 4:
        return False
    return all(abs(a - 90.0) <= tolerance for a in angles)


def _skew_angles(angles: Sequence[float], tolerance: float = 22.0) -> bool:
    """True for parallelogram/diamond corners: roughly 60 or 120 degrees.

    Note the disjunction: a corner near 60 must not also be required to be near
    120, which no angle can be.
    """
    if len(angles) != 4:
        return False
    return all(
        abs(a - 90.0) > tolerance and (abs(a - 60.0) <= tolerance or abs(a - 120.0) <= tolerance)
        for a in angles
    )


def _is_convex(points: np.ndarray) -> bool:
    n = len(points)
    if n < 4:
        return True
    signs = []
    for i in range(n):
        a, b, c = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-6:
            signs.append(cross > 0)
    return len(set(signs)) <= 1


def _corner_radius_measure(points: np.ndarray) -> float:
    """Fraction of vertices that are strongly rounded rather than sharp.

    Rounded rectangles survive approximation with extra vertices clustered at
    the corners; sharp rectangles do not.
    """
    if len(points) < 8:
        return 0.0
    lengths = _side_lengths(points)
    longest = max(lengths) if lengths else 0.0
    if longest <= 0:
        return 0.0
    short = [length for length in lengths if length < longest * 0.12]
    return len(short) / len(lengths)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


#: Fill ratio (contour area / bounding-box area) separates inscribed shapes
#: from drawn outlines. A triangle or diamond fills half its box; an ellipse
#: fills pi/4; a drawn rectangle fills essentially all of it.
FILL_DIAMOND_MAX = 0.62
FILL_ELLIPSE_MAX = 0.88
FILL_OUTLINE_MIN = 0.90
#: Two contours with this much overlap are the outer and inner edge of one
#: stroked outline, not two shapes.
NESTED_OUTLINE_IOU = 0.82
#: A contour with a corner sharper than this has ink meeting ink, which means it
#: wraps several things rather than drawing one of them.
DEGENERATE_ANGLE_MIN = 40.0


def classify_polygon(points: np.ndarray, area: float, perimeter: float) -> tuple[ShapeType, dict, float]:
    """Classify a polygon from its geometry.

    Returns ``(type, measurements, confidence)``. Confidence is a geometric
    agreement score, not a statistical probability.

    The tests are ordered from most distinctive to least, and each is chosen so
    that it cannot fire on the neighbouring class:

    * a diamond or triangle is *inscribed* in its bounding box;
    * an ellipse is inscribed and smoothly curved;
    * a drawn rectangle fills its box almost completely;
    * a parallelogram has a flat top but skewed corners;
    * a cylinder has flat top and bottom with shallow arcs between them.
    """
    n = len(points)
    angles = vertex_angles(points)
    lengths = _side_lengths(points)
    rect_area, rect_w, rect_h = min_area_rect(points)
    min_rect_fill = area / rect_area if rect_area else 0.0

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    box_w = max(1e-6, max(xs) - min(xs))
    box_h = max(1e-6, max(ys) - min(ys))
    fill = area / (box_w * box_h)
    circularity = (4 * math.pi * area) / (perimeter**2) if perimeter else 0.0
    convex = _is_convex(points)
    rounded = _corner_radius_measure(points)
    flat_top = _has_flat_top(points)
    flat_bottom = _has_flat_bottom(points)
    aspect = box_w / box_h

    measures = {
        "vertices": n,
        "fill_ratio": round(fill, 4),
        "min_rect_fill": round(min_rect_fill, 4),
        "circularity": round(circularity, 4),
        "convex": convex,
        "corner_rounding": round(rounded, 4),
        "flat_top": flat_top,
        "flat_bottom": flat_bottom,
        "aspect": round(aspect, 3),
        "angles": [round(a, 2) for a in angles],
        "side_lengths": [round(length, 2) for length in lengths],
        "min_rect": [round(rect_w, 2), round(rect_h, 2)],
    }

    # 0. Reject degenerate outlines before anything else. A contour that hugs a
    #    connected component of several boxes and its connectors develops spikes
    #    where ink meets ink, producing near-zero angles. Those are never a
    #    drawn primitive, whatever else they happen to satisfy.
    if n >= 3 and min(angles) < DEGENERATE_ANGLE_MIN:
        return ShapeType.UNKNOWN, measures, 0.2

    # 1. Smooth inscribed curve: ellipse or circle.
    if n >= 7 and convex and fill <= FILL_ELLIPSE_MAX and circularity >= 0.70:
        return ShapeType.ELLIPSE, measures, min(1.0, circularity)

    # 2. Small triangle: an arrowhead; large: a real shape.
    if n == 3 and convex:
        if area < 2500:
            return ShapeType.ARROWHEAD, measures, 0.6
        return ShapeType.TRIANGLE, measures, 0.75

    # 3. Inscribed quadrilateral with skewed corners: a diamond. A triangle also
    #    fills half its box, so the vertex count is what separates them.
    if n == 4 and convex and fill <= FILL_DIAMOND_MAX:
        return ShapeType.DIAMOND, measures, 0.86

    # 4. Drawn outline filling its box: a rectangle (possibly rounded).
    if n == 4 and convex and fill >= FILL_OUTLINE_MIN:
        if rounded > 0.12:
            return ShapeType.ROUNDED_RECTANGLE, measures, 0.72
        return ShapeType.RECTANGLE, measures, 0.92

    # 5. Cylinder: flat top and bottom with arcs between, so it fills less than
    #    a rectangle but is not inscribed either. Its corners stay near-square,
    #    which is what separates it from a parallelogram's skewed corners.
    if (
        n >= 4
        and convex
        and flat_top
        and flat_bottom
        and 0.68 <= fill <= FILL_OUTLINE_MIN
        and aspect >= 1.3
        and _right_angles(angles, tolerance=28.0)
    ):
        return ShapeType.CYLINDER, measures, 0.62

    # 6. Quadrilateral with a flat top and skewed corners: parallelogram.
    if n == 4 and convex and flat_top and fill >= FILL_DIAMOND_MAX:
        if _skew_angles(angles) or _skew_shift(points):
            return ShapeType.PARALLELOGRAM, measures, 0.76
        if _right_angles(angles):
            return ShapeType.RECTANGLE, measures, 0.7
        return ShapeType.PARALLELOGRAM, measures, 0.6

    # 7. Coarse approximation of a rounded rectangle.
    if n >= 5 and convex and fill >= 0.82:
        return ShapeType.ROUNDED_RECTANGLE, measures, 0.6

    return ShapeType.UNKNOWN, measures, 0.3


def _has_flat_bottom(points: np.ndarray, tolerance_ratio: float = 0.06) -> bool:
    """True when two or more vertices share the bottom edge of the shape."""
    ys = [float(p[1]) for p in points]
    bottom = max(ys)
    span = bottom - min(ys)
    if span <= 0:
        return False
    return sum(1 for y in ys if abs(y - bottom) <= span * tolerance_ratio) >= 2


def _has_flat_top(points: np.ndarray, tolerance_ratio: float = 0.06) -> bool:
    """True when two or more vertices share the top edge of the shape."""
    ys = [float(p[1]) for p in points]
    top = min(ys)
    span = max(ys) - top
    if span <= 0:
        return False
    return sum(1 for y in ys if abs(y - top) <= span * tolerance_ratio) >= 2


def _skew_shift(points: np.ndarray, tolerance: float = 0.12) -> bool:
    """True when the top edge is horizontally offset from the bottom edge."""
    if len(points) != 4:
        return False
    xs = [float(points[i][0]) for i in range(4)]
    ys = [float(points[i][1]) for i in range(4)]
    top = [xs[i] for i in range(4) if abs(ys[i] - min(ys)) < (max(ys) - min(ys)) * 0.2]
    bottom = [xs[i] for i in range(4) if abs(ys[i] - max(ys)) < (max(ys) - min(ys)) * 0.2]
    if len(top) != 2 or len(bottom) != 2:
        return False
    shift = abs(sum(top) / 2 - sum(bottom) / 2)
    width = max(xs) - min(xs)
    return width > 0 and shift / width > tolerance


def parallel_vectors_with_rotation(points: np.ndarray, tolerance: float = 0.88) -> bool:
    """True when a quad's opposite edges point in opposite directions (rotated)."""
    if len(points) != 4:
        return False
    vectors = []
    for i in range(4):
        a = points[i]
        b = points[(i + 1) % 4]
        vectors.append(np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64))
    for i in (0, 1):
        v, w = vectors[i], vectors[i + 2]
        nv, nw = np.linalg.norm(v), np.linalg.norm(w)
        if nv == 0 or nw == 0:
            return False
        if abs(float(np.dot(v, w)) / (nv * nw)) > tolerance:
            return False
    return True


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

#: Below this fraction of image area a contour is treated as noise, not a shape.
MIN_SHAPE_AREA_RATIO = 0.00035
#: A group frame is a drawn outline, so it fills its own bounding box. This is
#: what separates a subsystem box from a cluster of glyphs or from the outer
#: contour that wraps an entire connected drawing.
FRAME_MIN_FILL = 0.80
#: A frame's contents must occupy a real share of it. A box holding one caption
#: is a node, not a group.
FRAME_CHILD_AREA_RATIO = 0.12
#: A frame holds several things. One small child means a dashed stroke that
#: happened to close, not a subsystem boundary.
FRAME_MIN_CHILDREN = 2
#: A frame is drawn *around* its members, so it leaves visible slack. Without
#: this test, the outer contour of any connected cluster of boxes is published
#: as a node: structure that was never drawn.
FRAME_MIN_SLACK = 10
#: A drawn primitive fills its bounding box, except for inscribed shapes which
#: are whitelisted by type.
PRIMITIVE_MIN_FILL = 0.72
#: A contour nested inside a shape counts as a sub-shape only above this share
#: of the parent's area; below it, it is a glyph.
SUB_SHAPE_AREA_RATIO = 0.08


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - declared dependency
        from docweave.exceptions import DependencyMissingError

        raise DependencyMissingError("opencv-python", "diagram analysis", "diagrams") from exc
    return cv2


def binarize(image: np.ndarray) -> np.ndarray:
    """Foreground mask: dark strokes on a light ground become 255.

    Diagrams are line art on white, so Otsu on the inverted grayscale is both
    sufficient and completely deterministic.
    """
    cv2 = _import_cv2()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_shapes(
    image: np.ndarray,
    *,
    min_area_ratio: float = MIN_SHAPE_AREA_RATIO,
    id_prefix: str = "s",
) -> list[Shape]:
    """Find closed shape primitives in a diagram.

    Contour hierarchy matters here. In a line drawing where connectors touch the
    boxes they join, ``findContours`` returns one outer contour wrapping the
    whole connected component. Treating that as a shape would turn every
    diagram into a single blob, so the tree is walked: a contour that merely
    *encloses* its children without slack is a component and is discarded; a
    contour with real slack around its children is a group frame and is kept.

    Results are ordered top-down then left-to-right, so output is stable.
    """
    cv2 = _import_cv2()
    binary = binarize(image)
    height, width = binary.shape
    image_area = float(width * height)
    min_area = max(24.0, image_area * min_area_ratio)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return []

    children = _child_map(hierarchy)
    shapes: list[Shape] = []
    polygons: dict[str, np.ndarray] = {}

    for index in range(len(contours)):
        if int(hierarchy[0][index][3]) != -1:
            continue
        _walk_contour(
            index, contours, children, min_area, shapes, depth=0,
            id_prefix=id_prefix, polygons=polygons,
        )

    return _drop_wrapped_duplicates(
        _dedupe_overlapping(
            _drop_connector_enclosures(
                shapes + _dashed_frames(image, min_area, shapes, id_prefix), polygons
            )
        )
    )


def _dedupe_overlapping(shapes: list[Shape]) -> list[Shape]:
    """Drop a container that only wraps already-detected primitives."""
    kept: list[Shape] = []
    for shape in shapes:
        if shape.type is ShapeType.CONTAINER:
            x, y, w, h = shape.box
            inner = [
                s
                for s in shapes
                if s is not shape and s.type is not ShapeType.CONTAINER
                and _inside(shape.box, s.box, slack=2)
            ]
            if inner and max(s.box[2] * s.box[3] for s in inner) >= (w * h) * 0.8:
                # The contour walk already found a shape occupying this frame;
                # the dashed one is a duplicate reading of the same boundary.
                continue
        kept.append(shape)
    return kept


#: How much of a shape's own area must fall inside another shape's box before it
#: is treated as a second reading of the same drawing rather than a real thing.
WRAPPED_MIN_INSIDE = 0.88
#: ...and the two boxes must also be about the same size, so that a frame drawn
#: tightly around one node is not mistaken for that node.
WRAPPED_MIN_IOU = 0.80

#: A connector enclosure is a region bounded entirely by the connectors that
#: join boxes: where connectors meet the outlines they touch, the ink closes a
#: loop and ``findContours`` hands the enclosed blank back as a contour. It is
#: never drawn, yet it can fill its bounding box well enough to pass the coarse
#: primitive test. Its corners are exactly the points where a connector meets a
#: box outline, so each one lies on the outline of another shape. A coarse
#: polygon pinned to this many distinct shapes is read as an enclosure instead.
ENCLOSURE_MIN_SHAPES = 3
#: How close a corner must sit to another shape's outline to count as pinned.
ENCLOSURE_PIN_TOLERANCE = 4.0


def _distance_to_border(point, box: Box) -> float:
    """Distance from ``point`` to the outline of the rectangle ``box``."""
    px, py = point
    x, y, w, h = box
    if x <= px <= x + w and y <= py <= y + h:
        return min(px - x, x + w - px, py - y, y + h - py)
    if x <= px <= x + w:
        return min(abs(py - y), abs(py - (y + h)))
    if y <= py <= y + h:
        return min(abs(px - x), abs(px - (x + w)))
    cx = min(max(px, x), x + w)
    cy = min(max(py, y), y + h)
    return math.hypot(px - cx, py - cy)


def _shapes_pinned_at(points: np.ndarray, shape: Shape, shapes: Sequence[Shape]) -> set[str]:
    """Ids of other shapes whose outline a corner of ``shape`` sits on."""
    pinned: set[str] = set()
    for raw in points:
        point = (float(raw[0]), float(raw[1]))
        for other in shapes:
            if other is shape or other.type is ShapeType.CONTAINER:
                continue
            if _distance_to_border(point, other.box) <= ENCLOSURE_PIN_TOLERANCE:
                pinned.add(other.id)
                break
    return pinned


def _drop_connector_enclosures(
    shapes: list[Shape], polygons: dict[str, np.ndarray]
) -> list[Shape]:
    """Drop regions that are enclosed by connectors rather than drawn.

    Only coarse polygons are considered: a plain box, diamond, ellipse or
    triangle that happens to sit flush against its neighbours is a real shape,
    and dropping it would erase drawn structure. The enclosure produced by a
    ring of connectors instead arrives as the loose ``ROUNDED_RECTANGLE`` the
    classifier falls back to for an irregular many-sided blob.
    """
    kept: list[Shape] = []
    for shape in shapes:
        if (
            shape.type is ShapeType.ROUNDED_RECTANGLE
            and shape.measures.get("vertices", 0) >= 5
        ):
            points = polygons.get(shape.id)
            if points is not None and len(
                _shapes_pinned_at(points, shape, shapes)
            ) >= ENCLOSURE_MIN_SHAPES:
                continue
        kept.append(shape)
    return kept


def _drop_wrapped_duplicates(shapes: list[Shape]) -> list[Shape]:
    """Remove a contour that is the same drawing seen twice.

    A closed box can yield both its own contour and an outer one at almost the
    same size, and both pass the primitive test. Publishing both invents a node
    that was never drawn and splits its connectors between the two. Containers
    are exempt: a frame *should* enclose its members.
    """
    doomed: set[str] = set()
    for shape in shapes:
        if shape.type is ShapeType.CONTAINER:
            continue
        for other in shapes:
            if other is shape or other.type is ShapeType.CONTAINER:
                continue
            a, b = shape.box, other.box
            if box_iou(a, b) < WRAPPED_MIN_IOU:
                continue
            if _containment(a, b) >= WRAPPED_MIN_INSIDE:
                # `shape` sits inside `other`: `other` is the wrapper.
                doomed.add(other.id)
                break
            if _containment(b, a) >= WRAPPED_MIN_INSIDE:
                doomed.add(shape.id)
                break
    return [s for s in shapes if s.id not in doomed]


def _containment(inner, outer) -> float:
    """Fraction of ``inner``'s area that falls inside ``outer``."""
    x, y, w, h = inner
    ox, oy, ow, oh = outer
    ix0, iy0 = max(x, ox), max(y, oy)
    ix1, iy1 = min(x + w, ox + ow), min(y + h, oy + oh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area = float(w * h) or 1.0
    return (ix1 - ix0) * (iy1 - iy0) / area


def _dashed_frames(image, min_area: float, shapes: list, id_prefix: str) -> list:
    """Find rectangle frames drawn as dashed or broken strokes.

    A closed contour is the normal way shapes are found, and it cannot see a
    dashed rectangle: the gaps mean the outline never closes. The stroke is
    still four straight lines, so the frame is rebuilt from the line detector
    instead -- find pairs of parallel runs and join them into a rectangle.

    Only rectangles that *contain* something are kept. A pair of crossing lines
    that happens to form a box around nothing is decoration, not a boundary.
    """
    cv2 = _import_cv2()
    binary = binarize(image)
    height, width = binary.shape
    edges = cv2.Canny(binary, 60, 160, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=max(24, int(width * 0.06)),
        maxLineGap=14,
    )
    if lines is None:
        return []

    horizontal: list[tuple[float, float, float]] = []  # (y, x0, x1)
    vertical: list[tuple[float, float, float]] = []  # (x, y0, y1)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (float(v) for v in line)
        if abs(y2 - y1) <= abs(x2 - x1) * 0.2 and abs(x2 - x1) > 24:
            horizontal.append((round((y1 + y2) / 2, 1), min(x1, x2), max(x1, x2)))
        elif abs(x2 - x1) <= abs(y2 - y1) * 0.2 and abs(y2 - y1) > 24:
            vertical.append((round((x1 + x2) / 2, 1), min(y1, y2), max(y1, y2)))

    def merge(runs):
        """Join collinear runs so a dashed side becomes one span."""
        merged: list[list[float]] = []
        for position, low, high in sorted(runs):
            for run in merged:
                if abs(run[0] - position) <= 3 and low <= run[2] + 12 and high >= run[1] - 12:
                    run[1] = min(run[1], low)
                    run[2] = max(run[2], high)
                    break
            else:
                merged.append([position, low, high])
        return merged

    horizontals = merge(horizontal)
    verticals = merge(vertical)

    frames: list = []
    for top in horizontals:
        for bottom in horizontals:
            if bottom[0] - top[0] < 30:
                continue
            left_candidates = [
                x for x, _, _ in verticals if top[1] - 6 <= x <= top[2] + 6
            ]
            right_candidates = [
                x for x, _, _ in verticals if bottom[1] - 6 <= x <= bottom[2] + 6
            ]
            if not left_candidates or not right_candidates:
                continue
            # A side has to reach from the top run to the bottom run, and its x
            # must sit under both horizontal spans. Leftmost and rightmost of
            # those are the rectangle's two vertical sides -- picking the
            # extremes in the wrong order swaps them and rejects every frame.
            top_y, bottom_y = top[0], bottom[0]
            sides = [
                x
                for x, low, high in verticals
                if low <= top_y + 6
                and high >= bottom_y - 6
                and top[1] - 6 <= x <= top[2] + 6
                and bottom[1] - 6 <= x <= bottom[2] + 6
            ]
            if len(sides) < 2:
                continue
            left, right = min(sides), max(sides)
            if right - left < 30:
                continue

            box = (left, top_y, right - left, bottom_y - top_y)
            # The whole point is that the outline is *broken*. A solid
            # rectangle is already found by findContours, and accepting one here
            # duplicates every box in the drawing.
            sides_dashed = all(
                [
                    _stroke_is_dashed(
                        binary, (box[0], top_y), (box[0] + box[2], top_y)
                    ),
                    _stroke_is_dashed(
                        binary, (box[0], bottom_y), (box[0] + box[2], bottom_y)
                    ),
                    _stroke_is_dashed(binary, (left, box[1]), (left, box[1] + box[3])),
                    _stroke_is_dashed(
                        binary, (right, box[1]), (right, box[1] + box[3])
                    ),
                ]
            )
            if not sides_dashed:
                continue
            area = box[2] * box[3]
            if area < min_area:
                continue
            contained = [
                s
                for s in shapes
                if _inside(box, s.box, slack=2) and (s.box[2] * s.box[3]) < area * 0.6
            ]
            if len(contained) < 2:
                continue
            # A frame around something that is itself a frame is not a frame.
            if any(s.type is ShapeType.CONTAINER for s in contained):
                continue
            frames.append(
                (
                    area,
                    Shape(
                        id=f"{id_prefix}dash{len(frames)}",
                        type=ShapeType.CONTAINER,
                        box=box,
                        area=float(area),
                        perimeter=2 * (box[2] + box[3]),
                        confidence=0.72,
                        depth=0,
                        measures={
                            "bbox_fill": 0.99,
                            "dashed": 1.0,
                            "children": len(contained),
                        },
                    ),
                )
            )
    frames.sort(key=lambda item: item[0])
    return [shape for _area, shape in frames]


def _inside(outer, inner, slack: float = 0.0) -> bool:
    x, y, w, h = outer
    ix, iy, iw, ih = inner
    return (
        ix >= x - slack
        and iy >= y - slack
        and ix + iw <= x + w + slack
        and iy + ih <= y + h + slack
    )


def _child_map(hierarchy: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for index in range(len(hierarchy[0])):
        parent = int(hierarchy[0][index][3])
        mapping.setdefault(parent, []).append(index)
    return mapping


def _walk_contour(
    index: int,
    contours,
    children: dict[int, list[int]],
    min_area: float,
    shapes: list[Shape],
    depth: int,
    id_prefix: str,
    polygons: dict[str, np.ndarray] | None = None,
) -> None:
    cv2 = _import_cv2()
    contour = contours[index]
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    kids = children.get(index, [])

    if area < min_area or perimeter <= 0:
        for kid in kids:
            _walk_contour(kid, contours, children, min_area, shapes, depth, id_prefix, polygons)
        return

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
    if len(approx) < 3:
        return

    x, y, w, h = cv2.boundingRect(contour)
    box_area = float(w * h) or 1.0
    fill = area / box_area
    shape_type, measures, confidence = classify_polygon(approx, area, perimeter)
    measures["bbox_fill"] = fill

    child_area = sum(float(cv2.contourArea(contours[k])) for k in kids)
    substantial = [
        k for k in kids if float(cv2.contourArea(contours[k])) >= SUB_SHAPE_AREA_RATIO * area
    ]
# A frame holds real content: several things, with real slack around them.
    # A contour whose bounding box is barely larger than the union of its
    # children is not a drawn frame: it is the outer boundary of a connected
    # component, and publishing it as a node invents structure nobody drew.
    slack = float("inf")
    if substantial:
        child_boxes = [cv2.boundingRect(contours[k]) for k in substantial]
        cx0 = min(b[0] for b in child_boxes)
        cy0 = min(b[1] for b in child_boxes)
        cx1 = max(b[0] + b[2] for b in child_boxes)
        cy1 = max(b[1] + b[3] for b in child_boxes)
        slack = min(cx0 - x, x + w - cx1, cy0 - y, y + h - cy1)

    is_frame = (
        len(substantial) >= FRAME_MIN_CHILDREN
        and fill >= FRAME_MIN_FILL
        and child_area > FRAME_CHILD_AREA_RATIO * area
        and slack >= FRAME_MIN_SLACK
    )

    if is_frame:
        _emit(shapes, contour, area, ShapeType.CONTAINER, measures, max(confidence, 0.85), depth, id_prefix)
        for kid in substantial:
            _walk_contour(kid, contours, children, min_area, shapes, depth + 1, id_prefix, polygons)
        return

    if kids:
        # Descend first: a shape's children are either sub-shapes or glyphs.
        for kid in substantial:
            _walk_contour(kid, contours, children, min_area, shapes, depth + 1, id_prefix, polygons)

    if substantial:
        # This contour encloses other real shapes and is not a frame drawn
        # around them, so it is the outer boundary of a connected component --
        # two boxes and the line joining them, say. RETR_TREE hands that
        # boundary back as a contour, and it can classify as a diamond or a
        # sliver that passes the primitive test. Publishing it invents a node
        # nobody drew, and worse, the connectors get attributed to it instead of
        # to the shapes they actually join.
        return

    # Emit only contours that actually look like a drawn primitive. A closed
    # outline fills its bounding box; a cluster of glyphs does not.
    if not _is_primitive(shape_type, fill):
        return
    before = len(shapes)
    _emit(shapes, contour, area, shape_type, measures, confidence, depth, id_prefix)
    if polygons is not None and len(shapes) > before:
        polygons[shapes[-1].id] = approx


def _is_primitive(shape_type: ShapeType, fill: float) -> bool:
    if shape_type is ShapeType.UNKNOWN:
        return False
    if shape_type in (ShapeType.DIAMOND, ShapeType.TRIANGLE, ShapeType.ARROWHEAD, ShapeType.LINE):
        return True  # these are inscribed by construction; fill is expected < 1
    return fill >= PRIMITIVE_MIN_FILL


def _emit(
    shapes: list[Shape],
    contour,
    area: float,
    shape_type: ShapeType,
    measures: dict,
    confidence: float,
    depth: int,
    id_prefix: str,
) -> None:
    cv2 = _import_cv2()
    x, y, w, h = cv2.boundingRect(contour)
    candidate = Shape(
        id=f"{id_prefix}{len(shapes)}",
        type=shape_type,
        box=(int(x), int(y), int(w), int(h)),
        area=area,
        perimeter=float(cv2.arcLength(contour, True)),
        measures=measures,
        confidence=float(confidence),
        depth=depth,
    )
    if _is_nested_outline(candidate, shapes):
        return
    shapes.append(candidate)


def _is_nested_outline(candidate: Shape, existing: Sequence[Shape]) -> bool:
    """A stroked outline yields two nearly identical contours; keep the outer one.

    Without this every drawn box becomes two nodes, one inset by the stroke
    width, which silently doubles the node count of every diagram.
    """
    for other in existing:
        if other.type is ShapeType.CONTAINER:
            continue
        if box_iou(candidate.box, other.box) >= NESTED_OUTLINE_IOU:
            return True
    return False


def _depth_of(hierarchy: np.ndarray, index: int) -> int:
    """Contour nesting depth: 0 means top level."""
    depth = 0
    parent = int(hierarchy[0][index][3])
    while parent != -1 and depth < 12:
        depth += 1
        parent = int(hierarchy[0][parent][3])
    return depth


def connector_ink(
    image: np.ndarray,
    shapes: Sequence[Shape] = (),
    text_boxes: Sequence[Box] = (),
) -> np.ndarray:
    """Ink with text removed, leaving shape outlines and connectors.

    Text is erased outright because connectors never run through labels. Shape
    outlines are *not* erased: doing so would also erase the short connectors
    that terminate on those boxes. Outlines are removed afterwards, by asking
    whether a detected line lies along a box border.
    """
    cv2 = _import_cv2()
    ink = binarize(image)
    for box in text_boxes:
        x, y, w, h = box
        cv2.rectangle(ink, (x - 1, y - 1), (x + w + 1, y + h + 1), 0, thickness=cv2.FILLED)
    return ink


def _connector_min_length(shapes: Sequence[Shape], width: int, height: int, ratio: float) -> float:
    """Connector length floor, scaled to the shapes rather than the canvas.

    A fixed fraction of the image would reject the short hops that make up most
    of a flowchart while still accepting stray marks on a sparse canvas.
    """
    diagonal = math.hypot(width, height)
    ceiling = diagonal * ratio
    if not shapes:
        return max(18.0, ceiling)
    sizes = sorted(min(s.box[2], s.box[3]) for s in shapes if s.box[2] > 0 and s.box[3] > 0)
    if not sizes:
        return max(18.0, ceiling)
    median = sizes[len(sizes) // 2]
    return max(18.0, min(ceiling, median * 0.45))


def detect_segments(
    image: np.ndarray,
    shapes: Sequence[Shape],
    text_boxes: Sequence[Box] = (),
    *,
    min_length_ratio: float = 0.12,
) -> list[Segment]:
    """Find straight connector lines between shapes."""
    cv2 = _import_cv2()
    ink = connector_ink(image, shapes, text_boxes)
    height, width = ink.shape
    min_length = _connector_min_length(shapes, width, height, min_length_ratio)

    edges = cv2.Canny(ink, 60, 160, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(6, int(min_length * 0.6)),
        minLineLength=min_length,
        maxLineGap=int(min_length * 0.4),
    )
    if lines is None:
        return []

    segments: list[Segment] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = (float(v) for v in line)
        if math.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        segment = Segment(
            start=(x1, y1), end=(x2, y2), dashed=_stroke_is_dashed(ink, (x1, y1), (x2, y2))
        )
        if _is_outline_of(segment, shapes):
            continue
        segments.append(segment)

    merged = _dedupe_segments(segments, tolerance=max(6.0, min_length * 0.25))
    # The join gap must span an arrowhead: the head is drawn over the line, so
    # the Hough transform reliably splits a connector in two there.
    joined = _join_collinear(merged, gap=max(24.0, min_length * 0.9))
    # Re-measure after joining: two solid fragments joined across an arrowhead
    # are solid, and a dash pattern that survived as several fragments is still
    # dashed only if the joined run actually has gaps.
    for segment in joined:
        segment.dashed = _stroke_is_dashed(ink, segment.start, segment.end)
    return joined


def _stroke_is_dashed(ink, start: tuple[float, float], end: tuple[float, float]) -> bool:
    """True when the stroke along this line is broken into evenly spaced pieces.

    Sampling ink along the segment is enough to tell the two apart: a solid line
    has ink at (almost) every sample, a dashed one has repeated gaps of a
    consistent size. The test requires *multiple* gaps, so a line that merely
    crosses a gap in a shape outline is not mistaken for a dashed connector.
    """
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length < 8:
        return False
    steps = max(24, int(length))
    hits: list[bool] = []
    for i in range(steps + 1):
        t = i / steps
        x = int(round(start[0] + (end[0] - start[0]) * t))
        y = int(round(start[1] + (end[1] - start[1]) * t))
        if not (0 <= y < ink.shape[0] and 0 <= x < ink.shape[1]):
            return False
        window = ink[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        hits.append(bool(window.any()))
    # Deliberately ragged: this compares each sample with the next, so the two
    # sequences differ by exactly one. Written out so the intent is not read as
    # an oversight by the reader or the linter.
    gaps = sum(1 for a, b in zip(hits, hits[1:], strict=False) if a and not b)
    covered = sum(hits) / len(hits)
    return gaps >= 2 and covered < 0.94


#: Perpendicular distance within which two fragments still count as one
#: collinear run. A 2-3px stroke split by an arrowhead stays inside this; two
#: parallel hops running side by side sit further apart and must stay separate.
COLLINEAR_OFFSET = 4.0


def _perpendicular_offset(line: Segment, point: tuple[float, float]) -> float:
    """Distance from a point to the infinite line through a segment."""
    (x1, y1), (x2, y2) = line.start, line.end
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm == 0:
        return math.hypot(point[0] - x1, point[1] - y1)
    return abs(dy * (point[0] - x1) - dx * (point[1] - y1)) / norm


def _join_collinear(segments: Sequence[Segment], gap: float) -> list[Segment]:
    """Rejoin fragments of one connector that the arrowhead split apart."""
    if not segments:
        return []
    remaining = list(segments)
    joined: list[Segment] = []
    used = [False] * len(remaining)

    for i, base in enumerate(remaining):
        if used[i]:
            continue
        used[i] = True
        current = Segment(base.start, base.end, base.thickness, base.dashed)
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(remaining):
                if used[j]:
                    continue
                if current.dashed != other.dashed:
                    continue  # a dashed and a solid stroke are two different connectors
                if abs(_angle_delta(current.angle_deg, other.angle_deg)) > 6:
                    continue
                if _perpendicular_offset(current, other.midpoint) > COLLINEAR_OFFSET:
                    continue  # parallel, but a second connector rather than a fragment
                gaps = [
                    point_segment_distance(a, other.start, other.end)
                    for a in (current.start, current.end)
                ] + [
                    point_segment_distance(b, current.start, current.end)
                    for b in (other.start, other.end)
                ]
                if min(gaps) > gap:
                    continue
                current = Segment(*_extreme_points(current, other))
                used[j] = True
                changed = True
        joined.append(current)

    return sorted(joined, key=lambda s: (round(s.start[1]), round(s.start[0])))


def _extreme_points(a: Segment, b: Segment) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two points furthest apart across two collinear segments."""
    points = [a.start, a.end, b.start, b.end]
    best = (0.0, points[0], points[1])
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.hypot(
                points[i][0] - points[j][0], points[i][1] - points[j][1]
            )
            if distance > best[0]:
                best = (distance, points[i], points[j])
    return best[1], best[2]


def _is_outline_of(segment: Segment, shapes: Sequence[Shape], tolerance: float = 5.0) -> bool:
    """True when both endpoints sit on the same border of one shape.

    Connectors legitimately *terminate* on a border, so a single endpoint on a
    border is not evidence. Both on the same straight border is.
    """
    for shape in shapes:
        if shape.type in (ShapeType.ARROWHEAD, ShapeType.LINE):
            continue
        x, y, w, h = shape.box
        borders = (
            ((x, y), (x + w, y)),          # top
            ((x, y + h), (x + w, y + h)),  # bottom
            ((x, y), (x, y + h)),          # left
            ((x + w, y), (x + w, y + h)),  # right
        )
        for a, b in borders:
            if (
                point_segment_distance(segment.start, a, b) <= tolerance
                and point_segment_distance(segment.end, a, b) <= tolerance
            ):
                return True
    return False


def _dedupe_segments(segments: Sequence[Segment], tolerance: float) -> list[Segment]:
    """Merge overlapping collinear detections from the Hough transform."""
    kept: list[Segment] = []
    for candidate in sorted(segments, key=lambda s: (-s.length, s.start, s.end)):
        duplicate = False
        for existing in kept:
            if existing.dashed != candidate.dashed:
                continue  # a dashed and a solid stroke are two different connectors
            same_line = abs(_angle_delta(existing.angle_deg, candidate.angle_deg)) <= 8
            if not same_line:
                continue
            offset = point_segment_distance(existing.midpoint, candidate.start, candidate.end)
            if offset <= tolerance and point_segment_distance(
                candidate.midpoint, existing.start, existing.end
            ) <= tolerance:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda s: (round(s.start[1]), round(s.start[0])))


def _angle_delta(a: float, b: float) -> float:
    delta = abs((a - b) % 180)
    return min(delta, 180 - delta)


#: Stroke thickness above which ink counts as an arrowhead rather than a line.
#: A 2px connector line has a distance-transform peak around 1-2; a filled
#: triangular head reaches 4 or more.
ARROWHEAD_THICKNESS = 2.2
ARROWHEAD_MAX_BOX = 70


def detect_arrowheads(
    image: np.ndarray,
    shapes: Sequence[Shape] = (),
    segments: Sequence[Segment] = (),
    text_boxes: Sequence[Box] = (),
    *,
    thickness: float = ARROWHEAD_THICKNESS,
) -> list[Shape]:
    """Find arrowheads by local ink thickness.

    An arrowhead is not a separate contour: it is fused into the connector and
    the shape it touches, so contour analysis cannot see it. What does
    distinguish it is that it is *thick*. The distance transform gives ink depth
    directly, and thresholding it isolates the heads while erasing hairlines.

    Text is removed first — bold glyph strokes are thick enough to masquerade as
    heads — and a head must terminate a connector, which rules out the rest.
    """
    cv2 = _import_cv2()
    ink = connector_ink(image, shapes, text_boxes)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    thick = (distance >= thickness).astype(np.uint8) * 255

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(thick, 8)
    out: list[Shape] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 4 or w > ARROWHEAD_MAX_BOX or h > ARROWHEAD_MAX_BOX:
            continue
        if w < 3 or h < 3:
            continue
        aspect = w / h if h else 0.0
        if aspect > 3.0 or (1 / aspect if aspect else 0) > 3.0:
            continue
        cx, cy = (float(v) for v in centroids[index])

        if shapes and _deep_inside(cx, cy, shapes):
            continue
        if segments and not _near_any_endpoint(cx, cy, segments):
            # A thick blob that terminates no connector is ink we cannot explain.
            continue

        out.append(
            Shape(
                id=f"a{len(out)}",
                type=ShapeType.ARROWHEAD,
                box=(x, y, w, h),
                area=float(area),
                perimeter=0.0,
                measures={
                    "center": [round(cx, 2), round(cy, 2)],
                    "thickness": thickness,
                    "aspect": round(aspect, 3),
                },
                confidence=0.7,
            )
        )
    return out


def _deep_inside(cx: float, cy: float, shapes: Sequence[Shape], margin_ratio: float = 0.45) -> bool:
    """True when a point sits well inside a shape, not merely touching its edge."""
    for shape in shapes:
        if shape.type in (ShapeType.ARROWHEAD, ShapeType.LINE):
            continue
        x, y, w, h = shape.box
        inset_x, inset_y = w * margin_ratio, h * margin_ratio
        if x + inset_x <= cx <= x + w - inset_x and y + inset_y <= cy <= y + h - inset_y:
            return True
    return False


def _near_any_endpoint(cx: float, cy: float, segments: Sequence[Segment], tolerance: float = 24.0) -> bool:
    for segment in segments:
        for point in (segment.start, segment.end):
            if math.hypot(cx - point[0], cy - point[1]) <= tolerance:
                return True
        if point_segment_distance((cx, cy), segment.start, segment.end) <= tolerance:
            return True
    return False


def _box_points(box: Box) -> np.ndarray:
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


def find_enclosing(shape: Shape, candidates: Sequence[Shape]) -> Shape | None:
    """Smallest candidate fully containing ``shape`` (excluding itself)."""
    best: Shape | None = None
    best_area = float("inf")
    for candidate in candidates:
        if candidate.id == shape.id or candidate.type in (ShapeType.LINE, ShapeType.ARROWHEAD):
            continue
        cx, cy = candidate.center
        if not candidate.contains_point(cx, cy):
            continue
        if candidate.contains_point(*shape.center) and candidate.area < best_area:
            best = candidate
            best_area = candidate.area
    return best


def image_size(image: np.ndarray) -> tuple[int, int]:
    return int(image.shape[1]), int(image.shape[0])
