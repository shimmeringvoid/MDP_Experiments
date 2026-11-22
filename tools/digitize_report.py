"""Recover data from the figures of the 2014 report.

The original code and data are lost, but the figures embedded in the PDF are
not.  This script reads them back:

* the line charts (Figures 2 and 7) are digitised by locating the axis tick
  marks and the round markers of each series;
* the grid-world drawings (Figures 1 and 3) are decoded into layouts and
  policies by looking at the colour of each lattice point and at the
  direction of the arrow leaving it.

Usage::

    pdfimages -png report/CS7641_Project3_MDP_2014.pdf report/figures/fig
    python tools/digitize_report.py report/figures src/mdplab/data

The outputs are small CSV/text files that ship with the package, so nothing
else in the repository depends on this script or on Pillow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# ----------------------------------------------------------------------------
# Line charts
# ----------------------------------------------------------------------------


def load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def colour_masks(im: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return {
        "dark": im.sum(axis=2) < 150,
        "ink": im.sum(axis=2) < 520,  # anything clearly darker than paper, e.g. thin grey tick marks
        "blue": (b > 180) & (r < 90) & (g < 90),
        "red": (r > 180) & (g < 90) & (b < 90),
    }


def _clusters(idx: np.ndarray, gap: int = 2) -> np.ndarray:
    """Centres of runs of consecutive integers."""
    if idx.size == 0:
        return np.empty(0)
    breaks = np.flatnonzero(np.diff(idx) > gap)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, idx.size - 1]
    return np.array([(idx[s] + idx[e]) / 2.0 for s, e in zip(starts, ends)])


def find_axes(dark: np.ndarray, x_range: tuple[int, int]) -> tuple[int, int, int, int]:
    """Bounding box (left, right, top, bottom) of the axes frame inside a column range."""
    x0, x1 = x_range
    sub = dark[:, x0:x1]
    rows = np.flatnonzero(sub.sum(axis=1) > 0.6 * (x1 - x0) * 0.6)
    cols = np.flatnonzero(sub.sum(axis=0) > 0.6 * sub.shape[0] * 0.6)
    row_c, col_c = _clusters(rows, gap=3), _clusters(cols, gap=3)
    return int(round(col_c[0] + x0)), int(round(col_c[-1] + x0)), int(round(row_c[0])), int(round(row_c[-1]))


def find_ticks(ink: np.ndarray, frame, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel positions of the interior tick marks on the x and y axes.

    Matplotlib draws every tick on all four sides of the frame.  The top and
    right edges are read because curves and markers rarely touch them, and the
    result is checked for even spacing.
    """
    left, right, top, bottom = frame
    inset = int(round(1.5 * scale)) + 1
    margin = 4 * scale
    xt = _clusters(np.flatnonzero(ink[top + inset, left + margin : right - margin]) + left + margin, gap=scale)
    yt = _clusters(np.flatnonzero(ink[top + margin : bottom - margin, right - inset]) + top + margin, gap=scale)
    for name, t in (("x", xt), ("y", yt)):
        if t.size < 2 or np.abs(np.diff(t) - np.median(np.diff(t))).max() > 1.5 * scale:
            raise RuntimeError(f"{name} ticks are not evenly spaced: {t}")
    return xt, yt


def linear_map(pixels: np.ndarray, values: np.ndarray):
    slope, intercept = np.polyfit(pixels, values, 1)
    resid = np.abs(slope * pixels + intercept - values).max()
    return (lambda p: slope * np.asarray(p) + intercept), resid


def find_markers(mask: np.ndarray, frame, radius: float) -> np.ndarray:
    """Centres (x, y in pixels) of the round markers of one colour.

    A marker is a local maximum of the distance transform that is nearly as
    deep as the marker's radius.  The connecting lines are thinner than the
    markers, so they never reach that depth, even where two segments meet.
    """
    left, right, top, bottom = frame
    sub = mask[top : bottom + 1, left : right + 1]
    depth = ndimage.distance_transform_edt(sub)
    size = 2 * int(round(0.8 * radius)) + 1
    peaks = (depth == ndimage.maximum_filter(depth, size=size)) & (depth >= 0.85 * radius)
    labels, n = ndimage.label(peaks, structure=np.ones((3, 3)))
    if n == 0:
        return np.empty((0, 2))
    com = np.array(ndimage.center_of_mass(peaks, labels, np.arange(1, n + 1)))
    centres = np.c_[com[:, 1] + left, com[:, 0] + top]
    return centres[np.argsort(centres[:, 0])]


def digitize_panel(path, x_range, x_ticks, y_ticks, scale, radius, series=("blue", "red")):
    """Digitise one panel.

    ``x_ticks`` and ``y_ticks`` give (value, step) of the first tick mark
    strictly inside the frame, counted from the left and from the bottom.
    """
    im = load(path)
    masks = colour_masks(im)
    frame = find_axes(masks["dark"], x_range)
    xt, yt = find_ticks(masks["ink"], frame, scale)
    x_vals = x_ticks[0] + x_ticks[1] * np.arange(xt.size)
    y_vals = y_ticks[0] + y_ticks[1] * np.arange(yt.size)[::-1]  # pixel rows grow downwards
    fx, rx = linear_map(xt, x_vals)
    fy, ry = linear_map(yt, y_vals)
    out = {"frame": frame, "n_xticks": int(xt.size), "n_yticks": int(yt.size), "tick_residual": (float(rx), float(ry))}
    for name in series:
        c = find_markers(masks[name], frame, radius)
        out[name] = np.c_[fx(c[:, 0]), fy(c[:, 1])]
    return out


# ----------------------------------------------------------------------------
# Grid-world drawings
# ----------------------------------------------------------------------------

ARROWS = "↑→↓←"
_STEP = ((-1, 0), (0, 1), (1, 0), (0, -1))  # (d_row, d_col) for up, right, down, left


def decode_grid(
    path,
    rows: int,
    cols: int,
    arrow_len: tuple[float, float],
    head_len: float = 13.0,
    shaft_width: float = 3.6,
    head_width: float = 10.6,
):
    """Decode a grid-world figure into a layout and a policy.

    ``arrow_len`` gives the length in pixels of a horizontal and of a vertical
    arrow; ``head_len``, ``shaft_width`` and ``head_width`` describe the arrow
    glyph.  All were measured on the figures themselves.

    Returns ``(layout_lines, policy_lines, flagged)`` where ``flagged`` lists
    the cells whose arrow is drawn in red (the report's "poor decisions").

    Reading the arrows takes some care, because each one starts on its own
    node and nearly reaches the next.  Where the lattice is roomy, an arrow
    leaving a node is recognised by its thin shaft right next to the node, a
    place no incoming arrow reaches.  Where an arrow is almost as long as the
    lattice spacing (the rows of Figure 3), the ink between two nodes is
    instead compared with four templates: nothing, an arrow one way, an arrow
    the other way, or both.
    """
    im = load(path)
    H, W = im.shape[:2]
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    green = (g > 100) & (r < 60) & (b < 60)
    red = (r > 200) & (g < 60) & (b < 60)
    yellow = (r > 200) & (g > 200) & (b < 80)
    blue = (b - np.maximum(r, g)) > 12  # arrow fill and its darker outline

    def centroids(mask, min_area):
        lab, n = ndimage.label(ndimage.binary_opening(mask, structure=np.ones((5, 5))))
        if n == 0:
            return np.empty((0, 2))
        idx = np.arange(1, n + 1)
        area = ndimage.sum(mask, lab, idx)
        com = np.array(ndimage.center_of_mass(mask, lab, idx))
        return com[area >= min_area][:, ::-1]  # (x, y)

    # Lattice from the centres of the square and round markers.
    marks = np.vstack([centroids(green, 80), centroids(red, 80)])

    def fit(coord, n):
        lo, hi = coord.min(), coord.max()
        best = None
        for span in range(n - 1, 0, -1):  # the outermost markers need not sit on the edges
            d = (hi - lo) / span
            for first in range(0, n - span):
                origin = lo - first * d
                idx = np.rint((coord - origin) / d)
                err = np.abs(coord - (origin + idx * d)).max()
                if err < 0.2 * d and (best is None or err < best[0] - 1e-9):
                    best = (err, origin, d)
            if best is not None:
                break
        _, origin, d = best
        idx = np.rint((coord - origin) / d)
        d, origin = np.polyfit(idx, coord, 1)
        return origin, d

    x0, dx = fit(marks[:, 0], cols)
    y0, dy = fit(marks[:, 1], rows)

    # --- cell types -------------------------------------------------------
    kind = np.empty((rows, cols), dtype="<U1")
    for i in range(rows):
        for j in range(cols):
            cx, cy = x0 + j * dx, y0 + i * dy
            win = (slice(int(max(cy - 4, 0)), int(min(cy + 5, H))), slice(int(max(cx - 4, 0)), int(min(cx + 5, W))))
            if yellow[win].mean() > 0.5:
                kind[i, j] = "G"
            elif green[win].mean() > 0.5:
                kind[i, j] = "#"
            elif red[win].mean() > 0.8:  # a solid red disc; a flagged arrow's box is hollow
                kind[i, j] = "x"
            else:
                kind[i, j] = "."

    # --- arrows -----------------------------------------------------------
    def ink_width(mask, cx, cy, d_row, d_col, t, half):
        """Number of inked pixels across the arrow axis, ``t`` pixels from the node."""
        px, py = cx + d_col * t, cy + d_row * t
        offs = np.arange(-half, half + 1)
        xs = np.rint(px + offs * abs(d_row)).astype(int)
        ys = np.rint(py + offs * abs(d_col)).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        return int(mask[ys[ok], xs[ok]].sum())

    def glyph(t, length):
        """Width of an arrow of the given length, ``t`` pixels from its tail."""
        base = length - head_len
        return np.where(t < base, shaft_width, np.clip(head_width * (length - t) / head_len, 0.0, None))

    def is_free(i, j):
        return 0 <= i < rows and 0 <= j < cols and kind[i, j] == "."

    def shaft_test(i, j, d):
        """Is there a shaft leaving node (i, j) in direction d?  None if the test cannot be applied."""
        d_row, d_col = _STEP[d]
        spacing = dy if d_row else dx
        length = arrow_len[1] if d_row else arrow_len[0]
        lo = 0.16 * length
        hi = length - head_len - 1
        if is_free(i + d_row, j + d_col):
            hi = min(hi, spacing - length - 2)  # stay clear of an arrow coming the other way
        if hi - lo < 3:
            return None
        cx, cy = x0 + j * dx, y0 + i * dy
        half = max(2, int(round(shaft_width / 2)))
        hits = np.mean([ink_width(blue | red, cx, cy, d_row, d_col, t, half) > 0 for t in np.arange(lo, hi + 0.5)])
        return bool(hits >= 0.6)

    def segment_errors(i, j):
        """Fit of the ink between (i, j) and (i + 1, j) to: nothing, down only, up only, both."""
        cx, cy = x0 + j * dx, y0 + i * dy
        ts = np.arange(3.0, dy - 3.0 + 0.5)
        seen = np.array([ink_width(blue, cx, cy, 1, 0, t, int(head_width)) for t in ts])
        down, up = glyph(ts, arrow_len[1]), glyph(dy - ts, arrow_len[1])
        return {
            (False, False): float((seen**2).sum()),
            (True, False): float(((seen - down) ** 2).sum()),
            (False, True): float(((seen - up) ** 2).sum()),
            (True, True): float(((seen - np.maximum(down, up)) ** 2).sum()),
        }

    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    action = np.full((rows, cols), -1)
    roomy = dy - arrow_len[1] - 2 - 0.16 * arrow_len[1] >= 3  # can vertical shafts be read directly?
    for i in range(rows):
        for j in range(cols):
            if kind[i, j] != ".":
                continue
            tests = [shaft_test(i, j, d) for d in range(4)]
            if roomy:
                hits = [d for d in range(4) if tests[d]]
                action[i, j] = hits[0] if len(hits) == 1 else -1
            elif tests[RIGHT] or tests[LEFT]:
                action[i, j] = RIGHT if tests[RIGHT] else LEFT

    if not roomy:
        # Every free node has exactly one arrow.  Nodes without a horizontal one
        # point up or down; choose the assignment that best explains the ink in
        # each column (a small Viterbi pass per column).
        BIG = 1e12
        for j in range(cols):
            states_prev, cost_prev, back = None, None, []
            for i in range(rows):
                if kind[i, j] != ".":
                    cand = [None]
                elif action[i, j] >= 0:
                    cand = [action[i, j]]
                else:
                    cand = [UP, DOWN]
                unary = []
                for a in cand:
                    c = 0.0
                    if a in (UP, DOWN):
                        # towards a wall, pit or obstacle only the node's own shaft can be there
                        t_up, t_down = shaft_test(i, j, UP), shaft_test(i, j, DOWN)
                        if not is_free(i - 1, j) and t_up is not None:
                            c += 0.0 if (a == UP) == t_up else BIG
                        if not is_free(i + 1, j) and t_down is not None:
                            c += 0.0 if (a == DOWN) == t_down else BIG
                    unary.append(c)
                if states_prev is None:
                    cost, ptr = unary, [0] * len(cand)
                else:
                    both_free = is_free(i - 1, j) and is_free(i, j)
                    err = segment_errors(i - 1, j) if both_free else None
                    cost, ptr = [], []
                    for a, u in zip(cand, unary):
                        options = []
                        for k, a_prev in enumerate(states_prev):
                            pair = err[(a_prev == DOWN, a == UP)] if both_free else 0.0
                            options.append(cost_prev[k] + pair)
                        best = int(np.argmin(options))
                        cost.append(options[best] + u)
                        ptr.append(best)
                back.append((cand, ptr))
                states_prev, cost_prev = cand, cost
            k = int(np.argmin(cost_prev))
            for i in range(rows - 1, -1, -1):
                cand, ptr = back[i]
                if cand[k] is not None:
                    action[i, j] = cand[k]
                k = ptr[k]

    policy = np.full((rows, cols), "?", dtype="<U1")
    flagged = []
    for i in range(rows):
        for j in range(cols):
            if kind[i, j] != ".":
                policy[i, j] = kind[i, j]
                continue
            if action[i, j] < 0:
                continue
            policy[i, j] = ARROWS[action[i, j]]
            d_row, d_col = _STEP[action[i, j]]
            cx, cy = x0 + j * dx, y0 + i * dy
            length = arrow_len[1] if d_row else arrow_len[0]
            ts = np.arange(0.2 * length, 0.5 * length)
            n_red = sum(ink_width(red, cx, cy, d_row, d_col, t, 3) for t in ts)
            n_blue = sum(ink_width(blue, cx, cy, d_row, d_col, t, 3) for t in ts)
            if n_red > n_blue:
                flagged.append((i, j))
    layout = ["".join(row) for row in kind]
    return layout, ["".join(row) for row in policy], flagged


# ----------------------------------------------------------------------------


def main(fig_dir: str, out_dir: str) -> None:
    fig_dir, out_dir = Path(fig_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    # Figure 7: crazy elevator, value vs policy iteration (iterations | seconds).
    # Policy iteration needed 2-4 iterations, which is too close to the axis to
    # read off reliably, so only value iteration is taken from the left panel.
    floors = np.arange(3, 60)
    table = np.full((floors.size, 4), np.nan)
    p = digitize_panel(fig_dir / "fig-008.png", (0, 2650), (10, 10), (50, 50), scale=7, radius=12)
    found = np.rint(p["blue"][:, 0]).astype(int)
    assert (found == floors).all(), "figure 7: unexpected floor numbers"
    table[:, 0] = p["blue"][:, 1]
    p = digitize_panel(fig_dir / "fig-008.png", (2650, 5234), (10, 10), (0.5, 0.5), scale=7, radius=12)
    for col, name in ((1, "blue"), (2, "red")):
        idx = np.rint(p[name][:, 0]).astype(int) - 3
        table[idx, col] = p[name][:, 1]
    np.savetxt(
        out_dir / "report_fig7_elevator.csv",
        np.c_[floors, table[:, :3]],
        delimiter=",",
        header="floors,vi_iterations,vi_seconds,pi_seconds",
        comments="",
        fmt=["%d", "%.2f", "%.4f", "%.4f"],
    )
    summary["report_fig7_elevator.csv"] = {"rows": int(floors.size), "vi_seconds_read": int(np.isfinite(table[:, 1]).sum())}

    # Figure 2: grid world, value vs policy iteration.  fig-002 shows worlds up
    # to 210 states, fig-003 all sizes.  The sizes are 3x3, 3x4, 4x4, ..., 38x39,
    # so every marker is snapped to the nearest of those.  Where both figures
    # show a size, the small-range figure is the more precise one.
    sizes = np.array(sorted({n * n for n in range(3, 39)} | {n * (n + 1) for n in range(3, 39)}))

    def snap(points):
        out = {}
        for x, y in points:
            k = int(np.abs(sizes - x).argmin())
            if abs(sizes[k] - x) <= max(3.0, 0.012 * sizes[k]):
                out.setdefault(int(sizes[k]), []).append(y)
        return {k: float(np.mean(v)) for k, v in out.items()}

    merged = {}
    for fig, xt, yt_it, yt_s, split in (
        ("fig-003.png", (200, 200), (200, 200), (1000, 1000), 2770),
        ("fig-002.png", (50, 50), (100, 100), (20, 20), 2740),
    ):
        im_w = load(fig_dir / fig).shape[1]
        a = digitize_panel(fig_dir / fig, (0, split), xt, yt_it, scale=7, radius=12)
        b = digitize_panel(fig_dir / fig, (split, im_w), xt, yt_s, scale=7, radius=12)
        for key, pts in (("vi_iterations", a["blue"]), ("pi_iterations", a["red"]), ("vi_seconds", b["blue"]), ("pi_seconds", b["red"])):
            if fig == "fig-003.png":
                # below 210 states the markers of the full-range figure overlap; leave those to fig-002
                pts = pts[pts[:, 0] > 215]
            merged.setdefault(key, {}).update(snap(pts))
    rows = []
    for size in sizes:
        rows.append([size] + [merged[k].get(int(size), np.nan) for k in ("vi_iterations", "pi_iterations", "vi_seconds", "pi_seconds")])
    rows = np.array(rows)
    np.savetxt(
        out_dir / "report_fig2_gridworld.csv",
        rows,
        delimiter=",",
        header="states,vi_iterations,pi_iterations,vi_seconds,pi_seconds",
        comments="",
        fmt=["%d", "%.1f", "%.1f", "%.2f", "%.2f"],
    )
    summary["report_fig2_gridworld.csv"] = {
        "rows": int(len(rows)),
        "read": {k: int(np.isfinite(rows[:, i + 1]).sum()) for i, k in enumerate(("vi_iterations", "pi_iterations", "vi_seconds", "pi_seconds"))},
    }

    # Figure 5: two bar charts of elevator transition probabilities (50 floors).
    # Bars are yellow, 0.8 wide, starting at the floor number.
    im5 = load(fig_dir / "fig-006.png")
    masks5 = colour_masks(im5)
    yellow = (im5[..., 0] > 200) & (im5[..., 1] > 200) & (im5[..., 2] < 90)
    columns = []
    # Both axes run exactly from tick to tick (0-50 floors; 0-0.040 and 0-0.025),
    # and the bars hide the tick marks, so the frame itself is the calibration.
    for x_range, y_max in (((0, 2600), 0.040), ((2600, im5.shape[1]), 0.025)):
        left, right, top, bottom = find_axes(masks5["dark"], x_range)
        heights = []
        for floor in range(50):
            col = int(round(left + (floor + 0.4) / 50 * (right - left)))
            rows_y = np.flatnonzero(yellow[top:bottom, col - 3 : col + 4].any(axis=1))
            heights.append(y_max * (bottom - (top + rows_y.min())) / (bottom - top) if rows_y.size else 0.0)
        columns.append(heights)
    np.savetxt(
        out_dir / "report_fig5_elevator.csv",
        np.c_[np.arange(50), np.array(columns).T],
        delimiter=",",
        header="floor,p_from10_button5,p_from5_button10",
        comments="",
        fmt=["%d", "%.5f", "%.5f"],
    )
    summary["report_fig5_elevator.csv"] = {"sum_left": float(np.sum(columns[0])), "sum_right": float(np.sum(columns[1]))}

    # Figures 4 and 8 are small bitmaps with two colours of marker that often
    # overlap (red is drawn on top of blue).  The x positions are known, so the
    # markers are assigned to them and whatever is hidden is left empty.
    def read_small(fig, x_first, x_step, y_first, y_step, xs_expected, per_x, window):
        im = load(fig_dir / fig)
        masks = colour_masks(im)
        frame = find_axes(masks["dark"], (0, im.shape[1]))
        left, right, top, bottom = frame
        xt = _clusters(np.flatnonzero(masks["ink"][top + 2, left + 4 : right - 4]) + left + 4, gap=1)
        yt = _clusters(np.flatnonzero(masks["ink"][top + 4 : bottom - 4, right - 2]) + top + 4, gap=1)
        fx, _ = linear_map(xt, x_first + x_step * np.arange(xt.size))
        fy, _ = linear_map(yt, y_first + y_step * np.arange(yt.size)[::-1])
        out = {}
        for name in ("blue", "red"):
            c = find_markers(masks[name], frame, radius=3.2)
            vals = np.full((len(xs_expected), per_x), np.nan)
            xs, ys = fx(c[:, 0]), fy(c[:, 1])
            for k, x in enumerate(xs_expected):
                near = np.sort(ys[np.abs(xs - x) < window])[::-1]
                if near.size >= per_x:  # with fewer markers visible, one cannot tell which curve is which
                    vals[k] = near[:per_x]
            out[name] = vals
        return out

    floors8 = np.arange(3, 60)
    f8 = read_small("fig-009.png", 10, 10, 50, 50, floors8, per_x=2, window=0.4)
    np.savetxt(
        out_dir / "report_fig8_elevator.csv",
        np.c_[floors8, f8["blue"], f8["red"]],
        delimiter=",",
        header="floors,vi_reward,vi_utility,q_reward,q_utility",
        comments="",
        fmt=["%d", "%.1f", "%.1f", "%.1f", "%.1f"],
    )
    summary["report_fig8_elevator.csv"] = {k: int(np.isfinite(v).sum()) for k, v in f8.items()}

    states4 = np.array([9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64, 72, 81, 90])
    f4 = read_small("fig-005.png", 10, 10, 10, 10, states4, per_x=1, window=1.2)
    np.savetxt(
        out_dir / "report_fig4_gridworld.csv",
        np.c_[states4, f4["blue"], f4["red"]],
        delimiter=",",
        header="states,vi_mean_reward,q_mean_reward",
        comments="",
        fmt=["%d", "%.1f", "%.1f"],
    )
    summary["report_fig4_gridworld.csv"] = {k: int(np.isfinite(v).sum()) for k, v in f4.items()}

    # Figures 1a / 1b: the 7 x 8 world and the two policies
    lay_a, pol_a, _ = decode_grid(fig_dir / "fig-001.png", 7, 8, arrow_len=(56, 55), head_len=22)
    lay_b, pol_b, flagged = decode_grid(fig_dir / "fig-000.png", 7, 8, arrow_len=(56, 55), head_len=22)
    assert lay_a == lay_b, "figures 1a and 1b should show the same world"
    (out_dir / "report_fig1_layout.txt").write_text("\n".join(lay_a) + "\n")
    (out_dir / "report_fig1a_policy.txt").write_text("\n".join(pol_a) + "\n")
    (out_dir / "report_fig1b_policy.txt").write_text("\n".join(pol_b) + "\n")
    summary["fig1_flagged_cells"] = flagged

    # Figure 3: the 39 x 39 world and its optimal policy
    lay, pol, _ = decode_grid(fig_dir / "fig-004.png", 39, 39, arrow_len=(30, 24))
    (out_dir / "report_fig3_layout.txt").write_text("\n".join(lay) + "\n")
    (out_dir / "report_fig3_policy.txt").write_text("\n".join(pol) + "\n")
    summary["fig3_counts"] = {ch: sum(row.count(ch) for row in lay) for ch in ".#xG"}
    summary["fig3_unreadable_arrows"] = sum(row.count("?") for row in pol)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
