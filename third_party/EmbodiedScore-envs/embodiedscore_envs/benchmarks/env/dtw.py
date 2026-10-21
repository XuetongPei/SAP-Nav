"""FastDTW (Salvador & Chan 2007) as the habitat-lab 0.1.7 / VLN-CE boards
computed it — a transcription of ``fastdtw`` 0.3.4's *Cython* build
(``_fastdtw.pyx``), not of its pure-Python fallback, in numba.

The two builds of that package are not the same algorithm: the Cython one
breaks cost ties corner-last (``up`` only when strictly smallest, then
``left`` when strictly below ``corner``, else ``corner``) and never takes the
diagonal into a cell whose previous row ends one short of it; the pure-Python
one takes the first minimum in (up, left, corner) order and allows that
diagonal. The boards were scored with the Cython build, so its quirks are
kept here on purpose — nDTW differs by ~1e-4 between the two.

Only what nDTW needs is transcribed: ``radius=1``-style expansion (``radius``
is a parameter), euclidean point distance in float64 (habitat's
``euclidean_distance`` on the location lists), the distance and the path.

``fastdtw(x, y, radius=1)`` -> ``(distance, path)`` mirrors the package's
signature so callers read the same. ``x``/``y`` are sequences of points
(``(n, d)``); 1-D input is treated as points of dimension 1.
"""

from __future__ import annotations

import math

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

_INF = math.inf


@intrinsic
def _fma(typingctx, a, b, c):
    """a * b + c with one rounding (LLVM ``llvm.fma``). numpy's ``dot`` on a
    2- or 3-vector — what ``np.linalg.norm(v, ord=2)`` squares — is an FMA
    chain in OpenBLAS's small-n kernel, so the squared norm is accumulated the
    same way here to land on the same bits."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        dbl = ir.DoubleType()
        fn = builder.module.declare_intrinsic("llvm.fma", [dbl], ir.FunctionType(dbl, [dbl, dbl, dbl]))
        return builder.call(fn, args)

    return sig, codegen


@njit(cache=True)
def _windowed_dtw(x, y, low, high):
    """DTW over the cells ``(i, j)`` with ``low[i] <= j <= high[i]``, rows in
    order; ``low``/``high`` are non-decreasing (a monotone path's band).
    Returns the cost and the path as two index arrays."""
    len_x = x.shape[0]
    dim = x.shape[1]
    off = np.empty(len_x + 1, np.int64)
    off[0] = 0
    for i in range(len_x):
        off[i + 1] = off[i] + (high[i] - low[i] + 1)
    n = off[len_x]
    wx = np.empty(n, np.int64)
    wy = np.empty(n, np.int64)
    cost = np.empty(n + 1, np.float64)
    prev = np.empty(n + 1, np.int64)
    cost[0] = 0.0
    prev[0] = -1
    idx = 0
    for xi in range(len_x):
        lo = low[xi]
        hi = high[xi]
        for yi in range(lo, hi + 1):
            s = 0.0
            for k in range(dim):
                d = y[yi, k] - x[xi, k]
                s = _fma(d, d, s)
            dt = math.sqrt(s)          # == np.linalg.norm(y[yi] - x[xi], ord=2)
            if xi == 0 and yi == 0:
                left = -1
                up = -1
                corner = 0
            else:
                left = -1 if yi == lo else idx
                up = -1 if xi == 0 or yi > high[xi - 1] else off[xi - 1] + (yi - low[xi - 1]) + 1
                # the Cython build's rule, quirk included: no diagonal into a cell one past
                # the previous row's end
                corner = -1 if xi == 0 or yi < low[xi - 1] + 1 or yi > high[xi - 1] else up - 1
            d_left = cost[left] if left != -1 else _INF
            d_up = cost[up] if up != -1 else _INF
            d_corner = cost[corner] if corner != -1 else _INF
            if d_up < d_left and d_up < d_corner:
                cost[idx + 1] = d_up + dt
                prev[idx + 1] = up
            elif d_left < d_corner:
                cost[idx + 1] = d_left + dt
                prev[idx + 1] = left
            else:
                cost[idx + 1] = d_corner + dt
                prev[idx + 1] = corner
            wx[idx] = xi
            wy[idx] = yi
            idx += 1
    # trace back from the last cell to the virtual start
    m = 0
    c = n
    while c != 0:
        m += 1
        c = prev[c]
    px = np.empty(m, np.int64)
    py = np.empty(m, np.int64)
    c = n
    k = m - 1
    while c != 0:
        px[k] = wx[c - 1]
        py[k] = wy[c - 1]
        k -= 1
        c = prev[c]
    return cost[n], px, py


@njit(cache=True)
def _expand_window(px, py, len_x, len_y, radius):
    """The band around a coarse path, widened by ``radius`` and doubled to the
    fine grid: per fine row ``i`` the columns ``low[i] .. high[i]``."""
    max_x = px[px.shape[0] - 1]
    max_y = py[py.shape[0] - 1]
    low1 = np.empty(max_x + 1, np.int64)
    high1 = np.empty(max_x + 1, np.int64)
    cur_x = -1
    prv_y = 0
    for t in range(px.shape[0]):
        xi = px[t]
        yi = py[t]
        if xi != cur_x:
            cur_x = xi
            low1[xi] = yi
            if cur_x > 0:
                high1[xi - 1] = prv_y
        prv_y = yi
    high1[max_x] = prv_y
    n2 = max_x + radius + 1
    low2 = np.empty(n2, np.int64)
    high2 = np.empty(n2, np.int64)
    for xi in range(max_x + 1):
        a = max(0, xi - radius)
        low2[xi] = max(0, low1[a] - radius)
        b = min(max_x, xi + radius)
        high2[xi] = min(max_y + radius, high1[b] + radius)
    for xi in range(max_x + 1, n2):
        low2[xi] = low2[xi - 1]
        high2[xi] = high2[xi - 1]
    n3 = min(len_x, 2 * n2)
    low = np.empty(n3, np.int64)
    high = np.empty(n3, np.int64)
    for xi in range(n2):
        for r in (2 * xi, 2 * xi + 1):
            if r < len_x:
                low[r] = 2 * low2[xi]
                high[r] = min(len_y - 1, 2 * high2[xi] + 1)
    return low, high


def _as_points(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("x and y must be sequences of points")
    return np.ascontiguousarray(arr)


def _reduce_by_half(a: np.ndarray) -> np.ndarray:
    mx = a.shape[0] - a.shape[0] % 2
    return np.ascontiguousarray((a[:mx:2] + a[1:mx:2]) / 2)


def fastdtw(x, y, radius: int = 1) -> tuple[float, list[tuple[int, int]]]:
    """Approximate DTW distance between two point sequences and the warping
    path, exactly as ``fastdtw.fastdtw(x, y, radius, dist=euclidean)`` of the
    package's Cython build computes them."""
    x = _as_points(x)
    y = _as_points(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("second dimension of x and y must be the same")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    min_size = radius + 2
    levels = [(x, y)]
    while levels[-1][0].shape[0] >= min_size and levels[-1][1].shape[0] >= min_size:
        cx, cy = levels[-1]
        levels.append((_reduce_by_half(cx), _reduce_by_half(cy)))
    bx, by = levels[-1]
    low = np.zeros(bx.shape[0], np.int64)
    high = np.full(bx.shape[0], by.shape[0] - 1, np.int64)
    cost, px, py = _windowed_dtw(bx, by, low, high)
    for cx, cy in reversed(levels[:-1]):
        low, high = _expand_window(px, py, cx.shape[0], cy.shape[0], radius)
        cost, px, py = _windowed_dtw(cx, cy, low, high)
    return float(cost), list(zip(px.tolist(), py.tolist(), strict=True))


def dtw(x, y) -> tuple[float, list[tuple[int, int]]]:
    """Exact DTW (the package's ``dtw``): every cell is in the window."""
    x = _as_points(x)
    y = _as_points(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("second dimension of x and y must be the same")
    low = np.zeros(x.shape[0], np.int64)
    high = np.full(x.shape[0], y.shape[0] - 1, np.int64)
    cost, px, py = _windowed_dtw(x, y, low, high)
    return float(cost), list(zip(px.tolist(), py.tolist(), strict=True))
