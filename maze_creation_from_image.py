"""
Pipeline that extracts maze-style traversal paths from black pixels in an image, builds per-zone
coverage routes, links zones together, and exports plots plus serialized data for downstream use.

Dependencies:
- __future__.annotations for forward references; numpy for array math; PIL.Image for RGBA loading and
  resizing; matplotlib.pyplot for visualization; collections.deque for BFS/DFS queues; random for
  shuffling; pathlib.Path for file handling; math.atan2 for angular sorting; json and time for dumps
  and progress tracking.

Data conventions:
- mask: boolean numpy array (H, W) where True marks a black/valid cell.
- comps/comp_masks: list of components as (x, y) tuples and matching boolean masks.
- order: list of component indices (nearest-neighbor or radial) that defines traversal order.
- zone_start_end: per-zone dicts with "start"/"end" boundary cells derived from bridges.
- bridges/bridges_linear: boundary-to-boundary connector paths between successive zones; last bridge
  closes the loop and defines global_start/global_end.
- zone_paths: per-zone self-avoiding paths; full_path: concatenation of zone paths and bridges with
  duplicates removed.

Functions and their key variables:
- load_black_mask(path, target_width, alpha_thresh, rgb_thresh): loads and rescales an image, splits
  RGBA channels (arr_r/g/b/a), computes luminance (lum), and returns a boolean mask of dark pixels.
- find_components(mask): BFS flood-fill over mask using dirs4; tracks seen set, builds comps (list of
  (x, y)) and comp_masks (boolean grids).
- plot_components(comp_masks, prefix): writes one PNG per component; ensures parent dir via Path.
- centroid(comp): mean of x and y coordinates for ordering heuristics.
- order_zones_nearest(comps): greedy nearest-neighbor ordering starting at the minimal x+y centroid;
  uses remaining/current indices to build order.
- order_zones_radial(comps): sorts component indices by angle around the global centroid (atan2).
- plot_order(mask, comps, order, filename): overlays centroids, order numbers, and dashed connections
  on the mask.
- boundary_points(comp, mask): perimeter cells whose 4-neighborhood leaves the mask.
- shortest_bridge(boundary_a, boundary_b, mask): BFS on background (walkable) cells with boundary
  cells temporarily walkable; uses parent map for reconstruction, with Manhattan fallback when no
  path is found. Key vars: walkable, targets, parent, found.
- compute_links(mask, comps, order): assembles bridges and zone_start_end by connecting ordered zones
  in a loop via boundary_points and shortest_bridge; sets each zone's start/end from bridge ends.
- plot_links(mask, comps, order, bridges, filename): plots bridges with start/end markers and zone
  labels for debugging.
- spanning_tree_from_start(comp, start): randomized DFS to build an undirected adjacency tree over a
  component; uses dirs4 shuffle, visited set, and tree dict keyed by coordinates.
- euler_walk_to(tree, start, end, allowed=None): traverses every tree edge (used set) starting from
  start, then BFSes to end within allowed cells; stack tracks current path; parent map rebuilds the
  final leg.
- build_zone_paths(mask, comps, order, zone_start_end): randomized backtracking per ordered zone to
  maximize coverage from its start to end within a time limit. Builds adj dict, visited/path sets,
  best_path/best_cov metrics, deadline/last_report timers, reachable() prune, and dfs() recursion
  until target coverage or timeout; returns paths and misses (unvisited counts).
- plot_zone_paths(mask, zone_paths, order, filename_prefix, must_visits): per-zone overlays with
  start/end markers and optional must-visit cells.
- plot_zone_paths_combined(mask, zone_paths, order, filename): combined overlay of all zones with
  labeled start/end tags.
- plot_zone_zoom(mask, zone_paths, order, idx, filename): zoomed-in plot of a specific zone path.
- build_full_path(zone_paths, bridges_linear, global_start, global_end): concatenates zone paths and
  bridges, normalizes coordinates via norm helper, strips duplicates/loops to produce a simple path.
- write_path_maze(path_seq, filename): writes start/end and adjacency for each cell in plain text
  format expected by maze consumers.
- plot_assembly(mask, zone_paths, order, bridges, global_start, global_end, filename): full overlay
  of zone paths, bridges, and global start/end markers.
- main(): CLI entry point; loads logo file/width/mode from argv, builds mask/components/order/bridges,
  computes per-zone paths and coverage stats, saves plots/npz/json summaries, and writes maze_path.txt
  plus zones_links.json for inspection.

Key state variables across the pipeline:
- mask/comp_masks: boolean grids; comps/ordered: lists of coordinate tuples; order: traversal order;
  zone_start_end: per-zone entry/exit mapping; bridges/bridges_linear: inter-zone connectors;
  zone_paths: per-zone traversal; misses: counts of unvisited cells; global_start/global_end/full_path:
  concatenated output; dirs4: reused 4-neighbor offsets; deadline/time_limit/last_report: timers that
  guard per-zone DFS and status printing.
"""

from __future__ import annotations

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import random
from pathlib import Path
from math import atan2
import json
import time
#region: cover_path integration
from cover_path import strategy_random_backtrack, recursive_fill
from curl_cover_path import strategy_random_backtrack as curl_strategy_random_backtrack
#endregion


#region: chargement et masque noir

def load_black_mask(path: str, target_width: int = 500, alpha_thresh: int = 10, rgb_thresh: int = 128):
    """Load an RGBA image, resize it, and return a boolean mask of dark pixels.

    Parameters control the target output width (height is scaled), alpha cutoff
    (alpha_thresh), and RGB luminance threshold (rgb_thresh). Splits channels
    into arr_r/g/b/a, rescales each with bilinear filtering, computes mean
    luminance (lum), and marks pixels with sufficient alpha and low luminance
    as True. The resulting mask is vertically flipped (np.flipud) so y=0
    matches a bottom-left origin. Used by main() to derive the maze zone mask;
    returns a numpy boolean array shaped (H, W). No external state is mutated.
    """
    img = Image.open(path).convert("RGBA")
    r, g, b, a = img.split()
    arr_r = np.array(r, dtype=np.uint8)
    arr_g = np.array(g, dtype=np.uint8)
    arr_b = np.array(b, dtype=np.uint8)
    arr_a = np.array(a, dtype=np.uint8)

    h, w = arr_a.shape
    new_w = target_width
    new_h = int(h * new_w / w)
    arr_r = np.array(Image.fromarray(arr_r).resize((new_w, new_h), Image.BILINEAR))
    arr_g = np.array(Image.fromarray(arr_g).resize((new_w, new_h), Image.BILINEAR))
    arr_b = np.array(Image.fromarray(arr_b).resize((new_w, new_h), Image.BILINEAR))
    arr_a = np.array(Image.fromarray(arr_a).resize((new_w, new_h), Image.BILINEAR))

    lum = (arr_r.astype(np.int32) + arr_g.astype(np.int32) + arr_b.astype(np.int32)) // 3
    mask = (arr_a > alpha_thresh) & (lum < rgb_thresh)
    mask = np.flipud(mask)
    return mask
#endregion

#region: composants connexes

def find_components(mask: np.ndarray):
    """Extract 4-connected components from a mask via BFS flood fill.

    Iterates over all True cells not yet seen, explores neighbors in dirs4
    using a deque queue, and collects both coordinate lists (comps) and per-
    component boolean masks (comp_masks). Returns (comps, comp_masks). Used by
    main() after mask creation. Mutates local `seen`; leaves input mask intact.
    """
    H, W = mask.shape
    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    comp_masks = []
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        if seen[y, x]:
            continue
        comp = []
        q = deque([(x, y)])
        seen[y, x] = True
        while q:
            cx, cy = q.popleft()
            comp.append((cx, cy))
            for dx, dy in dirs4:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((nx, ny))
        comps.append(comp)
        comp_mask = np.zeros_like(mask, dtype=bool)
        for cx, cy in comp:
            comp_mask[cy, cx] = True
        comp_masks.append(comp_mask)
    return comps, comp_masks
#endregion

#region: visualisation

def plot_components(comp_masks, prefix="zone"):
    """Save one PNG per component mask with a simple grayscale visualization.

    Iterates comp_masks, plots each with imshow(origin lower), titles the size,
    and saves as f\"{prefix}_{i}.png\". Ensures parent directory exists via
    Path(prefix).parent.mkdir. Used in main() for inspection. No return.
    """
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for i, cmask in enumerate(comp_masks, 1):
        plt.figure(figsize=(6, 4))
        plt.imshow(cmask, cmap="gray_r", origin="lower")
        plt.title(f"Zone {i} ({cmask.sum()} cells)")
        plt.axis("equal")
        plt.savefig(f"{prefix}_{i}.png", dpi=150)
    plt.close()
#endregion

#region: ordonnancement des zones

def centroid(comp):
    """Compute the centroid (mean x, mean y) of a component of (x, y) tuples."""
    xs = [p[0] for p in comp]
    ys = [p[1] for p in comp]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def order_zones_nearest(comps):
    """Greedy nearest-neighbor ordering of components.

    Starts from the centroid with minimal x+y (upper-left-ish) then repeatedly
    picks the remaining centroid with minimal Manhattan distance to current.
    Returns a list of component indices. Used by main() when mode != radial.
    """
    centroids = [centroid(c) for c in comps]
    n = len(comps)
    if n == 0:
        return []
    start = min(range(n), key=lambda i: centroids[i][0] + centroids[i][1])
    order = [start]
    remaining = set(range(n)) - {start}
    current = start
    while remaining:
        cx, cy = centroids[current]
        nxt = min(remaining, key=lambda i: abs(centroids[i][0]-cx)+abs(centroids[i][1]-cy))
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order

def order_zones_radial(comps):
    """Order components by angle around the global centroid (circular sweep)."""
    centroids = [centroid(c) for c in comps]
    if not centroids:
        return []
    cx = sum(x for x, _ in centroids) / len(centroids)
    cy = sum(y for _, y in centroids) / len(centroids)
    return sorted(range(len(comps)), key=lambda i: atan2(centroids[i][1]-cy, centroids[i][0]-cx))

def plot_order(mask, comps, order, filename="zone_order.png"):
    """Plot component centroids with visitation order and connecting dashed lines.

    Overlays the mask, labels each centroid with its position in `order`, draws
    dashed connections following the sequence, and saves to filename.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="gray_r", origin="lower", alpha=0.3)
    centroids = [centroid(c) for c in comps]
    for idx, i in enumerate(order):
        cx, cy = centroids[i]
        plt.plot(cx, cy, "ro")
        plt.text(cx, cy, str(idx+1), color="red", fontsize=8)
    for a, b in zip(order, order[1:]):
        ax, ay = centroids[a]; bx, by = centroids[b]
        plt.plot([ax, bx], [ay, by], "b--", linewidth=1)
    plt.axis("equal")
    plt.title("Ordre des zones")
    plt.savefig(filename, dpi=150)
    plt.close()


def boundary_points(comp, mask):
    """Return the list of boundary cells of comp (touching background or border)."""
    H, W = mask.shape
    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    boundary = []
    for x, y in comp:
        for dx, dy in dirs4:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or not mask[ny, nx]:
                boundary.append((x, y))
                break
    return boundary


def shortest_bridge(boundary_a, boundary_b, mask):
    """Shortest path between two zone boundaries through background cells.

    Treats non-mask cells as walkable, temporarily allowing boundary cells from
    both sets. BFS (deque) over walkable grid tracks parent pointers; returns
    the reconstructed route list if found, else a Manhattan fallback. Used by
    compute_links to connect successive zones. Mutates local walkable only.
    """
    H, W = mask.shape
    walkable = ~mask.copy()
    for p in boundary_a + boundary_b:
        walkable[p[1], p[0]] = True
    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    targets = set(boundary_b)
    q = deque()
    parent = {}
    for p in boundary_a:
        q.append(p)
        parent[p] = None
    found = None
    while q:
        x, y = q.popleft()
        if (x, y) in targets:
            found = (x, y)
            break
        for dx, dy in dirs4:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not walkable[ny, nx]:
                continue
            if (nx, ny) in parent:
                continue
            parent[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if found is None:
        # fallback Manhattan
        ax, ay = boundary_a[0]
        bx, by = boundary_b[0]
        path = [(ax, ay)]
        while (ax, ay) != (bx, by):
            if ax != bx:
                ax += 1 if bx > ax else -1
            elif ay != by:
                ay += 1 if by > ay else -1
            path.append((ax, ay))
        return path
    route = []
    cur = found
    while cur is not None:
        route.append(cur)
        cur = parent[cur]
    return list(reversed(route))


def compute_links(mask, comps, order):
    """Build bridges between ordered zones and assign per-zone start/end points.

    Computes boundary points per ordered component, calls shortest_bridge
    between each successive pair (including last->first), stores each bridge,
    and sets zone_start_end entries so that each zone ends at its outgoing
    bridge start and starts at the previous bridge end. Returns (bridges,
    zone_start_end) for downstream per-zone path construction.
    """
    ordered = [comps[i] for i in order]
    bpoints = [boundary_points(c, mask) for c in ordered]
    bridges = []
    zone_start_end = [dict(start=None, end=None) for _ in ordered]

    n = len(ordered)
    for idx in range(n):
        a = idx
        b = (idx + 1) % n  # boucle pour relier dernière -> première
        bridge = shortest_bridge(bpoints[a], bpoints[b], mask)
        bridges.append(bridge)
        # start/end pour zones
        zone_start_end[a]["end"] = bridge[0]
        zone_start_end[b]["start"] = bridge[-1]

    return bridges, zone_start_end


def plot_links(mask, comps, order, bridges, filename="zone_links.png"):
    """Visualize bridges between zones with start/end markers and labels."""
    plt.figure(figsize=(10, 7))
    plt.imshow(mask, cmap="gray_r", origin="lower", alpha=0.2)
    colors = plt.cm.tab10.colors
    for idx, br in enumerate(bridges):
        xs = [p[0] for p in br]
        ys = [p[1] for p in br]
        plt.plot(xs, ys, color=colors[idx % 10], linewidth=1.5)
        plt.plot(xs[0], ys[0], "go")  # start
        plt.plot(xs[-1], ys[-1], "ro")  # end
        src = order[idx]
        dst = order[(idx + 1) % len(order)]
        plt.text(xs[0], ys[0], f"{src+1}", color="green", fontsize=8)
        plt.text(xs[-1], ys[-1], f"{dst+1}", color="red", fontsize=8)
    plt.axis("equal")
    plt.title("Liens entre zones (vert=start, rouge=end)")
    plt.savefig(filename, dpi=150)
    plt.close()
#endregion

#region: chemins internes à chaque zone

def spanning_tree_from_start(comp, start):
    """Build a randomized spanning tree over component cells starting at `start`.

    Uses a stack-based DFS with shuffled dirs4 neighbors to connect all cells
    in comp without cycles. Returns a dict `tree` where each node maps to its
    adjacent nodes (undirected). Inputs are the component list and start
    coordinate. No external state mutated. Useful for generating tree walks.
    """
    pts = set(comp)
    tree = {start: []}
    stack = [start]
    visited = {start}
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    while stack:
        x, y = stack.pop()
        neighs = dirs4[:]
        np.random.shuffle(neighs)
        for dx, dy in neighs:
            nx, ny = x + dx, y + dy
            if (nx, ny) in pts and (nx, ny) not in visited:
                visited.add((nx, ny))
                tree.setdefault((x, y), []).append((nx, ny))
                tree.setdefault((nx, ny), []).append((x, y))
                stack.append((nx, ny))
    return tree

def euler_walk_to(tree, start, end, allowed=None):
    """Traverse every edge of a spanning tree then connect to `end`.

    Walks edges using a stack and `used` edge set to simulate an Euler tour
    (each edge exactly once), recording the visited vertices in `path`.
    After exhausting edges, performs a BFS from current node to `end` within
    the `allowed` set (defaults to tree nodes, plus end) using parent pointers,
    and appends that route (dropping duplicate start node if present). Returns
    the assembled path list. No inputs are mutated.
    """
    path = [start]
    stack = [start]
    used = set()
    while stack:
        u = stack[-1]
        next_v = None
        for v in tree.get(u, []):
            key = tuple(sorted([u, v]))
            if key in used:
                continue
            next_v = v
            used.add(key)
            break
        if next_v is None:
            stack.pop()
            if stack:
                path.append(stack[-1])
        else:
            stack.append(next_v)
            path.append(next_v)
    # relier à end via BFS dans allowed
    if allowed is None:
        allowed = set(tree.keys())
    if end not in allowed:
        allowed = allowed | {end}
    from collections import deque
    q = deque([path[-1]])
    parent = {path[-1]: None}
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    found = None
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            found = (x, y)
            break
        for dx, dy in dirs4:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in allowed:
                continue
            if (nx, ny) in parent:
                continue
            parent[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if found:
        route = []
        cur = found
        while cur is not None:
            route.append(cur)
            cur = parent[cur]
        route = list(reversed(route))
        # éviter doublon du point de départ de la route
        if route and route[0] == path[-1]:
            route = route[1:]
        path.extend(route)
    return path

def build_zone_paths(mask, comps, order, zone_start_end, curl=False):
    """Construct a high-coverage path for each zone using cover_path's backtracking/fill."""
    ordered = [comps[i] for i in order]
    paths = []
    misses = []

    for idx, comp in enumerate(ordered):
        zone = set(comp)
        start = tuple(zone_start_end[idx]["start"]) if zone_start_end[idx]["start"] else None
        end = tuple(zone_start_end[idx]["end"]) if zone_start_end[idx]["end"] else None
        if start is None or end is None:
            paths.append([])
            misses.append(len(zone))
            continue

        # use cover_path's backtracking + recursive fill for coverage
        zone_time_limit = min(300.0, max(6.0, len(zone) / 20000))
        strat = curl_strategy_random_backtrack if curl else strategy_random_backtrack
        base_path = strat(zone, start, end, time_limit=zone_time_limit, plateau_seconds=5, live=False)
        path = recursive_fill(zone, start, end, base_path, time_limit=zone_time_limit, live=False)
        # simplify to prevent crossings/duplicates and enforce single-neighbor endpoints
        def simplify(seq, s, e):
            if not seq:
                return [s, e]
            seen = set()
            simple = []
            for p in seq:
                p = (int(p[0]), int(p[1]))
                if p in seen:
                    continue
                seen.add(p)
                simple.append(p)
            if simple[0] != s:
                simple.insert(0, s)
            if simple[-1] != e:
                simple.append(e)
            return simple

        final_path = simplify(path, start, end)
        paths.append(final_path)
        misses.append(len(zone) - len(set(final_path)))
        covered_pct = 100.0 * (len(zone) - misses[-1]) / len(zone)
        sys.stdout.write("\r")
        sys.stdout.write(f"Zone {order[idx]+1}: {covered_pct:.1f}% covered, {misses[-1]}/{len(zone)} missed\n")
        sys.stdout.flush()
    return paths, misses

def plot_zone_paths(mask, zone_paths, order, filename_prefix="zone_path", must_visits=None):
    """Save individual plots for each zone path with start/end markers.

    Iterates zone_paths ordered by `order`, converts coordinates to ints,
    overlays them on the mask, optionally scatters must_visits cells, and saves
    to files named by filename_prefix and zone index. Skips empty/malformed
    paths. No return; relies on matplotlib for rendering.
    """
    colors = plt.cm.tab10.colors
    for i, p in enumerate(zone_paths):
        if not p:
            continue
        coords = []
        for pt in p:
            if pt is None or len(pt) < 2:
                continue
            try:
                coords.append((int(pt[0]), int(pt[1])))
            except Exception:
                continue
        if not coords:
            continue
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap="gray_r", origin="lower", alpha=0.2)
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        plt.plot(xs, ys, color=colors[i % 10], linewidth=1.5)
        plt.plot(xs[0], ys[0], "go")
        plt.plot(xs[-1], ys[-1], "ro")
        if must_visits and must_visits[i]:
            mx = [pt[0] for pt in must_visits[i]]
            my = [pt[1] for pt in must_visits[i]]
            plt.scatter(mx, my, c="cyan", s=10, marker="s", label="must-visit")
            plt.legend(loc="upper right")
        plt.title(f"Chemin zone {order[i]+1}")
        plt.axis("equal")
        plt.savefig(f"{filename_prefix}_{order[i]+1}.png", dpi=150)
        plt.close()

def plot_zone_paths_combined(mask, zone_paths, order, filename="zone_paths_combined.png"):
    """Overlay all zone paths in one figure with labeled start/end markers."""
    colors = plt.cm.tab20.colors
    plt.figure(figsize=(10, 7))
    plt.imshow(mask, cmap="gray_r", origin="lower", alpha=0.2)
    for i, p in enumerate(zone_paths):
        if not p:
            continue
        coords = []
        for pt in p:
            if pt is None or len(pt) < 2:
                continue
            try:
                coords.append((int(pt[0]), int(pt[1])))
            except Exception:
                continue
        if not coords:
            continue
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        plt.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.2)
        plt.plot(xs[0], ys[0], "go", markersize=4)
        plt.plot(xs[-1], ys[-1], "ro", markersize=4)
        plt.text(xs[0], ys[0], f"S{order[i]+1}", color="green", fontsize=7)
        plt.text(xs[-1], ys[-1], f"E{order[i]+1}", color="red", fontsize=7)
    plt.axis("equal")
    plt.title("Chemins internes par zone (vert=start, rouge=end)")
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_zone_zoom(mask, zone_paths, order, idx=0, filename="zone_zoom.png"):
    """Zoomed plot of a single zone path for detailed inspection."""
    if not zone_paths or idx >= len(zone_paths):
        return
    p = zone_paths[idx]
    coords = []
    for pt in p:
        if pt is None or len(pt) < 2:
            continue
        try:
            coords.append((int(pt[0]), int(pt[1])))
        except Exception:
            continue
    if not coords:
        return
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray_r", origin="lower", alpha=0.1)
    plt.plot(xs, ys, color="blue", linewidth=0.5)
    plt.plot(xs[0], ys[0], "go", markersize=4)
    plt.plot(xs[-1], ys[-1], "ro", markersize=4)
    plt.xlim(min(xs)-2, max(xs)+2)
    plt.ylim(min(ys)-2, max(ys)+2)
    plt.gca().set_aspect('equal')
    plt.title(f"Zone {order[idx]+1} (zoom)")
    plt.savefig(filename, dpi=200)
    plt.close()

def build_full_path(zone_paths, bridges_linear, global_start, global_end):
    """Concatenate per-zone paths and bridges into a single global path.

    Sequence is zone_paths[0], bridge0, zone1, bridge1, ... Append global_start
    or global_end if missing. Normalizes heterogeneous coordinate containers via
    norm(), then removes immediate duplicates and any repeated cells to produce
    a simple self-avoiding path. Returns the simplified list.
    """
    seq = []
    for idx, zp in enumerate(zone_paths):
        seq.extend(zp)
        if idx < len(bridges_linear):
            seq.extend(bridges_linear[idx])
    if seq and global_start and seq[0] != global_start:
        seq = [global_start] + seq
    if seq and global_end and seq[-1] != global_end:
        seq = seq + [global_end]
    # simplification : pas de doublons ni retours sur une cellule déjà prise
    simple = []
    seen = set()
    def norm(pt):
        """Normalize heterogeneous point representations into a 2-int tuple or None."""
        if isinstance(pt, (list, tuple, np.ndarray)):
            if len(pt) == 0:
                return None
            if len(pt) >= 2 and not isinstance(pt[0], (list, tuple, np.ndarray)):
                try:
                    return (int(pt[0]), int(pt[1]))
                except Exception:
                    pass
            flat = []
            for item in pt:
                if isinstance(item, (list, tuple, np.ndarray)):
                    if len(item) >= 2:
                        flat.extend([int(item[0]), int(item[1])])
                    elif len(item) == 1:
                        flat.append(int(item[0]))
                else:
                    flat.append(int(item))
                if len(flat) >= 2:
                    return (flat[0], flat[1])
            return None
        try:
            return (int(pt[0]), int(pt[1]))
        except Exception:
            return None
    for p in seq:
        p = norm(p)
        if p is None:
            continue
        if simple and p[0] == simple[-1][0] and p[1] == simple[-1][1]:
            continue
        if p in seen:
            continue
        simple.append(p)
        seen.add(p)
    return simple

def write_path_maze(path_seq, filename="maze_path.txt"):
    """Write the path sequence to disk in a simple adjacency text format.

    Outputs start/end headers plus each cell with its neighbor count and
    immediate predecessor/successor coordinates. Used by main() to export the
    maze path for consumption elsewhere.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Start:\n")
        f.write(f"({path_seq[0][0]},{path_seq[0][1]})\n")
        f.write("# End:\n")
        f.write(f"({path_seq[-1][0]},{path_seq[-1][1]})\n")
        f.write("# Cells:\n")
        for i, (x, y) in enumerate(path_seq):
            neigh = []
            if i > 0:
                neigh.append(path_seq[i-1])
            if i + 1 < len(path_seq):
                neigh.append(path_seq[i+1])
            line = f"({x},{y}){len(neigh)}"
            for nx, ny in neigh:
                line += f"({nx},{ny})"
            f.write(line + "\n")

def plot_assembly(mask, zone_paths, order, bridges, global_start, global_end, filename="assembly.png"):
    """Combined plot showing all zone paths, bridges, and global endpoints."""
    colors = plt.cm.tab20.colors
    plt.figure(figsize=(10, 7))
    plt.imshow(mask, cmap="gray_r", origin="lower", alpha=0.2)
    # zones
    for i, p in enumerate(zone_paths):
        if not p:
            continue
        coords = []
        for pt in p:
            if pt is None or len(pt) < 2:
                continue
            try:
                coords.append((int(pt[0]), int(pt[1])))
            except Exception:
                continue
        if not coords:
            continue
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        plt.plot(xs, ys, color=colors[i % len(colors)], linewidth=1.0)
    # bridges (sans le dernier)
    for idx, br in enumerate(bridges):
        if not br:
            continue
        bcoords = []
        for p in br:
            if p is None or len(p) < 2:
                continue
            try:
                bcoords.append((int(p[0]), int(p[1])))
            except Exception:
                continue
        if not bcoords:
            continue
        xs = [p[0] for p in bcoords]
        ys = [p[1] for p in bcoords]
        plt.plot(xs, ys, color="black", linewidth=1.5, linestyle="--")
    # start/end globaux
    plt.plot(global_start[0], global_start[1], "go", markersize=6)
    plt.plot(global_end[0], global_end[1], "ro", markersize=6)
    plt.text(global_start[0], global_start[1], "START", color="green", fontsize=8)
    plt.text(global_end[0], global_end[1], "END", color="red", fontsize=8)
    plt.axis("equal")
    plt.title("Assemblage (zones + liens, start/end globaux)")
    plt.savefig(filename, dpi=150)
    plt.close()
#endregion

#region: maze generation

def write_mask_maze(path_seq, start=None, end=None, filename="maze_full.txt"):
    """Write a path-based maze file where neighbors are prev/next along the path.

    Ensures start/end are included in the cell list (prepends/appends if missing)
    and outputs the same text format as img_to_maze.py: each cell lists its
    immediate predecessor/successor in the path as neighbors.
    """
    if not path_seq:
        return
    seq = []
    seen = set()
    for pt in path_seq:
        try:
            p = (int(pt[0]), int(pt[1]))
        except Exception:
            continue
        if p in seen:
            continue
        seen.add(p)
        seq.append(p)
    if start is not None:
        start = (int(start[0]), int(start[1]))
        seq = [p for p in seq if p != start]
        seq.insert(0, start)
    if end is not None:
        end = (int(end[0]), int(end[1]))
        seq = [p for p in seq if p != end]
        seq.append(end)
    if not start:
        start = seq[0]
    if not end:
        end = seq[-1]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Start:\n")
        f.write(f"({start[0]},{start[1]})\n")
        f.write("# End:\n")
        f.write(f"({end[0]},{end[1]})\n")
        f.write("# Cells:\n")
        for i, (x, y) in enumerate(seq):
            neigh = []
            if i > 0:
                neigh.append(seq[i-1])
            if i + 1 < len(seq):
                neigh.append(seq[i+1])
            line = f"({x},{y}){len(neigh)}"
            for nx, ny in neigh:
                line += f"({nx},{ny})"
            f.write(line + "\n")
#endregion

#region: full path generation (rectangular maze with branching tree)

def build_branching_maze(path_seq, padding=1):
    """Build adjacency for a rectangular maze: path preserved, branches form a tree off the path.

    Returns (adjacency dict, added_edges) where added_edges are the extra neighbor links beyond
    the original path. Endpoints keep only their path neighbor; branches attach elsewhere.
    """
    if not path_seq:
        return {}, []
    seq = [(int(p[0]), int(p[1])) for p in path_seq]
    xs = [p[0] for p in seq]; ys = [p[1] for p in seq]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bounds = (minx - padding, maxx + padding, miny - padding, maxy + padding)
    # full rectangle grid
    grid = {(x, y) for x in range(bounds[0], bounds[1] + 1) for y in range(bounds[2], bounds[3] + 1)}
    path_set = set(seq)
    carved = set(path_set)
    adj = {p: set() for p in path_set}
    # path edges
    for a, b in zip(seq, seq[1:]):
        adj[a].add(b); adj[b].add(a)
    remaining = grid - carved
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    start, end = seq[0], seq[-1]
    added_edges = []
    max_degree = 3  # cap degree to avoid hubs/blue squares

    def neighbors(cell):
        x, y = cell
        for dx, dy in dirs4:
            cand = (x+dx, y+dy)
            if cand in grid:
                yield cand

    frontier = [c for c in remaining if any(n in carved for n in neighbors(c))]
    random.shuffle(frontier)

    def carve_to_parent(cell, parent):
        carved.add(cell)
        remaining.discard(cell)
        adj.setdefault(cell, set()).add(parent)
        adj.setdefault(parent, set()).add(cell)
        added_edges.append((parent, cell))

    while remaining:
        if not frontier:
            # connect an isolated cell by carving a Manhattan path to nearest carved cell
            current = remaining.pop()
            target = min(carved, key=lambda p: abs(p[0]-current[0]) + abs(p[1]-current[1]))
            path_line = []
            cx, cy = current
            tx, ty = target
            while (cx, cy) != (tx, ty):
                if cx != tx:
                    cx += 1 if tx > cx else -1
                elif cy != ty:
                    cy += 1 if ty > cy else -1
                path_line.append((cx, cy))
            prev = target
            for cell in path_line[::-1]:
                if cell not in carved:
                    if prev in (start, end) or len(adj.get(prev, [])) >= max_degree:
                        break
                    carve_to_parent(cell, prev)
                prev = cell
            frontier.extend([c for c in remaining if any(n in carved for n in neighbors(c))])
            continue

        cell = frontier.pop()
        if cell not in remaining:
            continue
        parents = [n for n in neighbors(cell) if n in carved]
        if not parents:
            continue
        # avoid start/end and degree overflows
        parents = [
            p for p in parents
            if p not in (start, end) and len(adj.get(p, [])) < max_degree
        ]
        if not parents:
            continue
        parent = random.choice(parents)
        carve_to_parent(cell, parent)
        # add new frontier cells adjacent to freshly carved cell
        for n in neighbors(cell):
            if n in remaining and any(nb in carved for nb in neighbors(n)):
                frontier.append(n)
        random.shuffle(frontier)

    return adj, added_edges

def write_rect_maze(adj, start, end, filename="maze_rect.txt"):
    """Write adjacency dict to text file in maze format."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Start:\n")
        f.write(f"({start[0]},{start[1]})\n")
        f.write("# End:\n")
        f.write(f"({end[0]},{end[1]})\n")
        f.write("# Cells:\n")
        for (x, y), neighs in adj.items():
            neigh_list = sorted(neighs)
            line = f"({x},{y}){len(neigh_list)}"
            for nx, ny in neigh_list:
                line += f"({nx},{ny})"
            f.write(line + "\n")
#endregion

#region: main

def main():
    """CLI entry point: build mask from an image, derive zone paths, and export artifacts.

    Reads argv for logo_path/target_width/mode, computes mask -> components ->
    order -> bridges -> per-zone paths, reports missed cells, saves compressed
    npz, multiple plots, JSON summary, and maze_path.txt. Mutates filesystem
    via writes; prints coverage stats to stdout. Returns None.
    """
    logo_path = sys.argv[1] if len(sys.argv) > 1 else "images.png"
    target_width = int(sys.argv[2]) if len(sys.argv) > 2 else 200 ##################
    mode = sys.argv[3] if len(sys.argv) > 3 else "nearest"  ########################
    curl = True#####################################################################
    if len(sys.argv) > 4:
        curl = sys.argv[4].lower() in ("curl", "true", "1", "yes")

    mask = load_black_mask(logo_path, target_width=target_width)
    comps, comp_masks = find_components(mask)

    print(f"Mask shape: {mask.shape}, black cells: {mask.sum()}, components: {len(comps)}")

    if mode == "radial":
        order = order_zones_radial(comps)
    else:
        order = order_zones_nearest(comps)

    bridges, zone_start_end = compute_links(mask, comps, order)
    zone_paths, misses = build_zone_paths(mask, comps, order, zone_start_end, curl=curl)
    misses = []
    ordered_comps = [comps[i] for i in order]
    for idx, comp in enumerate(ordered_comps):
        if idx < len(zone_paths):
            missed = len(comp) - len(set(zone_paths[idx]))
        else:
            missed = len(comp)
        misses.append(missed)
        print(f"{missed}/{len(comp)} cells missed in zone {order[idx]+1}")

    # ponts pour le chemin global : on retire le dernier (entre dernière et première)
    bridges_linear = bridges[:-1] if bridges else []
    global_start = bridges[-1][-1] if bridges else None
    global_end = bridges[-1][0] if bridges else None
    full_path = build_full_path(zone_paths, bridges_linear, global_start, global_end)

    np.savez_compressed("zones_data.npz",
                        mask=mask,
                        components=np.array(comps, dtype=object),
                        order=np.array(order, dtype=int),
                        bridges=np.array(bridges, dtype=object),
                        zone_start_end=np.array(zone_start_end, dtype=object),
                        zone_paths=np.array(zone_paths, dtype=object),
                        bridges_linear=np.array(bridges_linear, dtype=object),
                        global_start=np.array(global_start) if global_start else None,
                        global_end=np.array(global_end) if global_end else None,
                        full_path=np.array(full_path, dtype=object))

    plot_components(comp_masks, prefix="zone")
    plot_order(mask, comps, order, filename="zone_order.png")
    plot_links(mask, comps, order, bridges, filename="zone_links.png")
    plot_zone_paths(mask, zone_paths, order, filename_prefix="zone_path", must_visits=None)
    plot_zone_paths_combined(mask, zone_paths, order, filename="zone_paths_combined.png")
    plot_zone_zoom(mask, zone_paths, order, idx=0, filename="zone_zoom.png")
    if global_start and global_end:
        plot_assembly(mask, zone_paths, order, bridges_linear, global_start, global_end, filename="assembly.png")
    if full_path:
        write_path_maze(full_path, filename="maze_path.txt")
        start_for_maze = global_start if global_start else full_path[0]
        end_for_maze = global_end if global_end else full_path[-1]
        write_mask_maze(full_path, start=start_for_maze, end=end_for_maze, filename="maze_full.txt")
        rect_adj, added_edges = build_branching_maze(full_path, padding=1)
        write_rect_maze(rect_adj, start_for_maze, end_for_maze, filename="maze_rect.txt")
        with open("maze_rect_added_edges.txt", "w", encoding="utf-8") as f:
            for (a, b) in added_edges:
                f.write(f"{a}->{b}\n")

    # dump starts/ends in a small JSON for inspection
    def pair(pt):
        """Normalize a coordinate-like container into a 2-tuple of ints if possible."""
        try:
            if isinstance(pt, (list, tuple, np.ndarray)):
                if len(pt) >= 2 and not isinstance(pt[0], (list, tuple, np.ndarray)):
                    return (int(pt[0]), int(pt[1]))
                if len(pt) >= 1 and isinstance(pt[0], (list, tuple, np.ndarray)):
                    return (int(pt[0][0]), int(pt[0][1]))
        except Exception:
            return None
        return None

    info = {
        "order": [int(i) for i in order],
        "zone_start_end": [
            {"start": (int(v["start"][0]), int(v["start"][1])) if v["start"] else None,
             "end": (int(v["end"][0]), int(v["end"][1])) if v["end"] else None}
            for v in zone_start_end
        ],
        "bridges": [[(int(px), int(py)) for px, py in br] for br in bridges],
        "zone_paths": [
            [pair(pt) for pt in zp if pair(pt) is not None]
            for zp in zone_paths
        ],
        "bridges_linear": [[(int(px), int(py)) for px, py in br] for br in bridges_linear],
        "global_start": (int(global_start[0]), int(global_start[1])) if global_start else None,
        "global_end": (int(global_end[0]), int(global_end[1])) if global_end else None,
        "full_path": [[int(px), int(py)] for px, py in full_path],
        "misses": misses,
    }
    with open("zones_links.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
#endregion
