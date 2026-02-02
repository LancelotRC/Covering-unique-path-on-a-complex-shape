"""
Coverage-oriented path builders and visualizers used for maze-like traversal experiments.

Dependencies:
- Built-ins sys/random/time plus collections.deque for recursion control, randomness, and BFS queues.
- matplotlib.pyplot as plt for optional live previews and saved plots.

Data conventions:
- Zones are sets of integer (x, y) cells; paths are ordered lists of those coordinates.
- start/end are (x, y) tuples that must belong to the chosen zone.

Functions and key variables:
- make_square_zone/make_s_shape/make_ring: generate example zones (square grid, S-shape, annulus). No shared state; return a set of cells.
- neighbors(pt): cardinal neighbors used by all graph searches.
- shortest_path(a, b, zone, forbid): BFS that respects zone membership and a forbid set (except for b) to stitch tails back to the end; uses deque parent map reconstruction.
- strategy_random_backtrack(zone, start, end, time_limit, live, plateau_seconds, min_improve, path_):
  randomized DFS/backtracking that maximizes covered cells before reaching the end. Builds adj (neighbor list per cell), tracks best_path/best_cov, plateau_stop flag, deadline/last_report timers, and last_improve_time/last_plot_time for plateau detection and throttled plotting. reachable() prunes dead ends; dfs() mutates visited/path sets recursively; a final shortest_path tail guarantees end connectivity.
- find_split_edge(zone, path): scans an existing path to find an edge that borders the same remaining component (via components()) so it can be reopened; returns (idx, p, q, comp) or None.
- recursive_fill(zone, start, end, base_path, time_limit, live): repeatedly calls strategy_random_backtrack on holes identified around base_path, splicing improved sub_paths; recurses until no remaining cells.
- components(cells): BFS flood-fill returning a list of connected components; used for hole detection.
- shoelace_path(comp): serpentine sweep through a component's bounding box (alternating direction per row) to cover remaining cells deterministically.
- refine_shoelace(zone, best_path, end): connects current tail to nearest remaining component, shoelaces through it, then routes to end via shortest_path; relies on visited tracking.
- longest_straight_segment(path): finds the longest contiguous straight segment, returning (start_idx, length) for later splicing.
- split_segment_fill(zone, best_path, time_limit): picks the midpoint of the longest straight segment, finds an adjacent hole component, and re-runs strategy_random_backtrack on that sub_zone before splicing it back.
- fill_adjacent_component(zone, best_path, time_limit): looks for a path edge touching the same remaining component on both sides; replaces that edge with a sub_path from strategy_random_backtrack if it increases coverage.
- iter_split_and_fill(zone, path, max_loops, time_limit): iterative refinement loop applying fill_adjacent_component and split_segment_fill until coverage stops improving.
- plot_path(zone, start, end, path, filename): scatter-plot of the zone with the path overlayed and start/end markers; saves to disk.
- run_shape(zone, start, end, label): seeds RNG, sets an adaptive time limit, runs strategy_random_backtrack then recursive_fill with live plotting, prints coverage, and saves a test_path PNG via plot_path().
- main(): builds three sample zones (S-shape, ring, square) with deterministic starts/ends and runs the above pipeline for manual inspection.

Primary state variables (used across the search helpers):
- zone: set of allowed cells; visited/path: mutable traversal state inside DFS; best_path/best_cov: best-so-far metrics; adj: adjacency lookup per cell; plateau_stop/last_improve_time/deadline/last_report/last_plot_time: timers and flags that gate recursion, logging, and plotting.
"""

import sys
import random
import time
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt

#region: helpers

def make_square_zone(size=30):
    """Generate a square zone of given side length.

    Used by main() to build a simple benchmark grid. Returns a set of (x, y)
    integer coordinates covering [0, size) in both axes. No side effects.
    """
    return {(x, y) for y in range(size) for x in range(size)}

def make_s_shape(width=60, height=40, thickness=6):
    """Generate an S-shaped zone composed of three bars and two vertical links.

    Parameters set the rectangle width/height and bar thickness. Returns a set
    of (x, y) cells arranged in an S layout; used by main() for a varied test
    topology. Pure function, no external dependencies beyond locals.
    """
    zone = set()
    # top horizontal
    for y in range(thickness):
        for x in range(width):
            zone.add((x, y))
    # middle bar
    mid_y = height // 2
    for y in range(mid_y - thickness//2, mid_y + thickness//2 + 1):
        for x in range(width):
            zone.add((x, y))
    # bottom horizontal
    for y in range(height - thickness, height):
        for x in range(width):
            zone.add((x, y))
    # vertical connectors
    for y in range(0, mid_y + 1):
        for x in range(0, thickness):
            zone.add((x, y))
    for y in range(mid_y, height):
        for x in range(width - thickness, width):
            zone.add((x, y))
    return zone

def make_ring(center=(40,40), inner_r=12, outer_r=18):
    """Generate an annulus (ring) zone centered at `center`.

    Iterates over a bounding square and includes cells whose squared distance
    falls between inner_r^2 and outer_r^2. Used in main() as a torus-like
    coverage test. Returns a set of (x, y) cells.
    """
    cx, cy = center
    zone = set()
    for x in range(cx - outer_r - 2, cx + outer_r + 3):
        for y in range(cy - outer_r - 2, cy + outer_r + 3):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if inner_r * inner_r <= d2 <= outer_r * outer_r:
                zone.add((x, y))
    return zone

def neighbors(pt):
    """Return the 4-neighborhood (cardinal) for a point pt=(x, y)."""
    x, y = pt
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def shortest_path(a, b, zone, forbid):
    """BFS shortest path from a to b within `zone`, respecting a forbid set.

    `forbid` cells are avoided except when the cell is b. Returns a list of
    coordinates from a to b (inclusive) or [] if unreachable. Used to stitch
    tails and to connect to endpoints in several refinement steps. Uses a deque
    queue and parent backpointers; mutates no global state.
    """
    q = deque([a]); parent = {a: None}
    while q:
        u = q.popleft()
        if u == b:
            break
        for v in neighbors(u):
            if v not in zone:
                continue
            if forbid:
                if v in forbid and v != b:
                    continue
            if v in parent:
                continue
            parent[v] = u
            q.append(v)
    if b not in parent:
        return []
    res = []
    cur = b
    while cur is not None:
        res.append(cur)
        cur = parent[cur]
    return list(reversed(res))

def strategy_random_backtrack(zone, start, end, time_limit=5.0, live=False,
                              plateau_seconds=5.0, min_improve=5, path_=None):
    """Randomized DFS/backtracking to cover a zone from start to end.

    Builds adjacency (adj) from zone, then explores self-avoiding paths using
    dfs() with shuffle + degree heuristic. Tracks best_path/best_cov, timers
    (deadline, last_report, last_improve_time, last_plot_time), and a
    plateau_stop flag when no improvement for plateau_seconds. reachable()
    prunes branches that would disconnect the end. update_plot() optionally
    refreshes a live matplotlib view. On finish, if best_path does not end at
    `end`, a shortest_path tail is appended. Returns the best_path list.
    """
    adj = {p: [v for v in neighbors(p) if v in zone] for p in zone}
    best_path = shortest_path(start, end, zone, path_)
    best_cov = len(best_path) if best_path else 1
    deadline = time.time() + time_limit
    last_report = time.time()
    last_plot_time = time.time()
    last_plot_time = time.time()
    stop_reason = None
    target_cov = int(0.99 * len(zone))

    def reachable(node, remaining):
        """Check via BFS if `end` is reachable from node within remaining cells."""
        allowed = remaining | {end}
        q = deque([node]); seen = {node}
        while q:
            u = q.popleft()
            if u == end:
                return True
            for v in adj[u]:
                if v in seen or v not in allowed:
                    continue
                seen.add(v); q.append(v)
        return False

    sys.setrecursionlimit(200000)

    plateau_stop = False

    def dfs(u, visited, path):
        """Recursive randomized DFS; updates best_path/best_cov when improved."""
        nonlocal best_path, best_cov, last_report, last_improve_time, plateau_stop, last_plot_time, stop_reason
        if time.time() > deadline:
            plateau_stop = True
            stop_reason = stop_reason or "time_limit"
            return
        if best_cov >= target_cov or plateau_stop:
            return
        if time.time() - last_improve_time > plateau_seconds:
            plateau_stop = True
            stop_reason = stop_reason or "plateau"
            return
        if len(visited) > best_cov:
            best_cov = len(visited); best_path = path.copy(); last_improve_time = time.time()
            # update plot only on improvement
            if live and time.time() - last_plot_time > 1:
                plt.cla()
                zx = [p[0] for p in zone]; zy = [p[1] for p in zone]
                plt.scatter(zx, zy, c="lightgray", s=5, marker="s")
                bx = [p[0] for p in best_path]; by = [p[1] for p in best_path]
                plt.plot(bx, by, color="blue", linewidth=0.5)
                plt.plot(bx[0], by[0], "go"); plt.plot(bx[-1], by[-1], "ro")
                plt.axis("equal"); plt.title(f"Best cov {100.0*best_cov/len(zone):.1f}%")
                plt.pause(0.01)
                last_plot_time = time.time()
        now = time.time()
        if now - last_report >= 1.0:
            cov = 100.0 * len(visited) / len(zone)
            elapsed = time_limit - max(0.0, deadline - now)
            sys.stdout.write("\r")
            sys.stdout.write(f"[rand] cov {cov:.1f}% elapsed {elapsed:.1f}s best {100.0*best_cov/len(zone):.1f}%")
            sys.stdout.flush()
            last_report = now
        neighs = [v for v in adj[u] if v not in visited]
        random.shuffle(neighs)
        neighs.sort(key=lambda n: sum(1 for nb in adj[n] if nb not in visited))
        for v in neighs:
            rem = zone - visited - {v}
            if not reachable(v, rem):
                continue
            visited.add(v); path.append(v)
            dfs(v, visited, path)
            path.pop(); visited.remove(v)
            if best_cov >= target_cov or plateau_stop:
                return

    visited = {start}
    path = [start]
    last_improve_time = time.time()
    sys.stdout.write(f"[rand] start from {start} to {end} with zone size {len(zone)} tl={time_limit}s\n"); sys.stdout.flush()
    dfs(start, visited, path)
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.stdout.write(f"[rand] done reason={stop_reason or 'target'} best_cov={best_cov}/{len(zone)}\n")
    sys.stdout.flush()
    if best_path and best_path[-1] != end:
        tail = shortest_path(best_path[-1], end, zone, set(best_path))
        if tail:
            best_path = best_path + tail[1:]
    return best_path

#region: recursive split-and-fill solver

def find_split_edge(zone, path):
    """Find an edge in `path` that borders the same remaining component on both sides.

    Builds visited from path, computes remaining via components(), then scans
    consecutive path vertices to locate a split candidate that can be reopened.
    Returns tuple (i, p, q, comp) or None. Used by recursive_fill.
    """
    visited = set(path)
    remaining = zone - visited
    if not remaining:
        return None
    comps = components(remaining)
    for i in range(len(path) - 1):
        p = path[i]; q = path[i+1]
        neigh_p = [v for v in neighbors(p) if v in remaining]
        neigh_q = [v for v in neighbors(q) if v in remaining]
        if not (neigh_p and neigh_q):
            continue
        for comp in comps:
            if any(v in comp for v in neigh_p) and any(v in comp for v in neigh_q):
                return i, p, q, comp
    return None

def recursive_fill(zone, start, end, base_path, time_limit=35.0, live=False, plot=False, snap_counter=None, snapshot_prefix= None):
    """Re-run backtracking on uncovered components and splice improvements.

    Takes a base_path (from strategy_random_backtrack), identifies remaining
    components via components(), chooses the first edge that touches the same
    component on both sides, and runs strategy_random_backtrack on that subzone.
    Splices the sub_path into base_path and recurses until full coverage or no
    improvement. Returns an updated path; prints when recursion occurs. Plotting
    is disabled by default; when plot is True, before/after images per recursion
    step are saved.
    """
    visited = set(base_path)
    remaining = zone - visited
    if not remaining:
        return base_path
    if snap_counter is None:
        snap_counter = [0]
    sys.stdout.write(f"[recur] remaining cells {len(remaining)}; scanning edges\n"); sys.stdout.flush()
    comps = components(remaining)
    for i in range(len(base_path) - 1):
        p = base_path[i]; q = base_path[i+1]
        neigh_p = [v for v in neighbors(p) if v in remaining]
        neigh_q = [v for v in neighbors(q) if v in remaining]
        if not (neigh_p and neigh_q):
            continue
        target_comp = None
        for comp in comps:
            if any(v in comp for v in neigh_p) and any(v in comp for v in neigh_q):
                target_comp = comp
                break
        if target_comp is None:
            continue
        sub_zone = target_comp | {p, q}
        sys.stdout.write(f"[recur] retrying sub-zone size {len(sub_zone)} between {p}->{q}\n"); sys.stdout.flush()
        sub_path = strategy_random_backtrack(sub_zone, p, q, time_limit=time_limit, plateau_seconds=5 , live=live, path_ = base_path)
        if not sub_path or len(sub_path) <= 2:
            continue
        new_path = base_path[:i] + sub_path + base_path[i+1:]
        # try to recurse further
        result_path = recursive_fill(zone, start, end, new_path, time_limit=time_limit, live=live, plot=plot, snap_counter=snap_counter)
        if plot:
            plot_path(zone, start, end, base_path, f"test_path_recur{snap_counter[0]}_before.png")
            plot_path(zone, start, end, result_path, f"test_path_recur{snap_counter[0]}_after.png")
            snap_counter[0] += 1
        return result_path
    return base_path

#region: refinement (shoelace on remaining)

def components(cells):
    """Split a set of cells into 4-connected components.

    Uses BFS flood-fill with a deque to accumulate each component. Returns a
    list of sets. Pure utility shared by multiple refinement steps.
    """
    cells = set(cells)
    comps = []
    while cells:
        start = next(iter(cells))
        comp = set()
        q = deque([start])
        comp.add(start); cells.remove(start)
        while q:
            u = q.popleft()
            for v in neighbors(u):
                if v in cells:
                    cells.remove(v); comp.add(v); q.append(v)
        comps.append(comp)
    return comps

def shoelace_path(comp):
    """Produce a serpentine (row-wise) traversal covering all cells in comp.

    Sweeps the bounding box of the component, alternating left-to-right then
    right-to-left per row, skipping holes. Returns an ordered list of cells.
    """
    minx = min(x for x, _ in comp); maxx = max(x for x, _ in comp)
    miny = min(y for _, y in comp); maxy = max(y for _, y in comp)
    path = []
    for i, y in enumerate(range(miny, maxy + 1)):
        xs = list(range(minx, maxx + 1))
        if i % 2 == 1:
            xs = list(reversed(xs))
        for x in xs:
            if (x, y) in comp:
                path.append((x, y))
    return path

def refine_shoelace(zone, best_path, end):
    """Fill the largest remaining component using shoelace coverage and reconnect to end.

    Finds remaining via visited tracking, picks the component containing `end`
    (or the largest), connects tail to nearest cell in comp using shortest_path,
    runs shoelace_path over comp, and finally connects to end if needed.
    Returns a new path; no mutation of inputs besides local visited set.
    """
    visited = set(best_path)
    remaining = zone - visited
    if not remaining:
        return best_path
    comps = components(remaining)
    comp_end = None
    for c in comps:
        if end in c:
            comp_end = c
            break
    if comp_end is None:
        comp_end = max(comps, key=len)
    comp = comp_end
    # connect from current tail to nearest cell in comp
    tail = best_path[-1]
    candidates = sorted(list(comp), key=lambda p: abs(p[0]-tail[0])+abs(p[1]-tail[1]))
    connect = []
    for cand in candidates[:50]:
        seg = shortest_path(tail, cand, zone, visited)
        if seg:
            connect = seg[1:]
            break
    if not connect:
        return best_path
    new_path = best_path + connect
    visited.update(connect)
    # shoelace fill inside comp
    lace = shoelace_path(comp)
    for p in lace:
        if p in visited:
            continue
        new_path.append(p); visited.add(p)
    # connect to end if not already
    if new_path[-1] != end:
        seg = shortest_path(new_path[-1], end, zone, visited)
        if seg:
            new_path.extend(seg[1:])
    return new_path

#region: split a straight segment and fill adjacent hole (shoelace variant)

def longest_straight_segment(path):
    """Identify the longest straight (axis-aligned) segment in a path.

    Returns (start_idx, length) or None if no straight run longer than 1.
    Scans path once, tracking current run length; used by split_segment_fill.
    """
    best = (0, 0)
    cur_len = 1
    start_idx = 0
    for i in range(1, len(path)):
        if path[i][0] == path[i-1][0] or path[i][1] == path[i-1][1]:
            cur_len += 1
        else:
            cur_len = 1
            start_idx = i
        if cur_len > best[1]:
            best = (start_idx, cur_len)
    if best[1] < 2:
        return None
    return best

def split_segment_fill(zone, best_path, time_limit=2.0):
    """Reopen the midpoint of the longest straight segment to fill an adjacent hole.

    Chooses candidate side cell in remaining zone, finds its component, runs
    strategy_random_backtrack on sub_zone {comp + endpoints}, and splices the
    resulting sub_path into best_path. Returns the improved path or original.
    """
    seg = longest_straight_segment(best_path)
    if not seg:
        return best_path
    start_idx, length = seg
    mid_idx = start_idx + length // 2
    if mid_idx >= len(best_path) - 1:
        return best_path
    a = best_path[mid_idx]
    b = best_path[mid_idx + 1]
    # choose side cell into remaining
    remaining = zone - set(best_path)
    if a[0] == b[0]:  # vertical
        candidates = [(a[0]+1, a[1]), (a[0]-1, a[1])]
    else:  # horizontal
        candidates = [(a[0], a[1]+1), (a[0], a[1]-1)]
    side = None
    for c in candidates:
        if c in remaining:
            side = c
            break
    if side is None:
        return best_path
    # component adjacent to side
    comps = components(remaining)
    comp = None
    for c in comps:
        if side in c:
            comp = c
            break
    if comp is None:
        return best_path
    sub_zone = comp | {a, b}
    sub_path = strategy_random_backtrack(sub_zone, a, b, time_limit=time_limit, live=False)
    if not sub_path:
        return best_path
    # splice
    new_path = best_path[:mid_idx] + sub_path + best_path[mid_idx+1:]
    return new_path

#region: fill adjacent component by replacing an edge with a sub-path through the hole

def fill_adjacent_component(zone, best_path, time_limit=3.0):
    """Replace a path edge with a sub-path that traverses an adjacent component.

    Scans edges (p,q) where both endpoints touch the same remaining component,
    builds sub_zone with that component plus {p, q}, and runs
    strategy_random_backtrack to fill it. Returns spliced path or original.
    """
    visited = set(best_path)
    remaining = zone - visited
    if not remaining:
        return best_path
    comps = components(remaining)
    for i in range(len(best_path) - 1):
        p = best_path[i]; q = best_path[i+1]
        neigh_p = [v for v in neighbors(p) if v in remaining]
        neigh_q = [v for v in neighbors(q) if v in remaining]
        if not (neigh_p and neigh_q):
            continue
        for comp in comps:
            if any(v in comp for v in neigh_p) and any(v in comp for v in neigh_q):
                sub_zone = comp | {p, q}
                sub_path = strategy_random_backtrack(sub_zone, p, q, time_limit=time_limit, live=False)
                if sub_path and len(sub_path) > 2:
                    return best_path[:i] + sub_path + best_path[i+1:]
    return best_path

def iter_split_and_fill(zone, path, max_loops=10, time_limit=2.0):
    """Iteratively refine a path by applying fill_adjacent_component and split_segment_fill.

    Runs up to max_loops iterations, measuring coverage improvement (unique
    cells) between iterations; stops when no gain. Returns the refined path.
    """
    best = path
    for _ in range(max_loops):
        before = len(set(best))
        best = fill_adjacent_component(zone, best, time_limit=time_limit)
        best = split_segment_fill(zone, best, time_limit=time_limit)
        after = len(set(best))
        if after <= before:
            break
    return best

def plot_path(zone, start, end, path, filename):
    """Save a scatter/line plot of the zone and a path to `filename`.

    Uses matplotlib to draw zone cells (light gray), the path (blue), and
    start/end markers (green/red). Closes figure after saving. No return.
    """
    xs = [p[0] for p in zone]; ys = [p[1] for p in zone]
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, c="lightgray", s=5, marker="s")
    if path:
        px = [p[0] for p in path]; py = [p[1] for p in path]
        plt.plot(px, py, color="blue", linewidth=1)
        plt.plot(px[0], py[0], "go"); plt.plot(px[-1], py[-1], "ro")
    plt.axis("equal"); plt.title("Test path")
    plt.savefig(filename, dpi=150); plt.close()

def run_shape(zone, start, end, label, snapshot_prefix = None):
    """Run the full coverage pipeline on a specific zone shape and save stage plots.

    Seeds RNG for reproducibility, sets a time limit tl scaled by zone size,
    runs strategy_random_backtrack (stage 1 snapshot), then recursive_fill with
    pre/post recursion snapshots per detour, and finally saves a final plot.
    """
    sys.stdout.write(f"[run] {label} start zone={len(zone)} start={start} end={end}\n"); sys.stdout.flush()
    random.seed(42)
    # time limit scaled by zone size but capped to fit harness
    tl = min(12.0, max(6.0, len(zone) / 20000))
    base_path = strategy_random_backtrack(zone, start, end, time_limit=tl, plateau_seconds=5, live=False)
    # stage 1 snapshot: initial fill
    plot_path(zone, start, end, base_path, f"test_path_{label}_stage1.png")
    path = recursive_fill(zone, start, end, base_path, time_limit=tl, live=False,
                          snapshot_prefix=f"test_path_{label}_recur", snap_counter=[0])
    cov = len(set(path)) if path else 0
    print(f"[{label}] Final coverage: {cov}/{len(zone)} = {100.0*cov/len(zone):.1f}%")
    # stage 2 snapshot: final path
    plot_path(zone, start, end, path, f"test_path_{label}_final.png")
    return path
def main():
    """Demo entrypoint: build sample zones, run the backtracking pipeline, and save stage plots.

    Creates three zones (S-shape, ring, square), defines start/end pairs, and
    calls run_shape() on each. Produces coverage stats on stdout and saves
    stage/final PNGs via plot_path. Intended for manual inspection of the
    algorithms; no return value. Old stage images matching test_path_* are
    cleared at startup.
    """
    for f in Path(".").glob("test_path_*_*.png"):
        try:
            f.unlink()
        except Exception:
            pass
    # # S shape
    # s_w, s_h = 200, 120
    # s_zone = make_s_shape(s_w, s_h, thickness=10)
    # run_shape(s_zone, start=(0, 5), end=(s_w-1, s_h-1), label="s_shape", snapshot_prefix="s_shape_log")
    # Ring
    cx, cy = 250, 250
    ring_zone = make_ring(center=(cx, cy), inner_r=5, outer_r=15)
    run_shape(ring_zone, start=(cx - 15, cy), end=(cx + 15, cy), label="ring", snapshot_prefix="ring_log")
    # # Large square
    # size = 100
    # sq_zone = make_square_zone(size)
    # # pseudo-random reproducible start/end
    # start = (size//4, size//3)
    # end = (size - size//5, size//2)
    # run_shape(sq_zone, start=start, end=end, label="square", snapshot_prefix="square_log")

if __name__ == "__main__":
    main()
