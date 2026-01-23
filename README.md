# Maze Creation From Image

Turn a black-and-white logo into a maze-like coverage path. The pipeline thresholds an image into a grid, finds connected components, links them with bridges, and runs a coverage-focused path planner on every component before assembling one continuous route.

## Why
- **Initial idea**: carve a complex maze that visually follows a logo while still being solvable as a single walkable path.
- **Reasoning**: treat the logo pixels as a grid graph, build high-coverage (near-Hamiltonian) walks inside each connected component, then connect components with bridges so the final path is continuous.
- **Outcome**: a set of text exports (`maze_path.txt`, `maze_full.txt`, `maze_rect.txt`) and plots (`assembly.png`, `zone_path_*.png`, etc.) that mirror the input shape.

## Quick start
1) Create a virtualenv (recommended) and install deps:
```
pip install -r requirements.txt
```
2) Run the pipeline (positional args are optional, defaults in brackets):
```
python maze_creation_from_image.py path/to/logo.png [target_width=200] [mode=nearest|radial] [curl=true|false]
```
   - `target_width` resizes the input; height is scaled to preserve aspect ratio.
   - `mode` chooses component ordering (`nearest` greedy or `radial` sweep).
   - `curl` toggles a turn-biased backtracker (good for avoiding long straights).
3) Inspect outputs in the current folder (plots + text exports). Use `zones_data.npz` or `zones_links.json` for programmatic consumption.

## Files in this folder
- `maze_creation_from_image.py`: main CLI that loads the image, builds the grid mask, orders components, links them, generates per-zone coverage paths, and exports visuals and maze files.
- `cover_path.py`: coverage backtracker + recursive fill used for dense zone traversal.
- `curl_cover_path.py`: variant that biases turns to curl around holes before straightening.
- `requirements.txt`: minimal runtime dependencies (`numpy`, `Pillow`, `matplotlib`).

## Pipeline (what the script does)
1) **Image -> mask**: load RGBA, resize to `target_width`, threshold alpha/RGB to keep dark pixels, and flip vertically so y=0 is bottom-left.
2) **Connected components**: BFS flood-fill to extract zones (sets of `(x, y)` grid cells) and their boolean masks.
3) **Ordering**: either nearest-neighbor walk over centroids (`nearest`) or angular sweep around the global centroid (`radial`).
4) **Bridging zones**: find boundary cells per zone, then BFS through the background to build the shortest bridge between consecutive zones (last bridge closes the loop). These define per-zone start/end anchors.
5) **Per-zone coverage paths**: for each ordered zone, run a randomized DFS/backtracking solver (`cover_path` or `curl_cover_path`) with plateau detection and recursion to fill holes. The solver is goal-directed (start->end) but maximizes distinct visited cells.
6) **Assembly**: concatenate all zone paths with the bridges (minus the closing bridge) to form a single global path; remove duplicates/self-crossings.
7) **Exports**:
   - Plots: `zone_*.png`, `zone_order.png`, `zone_links.png`, `zone_path_*.png`, `zone_paths_combined.png`, `zone_zoom.png`, `assembly.png`.
   - Text: `maze_path.txt` (path adjacency), `maze_full.txt` (path as maze), `maze_rect.txt` + `maze_rect_added_edges.txt` (rectangular maze with tree-like branches).
   - Data: `zones_data.npz`, `zones_links.json` with starts/ends, bridges, coverage stats.

## Algorithmic notes
- Coverage per zone is a **self-avoiding randomized DFS/backtracking** with degree/turn heuristics, plateau stopping, and recursive re-runs on holes (components of unvisited cells). A shortest-path tail guarantees connectivity to the zone end.
- Component ordering uses a **greedy nearest-neighbor** heuristic or **radial sweep**; both are cheap and work well for clustered logos.
- Bridges are found with **BFS on the background** with a Manhattan fallback when disconnected.
- Optional rectangular maze export attaches a **spanning-tree-like carving** to the main path while keeping endpoints low-degree.
- These heuristics chase high coverage on grid graphs akin to Hamiltonian paths; they are not guaranteed optimal but are fast and reproducible (seeded).

## Inputs and expectations
- Use high-contrast images (dark logo on transparent/white background). Thin strokes may disappear after resizing; adjust `target_width` upward to retain detail.
- The mask is binary; gradients are not preserved. If you need more fidelity, pre-binarize the logo yourself.
- Runtime scales with pixel count; large logos at big widths can take minutes because each zone runs its own DFS/backtracking.

## Legal and IP considerations
- Only process artwork you **own or are licensed to use**. Logos can be protected by copyright and trademark; derivative maze shapes may still be governed by those rights.
- If you plan to publish or sell generated mazes, ensure you have explicit permission from the rights holder. Avoid third-party marks you do not control.
- This code is provided as-is with no grant of rights to any upstream artwork; you are responsible for complying with local IP law and your asset licenses.

## Literature and similar problems
- **Hamiltonian Path / Grid Coverage**: NP-complete on grids; see Garey & Johnson, *Computers and Intractability* (1979).
- **Coverage Path Planning**: e.g., Howie Choset, “Coverage for robotics,” *Annals of Mathematics and Artificial Intelligence* (2001).
- **Chinese Postman / Route Stitching**: Edmonds & Johnson, “Matching, Euler tours and the Chinese postman,” *Math. Programming* (1973).
- **Randomized DFS and Backtracking**: classic maze-carving/coverage heuristics; see Aldous & Fill, *Reversible Markov Chains and Random Walks on Graphs* for background on random walks.
- The code here uses practical variants of these ideas (randomized DFS with heuristics, component bridging, and serpentine shoelace fills) to get high coverage quickly.

## Example commands
- Nearest ordering with turn-biased coverage (default):
```
python maze_creation_from_image.py logo.png 400 nearest true
```
- Radial ordering with straighter paths:
```
python maze_creation_from_image.py logo.png 500 radial false
```

## Tips and troubleshooting
- If coverage stalls early, raise `target_width` or set `curl=true` to encourage curling around holes.
- To debug a specific zone, inspect `zone_path_<i>.png` and `zone_zoom.png`; gaps indicate unreachable pockets.
- Outputs are written next to the script; clean stale `test_path_*` images if you re-run the included demo code in `cover_path.py`/`curl_cover_path.py`.

## Next steps
- Tune thresholds in `load_black_mask` if your artwork has semi-transparent edges.
- Swap in your own ordering heuristic or bridge builder if you need different traversal semantics.
- Add unit tests around `compute_links`, `build_zone_paths`, and `build_full_path` if you plan to extend the codebase.
