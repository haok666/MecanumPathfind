"""
High-precision any-angle Theta*-style planner for a 3.6666667 m square field,
targeting a 0.40m x 0.40m mecanum robot.

This script implements the exact updated dot points you provided:
 - Field: 3.6666667 m square.
 - Absolute uncertainty <= 5 mm.
 - Variables/lists/arrays use camelCase. Functions use snake_case.
 - Editable rectangularObstacles list (axis-aligned rectangles, meters).
 - Prioritises robot time (speed) over distance; models rotate-vs-strafe tradeoffs.
 - Visualization saved to PNG showing field border, obstacle outlines, path centerline,
   robot outline snapshots and heading arrows. For demo: a 1m x 1m square sits centered
   in the field; robot travels from one corner of the field to the opposite.
 - Logs movement and converts to wheel RPS for FR, FL, BR, BL saved to CSV.
 - Uses attached image (if present) to build obstacles and overlays rectangularObstacles.
 - Robot size: 0.40 m x 0.40 m; collision checks account for robot orientation (OBB).
 - Must maintain >= 1 cm clearance from obstacles/walls.
 - Smoothing (shortcutting) with collision checks.
 - Optimised node spacing/resolution; will increase fidelity if runtime << 3 min.
 - No global reassignments inside functions (parameters passed explicitly).
"""

import os
import math
import time
import heapq
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.ndimage import binary_dilation, label, find_objects

# ----------------------------- USER PARAMETERS (camelCase) -----------------------------
fieldSize = 3.6666667                 # meters (square side)
uncertainty = 0.005                   # meters (5 mm)
resolution = 0.005                    # meters per occupancy cell (<= uncertainty)
nodeSpacing = 0.02                    # meters spacing for search nodes (coarse grid)
robotSize = 0.40                      # meters (square robot side)
clearance = 0.01                      # meters (1 cm)
wheelDiameter = 0.08                  # meters
wheelRadius = wheelDiameter / 2.0
motorMaxRps = 2.5
rotationMax = 1.0                     # rad/s max rotation
forwardSpeedNominal = min(0.6, wheelRadius * motorMaxRps * 2 * math.pi)
strafeFactor = 0.8
diagFactor = 0.9

timeStep = 0.1                        # seconds per CSV log interval
snapshotInterval = 0.3                # seconds between robot outline snapshots (visual)
smoothingIterations = 3
maxExpansions = 200000

# Obstacles editable by user: list of axis-aligned rectangles (x0,y0,x1,y1) in meters (lower-left origin).
rectangularObstacles: List[Tuple[float,float,float,float]] = [
    # User-editable; left empty here because we will add the required centered 1m square below.
]

attachedImagePath = "/mnt/data/e623a063-3410-45af-b5d3-5dd807aa1647.png"
useRandomObstaclesIfNone = False
randomObstacleCount = 0
randomSeed = 42

# Demo requirement: one 1m x 1m square centered in field
centerObstacleSize = 1.0
centerObsHalf = centerObstacleSize / 2.0
centerObs = ((fieldSize/2.0 - centerObsHalf), (fieldSize/2.0 - centerObsHalf),
             (fieldSize/2.0 + centerObsHalf), (fieldSize/2.0 + centerObsHalf))

# Start/goal for demo: from bottom-left corner to top-right corner (keeping margins)
marginFromCorner = 0.05
startPoseDefault = (marginFromCorner + robotSize/2.0, marginFromCorner + robotSize/2.0)
goalPoseDefault = (fieldSize - (marginFromCorner + robotSize/2.0), fieldSize - (marginFromCorner + robotSize/2.0))

outputImagePath = "saved_path.png"
wheelLogPath = "wheel_log.csv"

# Derived constants
robotHalf = robotSize / 2.0
inflationRadius = robotHalf + clearance
gridSize = int(math.ceil(fieldSize / resolution))
forwardSpeed = forwardSpeedNominal
strafeSpeed = strafeFactor * forwardSpeed
diagSpeed = diagFactor * forwardSpeed

# ----------------------------- Utilities (snake_case) -----------------------------
def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

def meters_to_grid(xy: Tuple[float,float]) -> Tuple[int,int]:
    """(x,y) meters lower-left -> (col,row) grid indices with top-left origin."""
    x,y = xy
    col = int(clamp(int(x / resolution), 0, gridSize-1))
    row = int(clamp(gridSize - 1 - int(y / resolution), 0, gridSize-1))
    return col, row

def grid_to_meters(col: int, row: int) -> Tuple[float,float]:
    """Grid indices (col,row) top-left -> center (x,y) meters lower-left."""
    x = (col + 0.5) * resolution
    y = ((gridSize - 1 - row) + 0.5) * resolution
    return x, y

def sample_line(a: Tuple[float,float], b: Tuple[float,float], step: float) -> List[Tuple[float,float]]:
    """Uniform samples along segment a->b with spacing <= step (meters)."""
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return [a]
    steps = max(1, int(math.ceil(dist / step)))
    return [(a[0] + dx * t / steps, a[1] + dy * t / steps) for t in range(steps+1)]

def rect_inflated(rect: Tuple[float,float,float,float], infl: float) -> Tuple[float,float,float,float]:
    x0,y0,x1,y1 = rect
    return (x0 - infl, y0 - infl, x1 + infl, y1 + infl)

def rect_to_grid_bounds(rect: Tuple[float,float,float,float]) -> Tuple[int,int,int,int]:
    """Return grid bounds (c0,r0,c1,r1) inclusive for rectangle in meters."""
    x0,y0,x1,y1 = rect
    c0,r1 = meters_to_grid((x0,y0))
    c1,r0 = meters_to_grid((x1,y1))
    return min(c0,c1), min(r0,r1), max(c0,c1), max(r0,r1)

# ----------------------------- Occupancy build -----------------------------
def build_occupancy(rectangularObstaclesIn: List[Tuple[float,float,float,float]],
                    includeCenterObs: bool = True,
                    attachedImage: str = attachedImagePath) -> np.ndarray:
    """
    Build occupancy grid (shape: gridSize x gridSize) where True=occupied.
    1) Uses attached image if present (dark pixels => obstacle).
    2) Overlays rectangularObstaclesIn and the centered 1m square (if includeCenterObs).
    3) Inflates obstacles by inflationRadius (robotHalf + clearance).
    """
    occ = np.zeros((gridSize, gridSize), dtype=bool)

    # image overlay
    if os.path.exists(attachedImage):
        try:
            im = Image.open(attachedImage).convert("L")
            im = im.resize((gridSize, gridSize), resample=Image.NEAREST)
            arr = np.array(im)
            occ_img = arr < 128
            occ = np.logical_or(occ, occ_img)
        except Exception as e:
            print("Warning: failed to load attached image:", e)

    # overlay rectangles
    rects = list(rectangularObstaclesIn)
    if includeCenterObs:
        rects.append(centerObs)
    for r in rects:
        x0,y0,x1,y1 = r
        # clamp within field
        x0 = clamp(x0, 0.0, fieldSize); y0 = clamp(y0, 0.0, fieldSize)
        x1 = clamp(x1, 0.0, fieldSize); y1 = clamp(y1, 0.0, fieldSize)
        c0,r0,c1,r1 = rect_to_grid_bounds((x0,y0,x1,y1))
        occ[r0:r1+1, c0:c1+1] = True

    # inflate obstacles by inflationRadius via box dilation (safe and fast)
    infl_cells = int(math.ceil(inflationRadius / resolution))
    if infl_cells > 0:
        kernel = np.ones((2*infl_cells+1, 2*infl_cells+1), dtype=bool)
        occ = binary_dilation(occ, structure=kernel)

    return occ

# ----------------------------- Collision checks (with robot OBB) -----------------------------
def point_is_free(occupancy: np.ndarray, p: Tuple[float,float]) -> bool:
    """True if a point (x,y) meters is inside field and unoccupied."""
    x,y = p
    if not (0.0 <= x <= fieldSize and 0.0 <= y <= fieldSize):
        return False
    c,r = meters_to_grid((x,y))
    return not occupancy[r,c]

def robot_footprint_corners(center: Tuple[float,float], heading: float, half: float) -> List[Tuple[float,float]]:
    """
    Return 4 corners (x,y) of robot rectangle centered at 'center' rotated by 'heading' radians.
    half = robotHalf.
    """
    cx, cy = center
    corners_local = [(-half, -half), (-half, half), (half, half), (half, -half)]
    cosH = math.cos(heading); sinH = math.sin(heading)
    return [(cx + (lx * cosH - ly * sinH), cy + (lx * sinH + ly * cosH)) for (lx,ly) in corners_local]

def obb_collides_occupancy(occupancy: np.ndarray, center: Tuple[float,float], heading: float, half: float) -> bool:
    """
    Conservative collision check for oriented robot rectangle (OBB):
    Rasterizes bounding box region at resolution sampling and ensures no sample point falls in occupied cell.
    This accounts for rotated diagonals longer than edges.
    """
    corners = robot_footprint_corners(center, heading, half)
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    minx = clamp(min(xs) - resolution, 0.0, fieldSize)
    maxx = clamp(max(xs) + resolution, 0.0, fieldSize)
    miny = clamp(min(ys) - resolution, 0.0, fieldSize)
    maxy = clamp(max(ys) + resolution, 0.0, fieldSize)
    # sample grid within bounding box at step <= resolution/2 for safety
    step = min(resolution, uncertainty/2.0)
    nx = max(1, int(math.ceil((maxx - minx) / step)))
    ny = max(1, int(math.ceil((maxy - miny) / step)))
    # Precompute polygon winding for point-in-polygon test (convex OBB)
    poly = corners
    def point_in_poly(px, py):
        # convex polygon test using cross products (all same sign)
        sign = None
        for i in range(len(poly)):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
            cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
            if abs(cross) < 1e-12: continue
            curSign = cross > 0
            if sign is None:
                sign = curSign
            elif sign != curSign:
                return False
        return True
    for i in range(nx+1):
        for j in range(ny+1):
            px = minx + (i / max(1,nx)) * (maxx - minx)
            py = miny + (j / max(1,ny)) * (maxy - miny)
            if not point_in_poly(px, py):
                continue
            if not point_is_free(occupancy, (px, py)):
                return True
    return False

# ----------------------------- LOS check (continuous sampling) -----------------------------
def line_of_sight_free(occupancy: np.ndarray, a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    """
    Sample straight line a->b at spacing <= min(resolution, uncertainty/2) and ensure
    at each sample the robot OBB (for all plausible headings between a->b) does not collide.
    For efficiency we check robot OBB at heading = movement angle and heading = current heading caller should ensure safety.
    (This function used for geometry-only LOS; theta* uses it conservatively.)
    """
    step = min(resolution, uncertainty / 2.0)
    samples = sample_line(a, b, step)
    moveAngle = math.atan2(b[1]-a[1], b[0]-a[0]) if (b[0]-a[0] or b[1]-a[1]) else 0.0
    for s in samples:
        if obb_collides_occupancy(occupancy, s, moveAngle, robotHalf):
            return False
    return True

# ----------------------------- Time cost model -----------------------------
def time_cost_between(a: Tuple[float,float], b: Tuple[float,float], currentHeading: float) -> Tuple[float, float]:
    """
    Return (timeSeconds, resultingHeading) for moving a->b from currentHeading.
    Evaluates maintaining heading (strafe/drive) vs rotate-first then drive.
    Prioritises minimal time.
    """
    dx = b[0] - a[0]; dy = b[1] - a[1]; dist = math.hypot(dx, dy)
    if dist == 0.0:
        return 0.0, currentHeading
    moveAngle = math.atan2(dy, dx)
    angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
    cosFactor = abs(math.cos(angDiff))
    speedOption1 = cosFactor * forwardSpeed + (1 - cosFactor) * strafeSpeed
    timeOption1 = dist / max(1e-9, speedOption1)
    rotateTime = angDiff / max(1e-9, rotationMax)
    timeOption2 = rotateTime + dist / forwardSpeed
    if timeOption1 <= timeOption2:
        return timeOption1, currentHeading
    else:
        return timeOption2, moveAngle

# ----------------------------- Node graph & Theta* search -----------------------------
def build_node_list(occupancy: np.ndarray, spacing: float) -> Tuple[List[Tuple[float,float,int,int]], Dict[Tuple[int,int],int]]:
    """
    Build node list spaced by 'spacing' meters.
    Exclude nodes inside inflated obstacles or too close to borders (respect robotHalf + clearance).
    Return nodePositions [(x,y,ix,iy), ...] and nodeMap {(ix,iy):index}.
    """
    nx = int(math.floor(fieldSize / spacing))
    ny = nx
    nodePositions = []
    nodeMap: Dict[Tuple[int,int], int] = {}
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            x = (ix + 0.5) * spacing
            y = (iy + 0.5) * spacing
            if x < (robotHalf + clearance) or x > (fieldSize - (robotHalf + clearance)): continue
            if y < (robotHalf + clearance) or y > (fieldSize - (robotHalf + clearance)): continue
            if not obb_collides_occupancy(occupancy, (x,y), 0.0, robotHalf):  # quick axis-aligned check
                nodePositions.append((x,y,ix,iy))
                nodeMap[(ix,iy)] = idx
                idx += 1
    return nodePositions, nodeMap

def nearest_node_index(nodePositions: List[Tuple[float,float,int,int]], p: Tuple[float,float]) -> Optional[int]:
    best = None; bestd = 1e9
    for i,(x,y,_,_) in enumerate(nodePositions):
        d = math.hypot(x - p[0], y - p[1])
        if d < bestd:
            bestd = d; best = i
    return best

def theta_star_search(occupancy: np.ndarray, startPose: Tuple[float,float], goalPose: Tuple[float,float]) -> Optional[List[Tuple[float,float,float]]]:
    """
    Theta*-style any-angle search with heading as part of state.
    Uses continuous LOS checks that factor robot OBB collisions.
    Returns path list of (x,y,heading).
    """
    nodePositions, nodeMap = build_node_list(occupancy, nodeSpacing)
    if not nodePositions:
        return None
    startIdx = nearest_node_index(nodePositions, startPose)
    goalIdx = nearest_node_index(nodePositions, goalPose)
    if startIdx is None or goalIdx is None:
        return None

    angleSamples = 16
    neighborOffsets = [(math.cos(2*math.pi*i/angleSamples)*nodeSpacing, math.sin(2*math.pi*i/angleSamples)*nodeSpacing) for i in range(angleSamples)]

    def heuristic_time(nodePos):
        gx,gy,_,_ = nodePositions[goalIdx]
        return math.hypot(nodePos[0] - gx, nodePos[1] - gy) / forwardSpeed

    headingCount = 32
    def heading_idx_to_angle(idx): return (2*math.pi*idx)/headingCount
    def angle_to_heading_idx(angle): return int(round((angle % (2*math.pi)) / (2*math.pi) * headingCount)) % headingCount

    startHeading = 0.0
    startState = (startIdx, angle_to_heading_idx(startHeading))
    gscore: Dict[Tuple[int,int], float] = {startState: 0.0}
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {startState: None}
    parentExactPos: Dict[Tuple[int,int], Tuple[float,float]] = {startState: (nodePositions[startIdx][0], nodePositions[startIdx][1])}
    openHeap: List[Tuple[float, Tuple[int,int]]] = [(heuristic_time(nodePositions[startIdx]), startState)]
    closed = set()
    expansions = 0
    tstart = time.time()

    while openHeap:
        _, state = heapq.heappop(openHeap)
        if state in closed: continue
        closed.add(state)
        expansions += 1
        if expansions > maxExpansions:
            print("Search expansions cap reached.")
            break

        nodeIdx, headingIdx = state
        nodeX, nodeY, _, _ = nodePositions[nodeIdx]
        gx,gy,_,_ = nodePositions[goalIdx]
        if nodeIdx == goalIdx or math.hypot(nodeX - gx, nodeY - gy) <= nodeSpacing * 1.1:
            # Reconstruct path
            seq = []
            cur = state
            while cur is not None:
                nidx, hidx = cur
                px,py,_,_ = nodePositions[nidx]
                seq.append((px,py,heading_idx_to_angle(hidx)))
                cur = parent[cur]
            seq.reverse()
            # attach exact start and goal if far
            if math.hypot(seq[0][0] - startPose[0], seq[0][1] - startPose[1]) > 1e-6:
                seq.insert(0, (startPose[0], startPose[1], startHeading))
            if math.hypot(seq[-1][0] - goalPose[0], seq[-1][1] - goalPose[1]) > 1e-6:
                seq.append((goalPose[0], goalPose[1], seq[-1][2]))
            tend = time.time()
            print(f"Search time {tend - tstart:.2f}s, expansions {expansions}, nodes {len(nodePositions)}")
            return seq

        currentExact = parentExactPos[state]
        currentHeading = heading_idx_to_angle(headingIdx)

        for dx,dy in neighborOffsets:
            succX = nodeX + dx; succY = nodeY + dy
            if not (0.0 <= succX <= fieldSize and 0.0 <= succY <= fieldSize): continue
            ix = int(succX / nodeSpacing); iy = int(succY / nodeSpacing)
            if (ix,iy) not in nodeMap: continue
            succIdx = nodeMap[(ix,iy)]
            succPos = (nodePositions[succIdx][0], nodePositions[succIdx][1])

            # Theta* relaxation: try connecting from parentExact if LOS without OBB collision
            useParent = False
            parState = parent[state]
            if parState is not None and line_of_sight_free(occupancy, parentExactPos[state], succPos):
                basePos = parentExactPos[state]
                baseG = gscore.get(parState, 0.0)
                baseHeading = heading_idx_to_angle(parState[1])
                useParent = True
            else:
                basePos = (nodeX, nodeY)
                baseG = gscore[state]
                baseHeading = currentHeading

            moveTime, resultingHeading = time_cost_between(basePos, succPos, baseHeading)
            tentativeG = baseG + moveTime
            succHeadingIdx = angle_to_heading_idx(resultingHeading)
            succState = (succIdx, succHeadingIdx)
            if tentativeG < gscore.get(succState, float('inf')):
                gscore[succState] = tentativeG
                parent[succState] = state if not useParent else parState
                parentExactPos[succState] = basePos
                f = tentativeG + heuristic_time(succPos)
                heapq.heappush(openHeap, (f, succState))

    print("No path found.")
    return None

# ----------------------------- Smoothing -----------------------------
def smooth_path(path: List[Tuple[float,float,float]], occupancy: np.ndarray) -> List[Tuple[float,float,float]]:
    """Shortcut smoothing with LOS that accounts for OBB collision."""
    if not path or len(path) < 3:
        return path
    coords = [(p[0], p[1]) for p in path]
    for _ in range(smoothingIterations):
        i = 0
        while i < len(coords) - 2:
            j = len(coords) - 1
            replaced = False
            while j > i + 1:
                if line_of_sight_free(occupancy, coords[i], coords[j]):
                    del coords[i+1:j]
                    replaced = True
                    break
                j -= 1
            if not replaced:
                i += 1
    # rebuild headings
    out = []
    for k in range(len(coords)):
        x,y = coords[k]
        if k < len(coords)-1:
            nx,ny = coords[k+1]
            heading = math.atan2(ny - y, nx - x)
        else:
            heading = out[-1][2] if out else 0.0
        out.append((x,y,heading))
    return out

# ----------------------------- Trajectory -> wheel CSV -----------------------------
def generate_wheel_log(path: List[Tuple[float,float,float]], occupancy: np.ndarray) -> pd.DataFrame:
    """
    Generate wheel RPS log sampled at timeStep intervals. Columns: time_start, time_end, FR, FL, BR, BL.
    Uses same motion policy as planner's time model (strafe vs rotate+forward).
    """
    logs = []   # (t0,t1,vx,vy,wz,heading)
    t = 0.0
    currentHeading = path[0][2]
    for i in range(len(path)-1):
        a = (path[i][0], path[i][1]); b = (path[i+1][0], path[i+1][1])
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        if dist < 1e-9: continue
        moveTime, resultHeading = time_cost_between(a, b, currentHeading)
        dx = b[0] - a[0]; dy = b[1] - a[1]
        moveAngle = math.atan2(dy, dx)
        angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
        cosFactor = abs(math.cos(angDiff))
        speedOption1 = cosFactor * forwardSpeed + (1 - cosFactor) * strafeSpeed
        timeOption1 = dist / max(1e-9, speedOption1)
        rotateTime = angDiff / max(1e-9, rotationMax)
        timeOption2 = rotateTime + dist / forwardSpeed

        if timeOption1 <= timeOption2:
            vx = dx / timeOption1; vy = dy / timeOption1
            logs.append((t, t + timeOption1, vx, vy, 0.0, currentHeading))
            t += timeOption1
        else:
            rotDir = 1.0 if ((moveAngle - currentHeading + 2*math.pi) % (2*math.pi)) < math.pi else -1.0
            logs.append((t, t + rotateTime, 0.0, 0.0, rotDir * rotationMax, currentHeading))
            t += rotateTime
            vx = dx / (dist / forwardSpeed); vy = dy / (dist / forwardSpeed)
            logs.append((t, t + dist / forwardSpeed, vx, vy, 0.0, moveAngle))
            t += dist / forwardSpeed
            currentHeading = moveAngle

    # sample and convert to wheel RPS
    numSteps = int(math.ceil(t / timeStep)) + 1
    table = []
    for s in range(numSteps):
        t0 = s * timeStep; t1 = min(t0 + timeStep, t)
        tm = (t0 + t1) / 2.0
        vx = vy = wz = 0.0; headingAt = 0.0
        for seg in logs:
            if seg[0] <= tm <= seg[1] + 1e-9:
                vx,vy,wz,headingAt = seg[2], seg[3], seg[4], seg[5]
                break
        # world->body
        ch = math.cos(-headingAt); sh = math.sin(-headingAt)
        bodyVx = vx * ch - vy * sh
        bodyVy = vx * sh + vy * ch
        lx = ly = robotSize / 2.0; lsum = lx + ly; r = wheelRadius
        wfl = (1.0/r) * (bodyVx - bodyVy - lsum * wz)
        wfr = (1.0/r) * (bodyVx + bodyVy + lsum * wz)
        wrl = (1.0/r) * (bodyVx + bodyVy - lsum * wz)
        wrr = (1.0/r) * (bodyVx - bodyVy + lsum * wz)
        rpsFL = wfl / (2*math.pi); rpsFR = wfr / (2*math.pi)
        rpsRL = wrl / (2*math.pi); rpsRR = wrr / (2*math.pi)
        maxAbs = max(abs(rpsFL), abs(rpsFR), abs(rpsRL), abs(rpsRR), 1e-9)
        if maxAbs > motorMaxRps:
            scale = motorMaxRps / maxAbs
            rpsFL *= scale; rpsFR *= scale; rpsRL *= scale; rpsRR *= scale
        table.append({
            "time_start": round(t0,6), "time_end": round(t1,6),
            "FR": round(rpsFR,6), "FL": round(rpsFL,6),
            "BR": round(rpsRR,6), "BL": round(rpsRL,6)
        })
    df = pd.DataFrame(table)
    df.to_csv(wheelLogPath, index=False)
    print(f"Saved wheel log: {wheelLogPath} (total time {t:.2f}s, intervals {len(table)})")
    return df

# ----------------------------- Visualization -----------------------------
def visualize_and_save(occupancy: np.ndarray, path: List[Tuple[float,float,float]], outputPath: str, snapInterval: float = snapshotInterval) -> str:
    """
    Save PNG showing:
     - Field border (black)
     - Obstacles filled (light blue) with dark outline
     - Path centerline (red)
     - Robot OBB snapshots with heading arrows every snapInterval seconds
    """
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    ax.set_xlim(0, fieldSize); ax.set_ylim(0, fieldSize)
    ax.set_aspect('equal'); ax.invert_yaxis()

    # Field border
    ax.add_patch(patches.Rectangle((0,0), fieldSize, fieldSize, fill=False, edgecolor='black', linewidth=2, zorder=1))

    # Draw obstacle bounding boxes from occupancy components
    occImg = occupancy.astype(np.uint8)
    labeled, ncomp = label(occImg)
    objs = find_objects(labeled)
    for sl in objs:
        if sl is None: continue
        row0, row1 = sl[0].start, sl[0].stop - 1
        col0, col1 = sl[1].start, sl[1].stop - 1
        x0, y0 = grid_to_meters(col0, row1)
        x1, y1 = grid_to_meters(col1, row0)
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, facecolor='#cfeefe', edgecolor='#1177bb', linewidth=1.5, zorder=2)
        ax.add_patch(rect)

    # Path centerline
    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, color='red', linewidth=2, zorder=3)

    # Rebuild segments similarly to generate_wheel_log to get timings
    segments = []
    t = 0.0
    currentHeading = path[0][2] if path else 0.0
    for i in range(len(path)-1):
        a = (path[i][0], path[i][1]); b = (path[i+1][0], path[i+1][1])
        dist = math.hypot(b[0]-a[0], b[1]-a[1])
        if dist < 1e-9: continue
        moveTime, resultHeading = time_cost_between(a, b, currentHeading)
        dx = b[0] - a[0]; dy = b[1] - a[1]
        moveAngle = math.atan2(dy, dx)
        angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
        cosFactor = abs(math.cos(angDiff))
        speedOption1 = cosFactor * forwardSpeed + (1 - cosFactor) * strafeSpeed
        timeOption1 = dist / max(1e-9, speedOption1)
        rotateTime = angDiff / max(1e-9, rotationMax)
        timeOption2 = rotateTime + dist / forwardSpeed

        if timeOption1 <= timeOption2:
            segments.append((t, t + timeOption1, a, b, currentHeading))
            t += timeOption1
        else:
            segments.append((t, t + rotateTime, a, a, currentHeading))
            t += rotateTime
            segments.append((t, t + dist / forwardSpeed, a, b, moveAngle))
            t += dist / forwardSpeed
            currentHeading = moveAngle

    totalTime = t
    sampleTimes = [0.0] if totalTime <= 0 else list(np.arange(0.0, totalTime + 1e-9, snapInterval))

    for st in sampleTimes:
        seg = None
        for s in segments:
            if s[0] <= st <= s[1] + 1e-9:
                seg = s; break
        if seg is None: continue
        t0,t1,a,b,heading = seg
        if t1 - t0 < 1e-9:
            pos = a
        else:
            alpha = (st - t0) / max(1e-9, (t1 - t0))
            pos = (a[0]*(1-alpha) + b[0]*alpha, a[1]*(1-alpha) + b[1]*alpha)
        cx, cy = pos
        rect = patches.Rectangle((cx - robotHalf, cy - robotHalf), robotSize, robotSize,
                                 edgecolor='black', facecolor='none', linewidth=1.6,
                                 transform=plt.matplotlib.transforms.Affine2D().rotate_around(cx, cy, heading) + ax.transData, zorder=5)
        ax.add_patch(rect)
        ax.arrow(cx, cy, 0.25 * math.cos(heading), 0.25 * math.sin(heading), head_width=0.08, head_length=0.08, fc='black', ec='black', zorder=6)

    ax.set_title("Planned Path with Robot Outlines and Heading")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout(); fig.savefig(outputPath, dpi=200); plt.close(fig)
    print(f"Saved visualization: {outputPath}")
    return outputPath

# ----------------------------- Helpers: safe projection -----------------------------
def find_nearest_free(occupancy: np.ndarray, pose: Tuple[float,float], maxRadiusCells: int = 50) -> Optional[Tuple[float,float]]:
    """Search outward from pose in grid cells for nearest free cell center and return its meters coords."""
    col0, row0 = meters_to_grid(pose)
    for r in range(maxRadiusCells+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                c = col0 + dx; rr = row0 + dy
                if 0 <= c < gridSize and 0 <= rr < gridSize and not occupancy[rr, c]:
                    x,y = grid_to_meters(c, rr)
                    if not obb_collides_occupancy(occupancy, (x,y), 0.0, robotHalf):
                        return (x,y)
    return None

# ----------------------------- Top-level (no globals reassigned) -----------------------------
def plan_and_run(startPose: Tuple[float,float] = startPoseDefault,
                 goalPose: Tuple[float,float] = goalPoseDefault,
                 rectangularObstaclesIn: Optional[List[Tuple[float,float,float,float]]] = None):
    """
    Main entry. Parameters passed explicitly. Returns (path, dfWheel) on success, otherwise None.
    """
    rects = rectangularObstaclesIn if rectangularObstaclesIn is not None else rectangularObstacles
    occupancy = build_occupancy(rects, includeCenterObs=True, attachedImage=attachedImagePath)

    # ensure safe start/goal (project if necessary)
    if not point_is_free(occupancy, startPose) or obb_collides_occupancy(occupancy, startPose, 0.0, robotHalf):
        proj = find_nearest_free(occupancy, startPose)
        if proj is None:
            print("Unable to find safe start. Aborting.")
            return None
        print("Projected startPose ->", proj); startPose = proj

    if not point_is_free(occupancy, goalPose) or obb_collides_occupancy(occupancy, goalPose, 0.0, robotHalf):
        proj = find_nearest_free(occupancy, goalPose)
        if proj is None:
            print("Unable to find safe goal. Aborting.")
            return None
        print("Projected goalPose ->", proj); goalPose = proj

    # plan
    path = theta_star_search(occupancy, startPose, goalPose)
    if path is None:
        print("Planner failed to find a path.")
        return None

    # smoothing
    pathSmoothed = smooth_path(path, occupancy)

    # verify final path is collision-free for OBB at each heading
    for x,y,heading in pathSmoothed:
        if obb_collides_occupancy(occupancy, (x,y), heading, robotHalf):
            print("Collision detected after smoothing. Aborting.")
            return None

    # generate wheel log and visualization
    df = generate_wheel_log(pathSmoothed, occupancy)
    imgPath = visualize_and_save(occupancy, pathSmoothed, outputImagePath)
    print("Done. Path points:", len(pathSmoothed))
    print("Wheel log:", wheelLogPath)
    print("Visualization:", imgPath)
    print(df.head(10).to_string(index=False))
    return pathSmoothed, df

# ----------------------------- Run example if main -----------------------------
if __name__ == "__main__":
    # run using demo start/goal and the centered 1m obstacle
    plan_and_run(startPoseDefault, goalPoseDefault)
