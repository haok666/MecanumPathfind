"""
High-precision Theta*-style path planner for a 3.6666667 m square field,
designed for a 0.40m x 0.40m mecanum robot. Implements:
 - Time-prioritised any-angle planning (Theta*-style LOS relaxation).
 - Continuous sampling LOS with configurable sampling step (meets 5 mm uncertainty).
 - Inflated obstacles (robot size + 1cm safety) from an editable rectangle list
   and optionally from an attached image (developer-provided).
 - Path smoothing (shortcutting with collision checks).
 - Trajectory -> wheel angular velocity table (FR, FL, BR, BL) saved to CSV.
 - Visualization saved to PNG: field border, obstacle outlines, red centerline,
   robot outline snapshots and heading arrows at intervals.
Naming conventions: variables/lists/arrays use camelCase; functions use under_score.
Run as a standalone script.
"""

import os
import math
import heapq
import time
import csv
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# ------------------------ USER PARAMETERS (camelCase) ------------------------
fieldSize = 3.6666667                # meters (square side)
uncertainty = 0.005                   # meters (max absolute uncertainty <= 5 mm)
resolution = 0.005                    # meters per occupancy cell (<= uncertainty)
nodeSpacing = 0.02                    # m spacing for search nodes (coarser than resolution)
robotSize = 0.40                      # m (robot square side)
clearance = 0.01                      # m required clearance from obstacles/walls (1 cm)
wheelDiameter = 0.08                  # m (80 mm)
wheelRadius = wheelDiameter / 2.0
motorMaxRps = 2.5                     # rotations per second
rotationMax = 1.0                     # rad/s max in-place rotation speed (tunable)
forwardSpeedNominal = min(0.6, wheelRadius * motorMaxRps * 2 * math.pi)  # m/s
strafeFactor = 0.8                    # strafing speed relative to forward
diagFactor = 0.9                      # diagonal speed relative to forward

# Logging / sampling
timeStep = 0.1                        # seconds per logged interval
snapshotInterval = 0.3                # seconds between robot-outline snapshots in visualization

# Smoothing and search caps
smoothingIterations = 3
maxExpansions = 200000

# Obstacles (editable). Format: (x_min, y_min, x_max, y_max) in meters measured from (0,0) lower-left.
# Keep rectangles axis-aligned at 90 degrees. Example: one rectangle in center:
rectangularObstacles = [
    # (0.9, 1.0, 1.6, 1.5),
]

# Use attached image (developer-provided) if present. Image assumed covering entire field.
attachedImagePath = "/mnt/data/e623a063-3410-45af-b5d3-5dd807aa1647.png"

# Random obstacles toggle (if rectangularObstacles empty, will generate some random ones for demo)
useRandomObstaclesIfNone = True
randomObstacleCount = 6
randomSeed = 42

# Start / goal (meters). Ensure they are at least robotHalf + clearance from walls/obstacles.
startPose = (0.3, 0.3)                  # (x, y) meters, center of robot
goalPose = (3.3666667, 3.3666667)       # (x, y) meters

outputImagePath = "saved_path.png"
wheelLogPath = "wheel_log.csv"

# ------------------------ Derived constants ------------------------
robotHalf = robotSize / 2.0
inflationRadius = robotHalf + clearance
gridSize = int(math.ceil(fieldSize / resolution))   # occupancy grid dimension (square)
forwardSpeed = forwardSpeedNominal
strafeSpeed = strafeFactor * forwardSpeed
diagSpeed = diagFactor * forwardSpeed

# ------------------------ Utility (snake_case) ------------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def meters_to_grid(p: Tuple[float, float]) -> Tuple[int, int]:
    """Convert (x,y) meters (origin lower-left) to occupancy indices (col, row) with origin top-left."""
    x, y = p
    col = int(x / resolution)
    row = gridSize - 1 - int(y / resolution)   # invert y for array indexing
    col = clamp(col, 0, gridSize - 1)
    row = clamp(row, 0, gridSize - 1)
    return col, row

def grid_to_meters(col: int, row: int) -> Tuple[float, float]:
    """Convert occupancy indices (col, row, top-left origin) to center (x,y) meters (lower-left origin)."""
    x = (col + 0.5) * resolution
    y = ((gridSize - 1 - row) + 0.5) * resolution
    return x, y

def sample_line(a: Tuple[float,float], b: Tuple[float,float], step: float) -> List[Tuple[float,float]]:
    """Uniformly sample points from a to b inclusive with spacing <= step (meters)."""
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return [a]
    steps = max(1, int(math.ceil(dist / step)))
    return [(a[0] + (dx * t / steps), a[1] + (dy * t / steps)) for t in range(steps + 1)]

def rect_inflated(rect, infl) -> Tuple[float,float,float,float]:
    """Return inflated rectangle (x0,y0,x1,y1)."""
    x0,y0,x1,y1 = rect
    return (x0 - infl, y0 - infl, x1 + infl, y1 + infl)

def rect_to_grid_cells(rect):
    """Return grid cell ranges (col0,row0,col1,row1) covering rect (meters)."""
    x0,y0,x1,y1 = rect
    col0, row1 = meters_to_grid((x0, y0))  # bottom-left -> col0,row1 (row1 is bottom)
    col1, row0 = meters_to_grid((x1, y1))  # top-right -> col1,row0
    # reorder
    c0 = min(col0, col1); c1 = max(col0, col1)
    r0 = min(row0, row1); r1 = max(row0, row1)
    return c0, r0, c1, r1

# ------------------------ Build occupancy ------------------------
def build_occupancy_from_image_and_rects() -> np.ndarray:
    """
    Build occupancy boolean array of shape (gridSize, gridSize).
    True => occupied.
    Uses attached image if available, overlays rectangularObstacles, inflates by inflationRadius.
    """
    occ = np.zeros((gridSize, gridSize), dtype=bool)

    # 1) Image-based obstacles (if image exists) - thresholding
    if os.path.exists(attachedImagePath):
        try:
            img = Image.open(attachedImagePath).convert("L")
            img = img.resize((gridSize, gridSize), resample=Image.NEAREST)
            arr = np.array(img)
            occ_img = arr < 128   # dark pixels -> obstacle
            occ = np.logical_or(occ, occ_img)
        except Exception as e:
            print("Image load failed, falling back to rectangles:", e)

    # 2) Overlay rectangles provided by user (meters)
    if len(rectangularObstacles) == 0 and useRandomObstaclesIfNone:
        random.seed(randomSeed)
        for _ in range(randomObstacleCount):
            w = random.uniform(0.2, 0.6)
            h = random.uniform(0.2, 0.6)
            x0 = random.uniform(0.1, fieldSize - 0.1 - w)
            y0 = random.uniform(0.1, fieldSize - 0.1 - h)
            rectangularObstacles.append((x0, y0, x0 + w, y0 + h))

    for rect in rectangularObstacles:
        x0,y0,x1,y1 = rect
        x0 = clamp(x0, 0.0, fieldSize); y0 = clamp(y0, 0.0, fieldSize)
        x1 = clamp(x1, 0.0, fieldSize); y1 = clamp(y1, 0.0, fieldSize)
        c0, r0, c1, r1 = rect_to_grid_cells((x0,y0,x1,y1))
        occ[r0:r1+1, c0:c1+1] = True

    # 3) Inflate by inflationRadius (in cells) using box dilation (safe, deterministic).
    infl_cells = int(math.ceil(inflationRadius / resolution))
    if infl_cells > 0:
        kernel = np.ones((2*infl_cells+1, 2*infl_cells+1), dtype=bool)
        # simple convolution-like dilation
        from scipy.ndimage import binary_dilation
        occ = binary_dilation(occ, structure=kernel)

    # 4) Ensure field borders are considered (but robot is centered, border inflation handled by node pruning)
    return occ

# ------------------------ Collision and LOS checks ------------------------
def point_is_free(occupancy: np.ndarray, p: Tuple[float,float]) -> bool:
    """Return True if point (x,y) in meters is free (not occupied) and inside field boundaries."""
    x,y = p
    if not (0.0 <= x <= fieldSize and 0.0 <= y <= fieldSize):
        return False
    col, row = meters_to_grid((x,y))
    return not occupancy[row, col]

def line_of_sight_free(occupancy: np.ndarray, a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    """
    Sample line between a and b using sample spacing <= min(resolution, uncertainty/2).
    Returns False if any sample point collides.
    """
    step = min(resolution, uncertainty / 2.0)
    for p in sample_line(a, b, step):
        if not point_is_free(occupancy, p):
            return False
    return True

# ------------------------ Time cost model (prioritises time) ------------------------
def time_cost_between(a: Tuple[float,float], b: Tuple[float,float], currentHeading: float) -> Tuple[float, float]:
    """
    Estimate minimal time to go from a->b given currentHeading (radians).
    Two choices evaluated:
      - strafe/move without changing heading (speed depends on angle between motion and heading)
      - rotate in place (minimal angle) then drive forward at forwardSpeed
    Returns (timeSeconds, resultingHeading).
    """
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, currentHeading
    moveAngle = math.atan2(dy, dx)
    # minimal angular difference [0, pi]
    angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
    # Option 1: move while maintaining heading
    # Speed interpolation: aligned -> forwardSpeed; perpendicular -> strafeSpeed; opposite -> forwardSpeed
    cosFactor = abs(math.cos(angDiff))
    speedOption1 = cosFactor * forwardSpeed + (1 - cosFactor) * strafeSpeed
    timeOption1 = dist / max(1e-6, speedOption1)
    # Option 2: rotate first, then drive forward
    rotateTime = angDiff / max(1e-6, rotationMax)
    timeOption2 = rotateTime + dist / forwardSpeed
    if timeOption1 <= timeOption2:
        return timeOption1, currentHeading
    else:
        return timeOption2, moveAngle

# ------------------------ Node graph + Theta*-style search ------------------------
def build_node_list(occupancy: np.ndarray, spacing: float) -> Tuple[List[Tuple[float,float,int,int]], Dict[Tuple[int,int], int]]:
    """
    Build list of candidate node centers spaced at 'spacing' meters.
    Return nodePositions list [(x,y,ix,iy), ...] and nodeMap {(ix,iy)->nodeIndex}.
    Removes nodes too close to inflated borders (ensures robot clearance).
    """
    nx = int(math.floor(fieldSize / spacing))
    ny = nx
    nodePositions = []
    nodeMap = {}
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            x = (ix + 0.5) * spacing
            y = (iy + 0.5) * spacing
            # enforce within safe margin from field borders
            if x < (robotHalf + clearance) or x > (fieldSize - (robotHalf + clearance)): continue
            if y < (robotHalf + clearance) or y > (fieldSize - (robotHalf + clearance)): continue
            if point_is_free(occupancy, (x,y)):
                nodePositions.append((x,y,ix,iy))
                nodeMap[(ix,iy)] = idx
                idx += 1
    return nodePositions, nodeMap

def nearest_node_index(nodePositions: List[Tuple[float,float,int,int]], p: Tuple[float,float]) -> Optional[int]:
    """Return nearest node index to point p by Euclidean distance. Returns None if none exist."""
    best = None; bestd = 1e9
    for i, (x,y,_,_) in enumerate(nodePositions):
        d = math.hypot(x - p[0], y - p[1])
        if d < bestd:
            bestd = d; best = i
    return best

def theta_star_search(occupancy: np.ndarray, startPose: Tuple[float,float], goalPose: Tuple[float,float]) -> Optional[List[Tuple[float,float,float]]]:
    """
    Theta*-style search over node graph with heading in state.
    Returns a path list of (x,y,heading) including exact start and goal (heading approx).
    """
    nodePositions, nodeMap = build_node_list(occupancy, nodeSpacing)
    if len(nodePositions) == 0:
        return None
    startIdx = nearest_node_index(nodePositions, startPose)
    goalIdx = nearest_node_index(nodePositions, goalPose)
    if startIdx is None or goalIdx is None:
        return None

    # neighbor offsets sampled in many angles to allow any-angle motion
    angleSamples = 16
    neighborOffsets = [(math.cos(2*math.pi*i/angleSamples)*nodeSpacing, math.sin(2*math.pi*i/angleSamples)*nodeSpacing) for i in range(angleSamples)]

    # heuristic: optimistic time using straight-line at forwardSpeed
    def heuristic_time(pos):
        return math.hypot(pos[0] - nodePositions[goalIdx][0], pos[1] - nodePositions[goalIdx][1]) / forwardSpeed

    # state = (nodeIndex, headingIdx)
    headingCount = 32
    def heading_idx_to_angle(idx): return (2*math.pi*idx)/headingCount
    def angle_to_heading_idx(angle): return int(round((angle % (2*math.pi)) / (2*math.pi) * headingCount)) % headingCount

    startHeading = 0.0
    startState = (startIdx, angle_to_heading_idx(startHeading))
    gscore: Dict[Tuple[int,int], float] = {startState: 0.0}
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {startState: None}
    parentExactPos: Dict[Tuple[int,int], Tuple[float,float]] = {startState: (nodePositions[startIdx][0], nodePositions[startIdx][1])}
    openHeap = []
    heapq.heappush(openHeap, (heuristic_time(nodePositions[startIdx]), startState))
    closed = set()
    expansions = 0
    startTime = time.time()

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

        # goal check: if node within nodeSpacing of goal node center
        if nodeIdx == goalIdx or math.hypot(nodeX - nodePositions[goalIdx][0], nodeY - nodePositions[goalIdx][1]) <= nodeSpacing * 1.1:
            # reconstruct path
            pathStates = []
            cur = state
            while cur is not None:
                nidx, hidx = cur
                pos = (nodePositions[nidx][0], nodePositions[nidx][1], heading_idx_to_angle(hidx))
                pathStates.append(pos)
                cur = parent[cur]
            pathStates.reverse()
            # prepend exact startPose and append exact goalPose
            if math.hypot(pathStates[0][0] - startPose[0], pathStates[0][1] - startPose[1]) > 1e-6:
                pathStates.insert(0, (startPose[0], startPose[1], startHeading))
            if math.hypot(pathStates[-1][0] - goalPose[0], pathStates[-1][1] - goalPose[1]) > 1e-6:
                pathStates.append((goalPose[0], goalPose[1], pathStates[-1][2]))
            endTime = time.time()
            # print timing summary
            print(f"Search time {endTime - startTime:.2f}s, expansions {expansions}, nodes {len(nodePositions)}")
            return pathStates

        # current parent's exact pos and heading
        currentExact = parentExactPos[state]
        currentHeading = heading_idx_to_angle(headingIdx)
        # generate successors via neighborOffsets
        for dx, dy in neighborOffsets:
            succX = nodeX + dx; succY = nodeY + dy
            if not (0.0 <= succX <= fieldSize and 0.0 <= succY <= fieldSize): continue
            # find nearest node index for succX,succY (quantize)
            ix = int(succX / nodeSpacing); iy = int(succY / nodeSpacing)
            if (ix,iy) not in nodeMap: continue
            succIdx = nodeMap[(ix,iy)]
            succPos = (nodePositions[succIdx][0], nodePositions[succIdx][1])

            # Theta* relaxation: try connecting from parentExact to succ (any-angle)
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

# ------------------------ Path smoothing (shortcutting) ------------------------
def smooth_path(path: List[Tuple[float,float,float]], occupancy: np.ndarray) -> List[Tuple[float,float,float]]:
    """
    Simple shortcut smoothing: try to replace sequences by straight LOS segments.
    Each replacement verified by line_of_sight_free at sampling step.
    """
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
                    # remove intermediate points
                    del coords[i+1:j]
                    replaced = True
                    break
                j -= 1
            if not replaced:
                i += 1
    # reattach headings (approximate using next point)
    newPath = []
    for k in range(len(coords)):
        x,y = coords[k]
        if k < len(coords) - 1:
            nx, ny = coords[k+1]
            heading = math.atan2(ny - y, nx - x)
        else:
            heading = newPath[-1][2] if newPath else 0.0
        newPath.append((x,y,heading))
    return newPath

# ------------------------ Trajectory -> wheel log ------------------------
def generate_wheel_log(path: List[Tuple[float,float,float]], occupancy: np.ndarray) -> pd.DataFrame:
    """
    Convert path into a time-parameterized log and then sample wheel RPS per timeStep.
    Uses earlier time_cost_between policy to choose strafe vs rotate+forward.
    Returns pandas DataFrame with columns: time_start, time_end, FR, FL, BR, BL (RPS).
    """
    logs = []   # (t0, t1, vx, vy, wz, heading_at_segment)
    t = 0.0
    currentHeading = path[0][2]
    for i in range(len(path)-1):
        a = (path[i][0], path[i][1]); b = (path[i+1][0], path[i+1][1])
        dist = math.hypot(b[0]-a[0], b[1]-a[1])
        if dist < 1e-8: continue
        moveTime, resultHeading = time_cost_between(a, b, currentHeading)
        # Determine whether the chosen strategy rotated (heading changed) or strafed (heading same)
        # Re-run decision to pick exact velocities
        dx = b[0] - a[0]; dy = b[1] - a[1]
        moveAngle = math.atan2(dy, dx)
        angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
        cosFactor = abs(math.cos(angDiff))
        speedOption1 = cosFactor * forwardSpeed + (1 - cosFactor) * strafeSpeed
        timeOption1 = dist / max(1e-6, speedOption1)
        rotateTime = angDiff / max(1e-6, rotationMax)
        timeOption2 = rotateTime + dist / forwardSpeed

        if timeOption1 <= timeOption2:
            # strafe/move maintaining heading
            vx_world = dx / timeOption1
            vy_world = dy / timeOption1
            logs.append((t, t + timeOption1, vx_world, vy_world, 0.0, currentHeading))
            t += timeOption1
            # heading unchanged
        else:
            # rotate in place
            rot_dir = 1.0 if ((moveAngle - currentHeading + 2*math.pi) % (2*math.pi)) < math.pi else -1.0
            logs.append((t, t + rotateTime, 0.0, 0.0, rot_dir * rotationMax, currentHeading))
            t += rotateTime
            # forward motion with new heading = moveAngle
            vx_world = dx / (dist / forwardSpeed)
            vy_world = dy / (dist / forwardSpeed)
            logs.append((t, t + dist / forwardSpeed, vx_world, vy_world, 0.0, moveAngle))
            t += dist / forwardSpeed
            currentHeading = moveAngle

    # Sample logs at timeStep intervals and map to wheel RPS
    numSteps = int(math.ceil(t / timeStep)) + 1
    table = []
    for s in range(numSteps):
        t0 = s * timeStep
        t1 = min(t0 + timeStep, t)
        tm = (t0 + t1) / 2.0
        vx = vy = wz = 0.0
        headingAt = 0.0
        for seg in logs:
            if seg[0] <= tm <= seg[1] + 1e-9:
                vx, vy, wz, headingAt = seg[2], seg[3], seg[4], seg[5]
                break
        # convert world velocities to robot body frame (rotate by -headingAt)
        ch = math.cos(-headingAt); sh = math.sin(-headingAt)
        bodyVx = vx * ch - vy * sh
        bodyVy = vx * sh + vy * ch
        # inverse kinematics for mecanum (wheel order FR, FL, BR, BL in output)
        lx = ly = robotSize / 2.0
        lsum = lx + ly
        r = wheelRadius
        # angular velocity rad/s
        wfl = (1.0/r) * (bodyVx - bodyVy - lsum * wz)
        wfr = (1.0/r) * (bodyVx + bodyVy + lsum * wz)
        wrl = (1.0/r) * (bodyVx + bodyVy - lsum * wz)
        wrr = (1.0/r) * (bodyVx - bodyVy + lsum * wz)
        # rad/s -> RPS
        rpsFL = wfl / (2 * math.pi)
        rpsFR = wfr / (2 * math.pi)
        rpsRL = wrl / (2 * math.pi)
        rpsRR = wrr / (2 * math.pi)
        # cap proportionally to motorMaxRps
        maxAbs = max(abs(rpsFL), abs(rpsFR), abs(rpsRL), abs(rpsRR), 1e-9)
        if maxAbs > motorMaxRps:
            scale = motorMaxRps / maxAbs
            rpsFL *= scale; rpsFR *= scale; rpsRL *= scale; rpsRR *= scale
        table.append({
            "time_start": t0,
            "time_end": t1,
            "FR": round(rpsFR, 6),
            "FL": round(rpsFL, 6),
            "BR": round(rpsRR, 6),
            "BL": round(rpsRL, 6)
        })
    df = pd.DataFrame(table)
    df.to_csv(wheelLogPath, index=False)
    print(f"Saved wheel log: {wheelLogPath}, totalTime {t:.2f}s, intervals {len(table)}")
    return df

# ------------------------ Visualization ------------------------
def visualize_and_save(occupancy: np.ndarray, path: List[Tuple[float,float,float]], snapshotIntervalSec: float = snapshotInterval):
    """
    Save a visualization PNG showing:
      - Field border (black)
      - Obstacles (filled light-blue with dark-blue outlines)
      - Inflated footprints (optional outline)
      - Path centerline (red)
      - Robot outline snapshots every 'snapshotIntervalSec' seconds with heading arrow
    """
    dpi = 200
    figSize = 8
    fig, ax = plt.subplots(figsize=(figSize, figSize), dpi=dpi)
    ax.set_xlim(0, fieldSize)
    ax.set_ylim(0, fieldSize)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # for lower-left origin visual match

    # Field border
    border = patches.Rectangle((0,0), fieldSize, fieldSize, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(border)

    # Obstacles from occupancy: extract rectangles by contouring coarse blobs for clear outlines
    occImg = occupancy.astype(np.uint8)
    # find connected components to get obstacle bounding boxes (simple approach)
    from scipy.ndimage import label, find_objects
    labeled, ncomp = label(occImg)
    objs = find_objects(labeled)
    for sl in objs:
        if sl is None: continue
        row0, row1 = sl[0].start, sl[0].stop - 1
        col0, col1 = sl[1].start, sl[1].stop - 1
        x0, y0 = grid_to_meters(col0, row1)  # lower-left of bounding box
        x1, y1 = grid_to_meters(col1, row0)  # upper-right
        width = x1 - x0
        height = y1 - y0
        # draw filled rectangle with outline
        rect = patches.Rectangle((x0, y0), width, height, facecolor='#cfeefe', edgecolor='#1177bb', linewidth=1.5, zorder=2)
        ax.add_patch(rect)

    # Path centerline
    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, color='red', linewidth=2, zorder=3)

    # Draw robot outline snapshots along path at given time intervals
    # First reconstruct a simple time parametric segmentation using time_cost_between decisions
    # We reuse the same policy as generate_wheel_log to compute timings.
    # Build segment list with (t0, t1, startPos, endPos, headingAtSegment)
    segments = []
    t = 0.0
    currentHeading = path[0][2]
    for i in range(len(path)-1):
        a = (path[i][0], path[i][1]); b = (path[i+1][0], path[i+1][1])
        dist = math.hypot(b[0]-a[0], b[1]-a[1])
        if dist < 1e-8: continue
        moveTime, resultHeading = time_cost_between(a, b, currentHeading)
        # decide which strategy used
        dx = b[0] - a[0]; dy = b[1] - a[1]
        moveAngle = math.atan2(dy, dx)
        angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
        cosFactor = abs(math.cos(angDiff))
        speedOption1 = cosFactor * forwardSpeed + (1 - cosFactor) * strafeSpeed
        timeOption1 = dist / max(1e-6, speedOption1)
        rotateTime = angDiff / max(1e-6, rotationMax)
        timeOption2 = rotateTime + dist / forwardSpeed

        if timeOption1 <= timeOption2:
            segments.append((t, t+timeOption1, a, b, currentHeading))
            t += timeOption1
        else:
            segments.append((t, t+rotateTime, a, a, currentHeading))  # rotation in place
            t += rotateTime
            segments.append((t, t + dist/forwardSpeed, a, b, moveAngle))
            t += dist/forwardSpeed
            currentHeading = moveAngle

    # sample snapshots at snapshotIntervalSec
    totalTime = t
    snapTimes = np.arange(0.0, totalTime + 1e-9, snapshotIntervalSec)
    for st in snapTimes:
        # find segment containing st
        seg = None
        for s in segments:
            if s[0] <= st <= s[1] + 1e-9:
                seg = s; break
        if seg is None:
            continue
        t0,t1,a,b,heading = seg
        if t1 - t0 < 1e-9:
            pos = a
        else:
            alpha = (st - t0) / (t1 - t0)
            pos = (a[0] * (1-alpha) + b[0]*alpha, a[1]*(1-alpha) + b[1]*alpha)
        # draw robot rectangle oriented by heading: rectangle centered at pos
        cx, cy = pos
        half = robotSize/2.0
        # rectangle patch using rotation transform
        rect = patches.Rectangle((cx - half, cy - half), robotSize, robotSize, linewidth=1.6,
                                 edgecolor='black', facecolor='none', zorder=5,
                                 transform=plt.matplotlib.transforms.Affine2D().rotate_around(cx, cy, heading) + ax.transData)
        ax.add_patch(rect)
        # heading arrow
        ax.arrow(cx, cy, 0.25 * math.cos(heading), 0.25 * math.sin(heading), head_width=0.08, head_length=0.08, fc='black', ec='black', zorder=6)

    # Save image
    plt.title("Planned Path with Robot Outlines and Heading")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.tight_layout()
    fig.savefig(outputImagePath, dpi=dpi)
    plt.close(fig)
    print(f"Saved visualization: {outputImagePath}")
    return outputImagePath

# ------------------------ Main ------------------------
def main():
    global startPose
    global goalPose
    occupancy = build_occupancy_from_image_and_rects()
    # safety: check start/goal are in free space and far enough from obstacles/walls
    if not point_is_free(occupancy, startPose):
        print("Start pose not free or too close to obstacles after inflation. Attempting to find nearest safe point.")
        # try project to nearest free cell center
        startIdx = meters_to_grid(startPose)
        found = False
        for r in range(0, 50):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    c = startIdx[0] + dx; rr = startIdx[1] + dy
                    if 0 <= c < gridSize and 0 <= rr < gridSize and not occupancy[rr, c]:
                        startPoseNew = grid_to_meters(c, rr)
                        if point_is_free(occupancy, startPoseNew):
                            print("Projected start to nearest safe point:", startPoseNew)
                            startPose = startPoseNew
                            found = True
                            break
                if found: break
            if found: break
        if not found:
            print("Could not find safe start. Aborting.")
            return

    if not point_is_free(occupancy, goalPose):
        print("Goal pose not free or too close to obstacles after inflation. Attempting to find nearest safe point.")
        goalIdx = meters_to_grid(goalPose)
        found = False
        for r in range(0, 50):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    c = goalIdx[0] + dx; rr = goalIdx[1] + dy
                    if 0 <= c < gridSize and 0 <= rr < gridSize and not occupancy[rr, c]:
                        goalPoseNew = grid_to_meters(c, rr)
                        if point_is_free(occupancy, goalPoseNew):
                            print("Projected goal to nearest safe point:", goalPoseNew)
                            goalPose = goalPoseNew
                            found = True
                            break
                if found: break
            if found: break
        if not found:
            print("Could not find safe goal. Aborting.")
            return

    # run planner
    path = theta_star_search(occupancy, startPose, goalPose)
    if path is None:
        print("Planner failed to find path.")
        return

    # smoothing
    pathSmoothed = smooth_path(path, occupancy)

    # ensure final path keeps 1cm clearance (redundant but safe): verify all points free
    for (x,y,_) in pathSmoothed:
        if not point_is_free(occupancy, (x,y)):
            print("Post-smoothing collision detected. Aborting.")
            return

    # generate wheel log CSV
    df = generate_wheel_log(pathSmoothed, occupancy)

    # visualization
    imgPath = visualize_and_save(occupancy, pathSmoothed)

    # print brief summary
    print("Path length (points):", len(pathSmoothed))
    print("Wheel log saved to:", wheelLogPath)
    print("Visualization saved to:", imgPath)
    # print first 10 rows of wheel log
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
