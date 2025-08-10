"""
High-precision any-angle Theta*-style planner for a 3.6666667 m square field
targeting a 0.40m x 0.40m mecanum robot.

Key properties (from your dot-points):
 - Field: 3.6666667 m square.
 - Uncertainty <= 5 mm (resolution sampling uses 0.005 m by default).
 - All variables/lists/arrays in camelCase; all functions in snake_case.
 - Editable rectangularObstacles list (axis-aligned rectangles in meters).
 - Prioritises time (not distance): models rotate-vs-strafe tradeoff.
 - Reads attached image (if present) to derive obstacles; otherwise uses rectangularObstacles or random demo obstacles.
 - Inflates obstacles by robot half-size + 1 cm clearance; robot must remain >= 1 cm from obstacles/walls.
 - Produces saved PNG visualization with: field border, obstacle outlines, red centerline path, robot outline snapshots and heading arrows.
 - Produces wheel log CSV with time intervals and wheel RPS for FR, FL, BR, BL.
 - No global reassignments inside functions; startPose and goalPose passed as parameters.
 - Reasonably optimized; tune nodeSpacing/resolution if runtime allows.
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

# ---------------------------- USER PARAMETERS (camelCase) ----------------------------
fieldSize = 3.6666667                  # meters (square side)
uncertainty = 0.005                    # meters (5 mm)
resolution = 0.005                     # meters per occupancy cell (<= uncertainty)
nodeSpacing = 0.02                     # meters between search nodes (coarse search grid)
robotSize = 0.40                       # meters (square robot side)
clearance = 0.01                       # meters (1 cm minimum clearance)
wheelDiameter = 0.08                   # meters
wheelRadius = wheelDiameter / 2.0
motorMaxRps = 2.5
rotationMax = 1.0                      # rad/s max in-place
forwardSpeedNominal = min(0.6, wheelRadius * motorMaxRps * 2 * math.pi)
strafeFactor = 0.8
diagFactor = 0.9

timeStep = 0.1                         # seconds per log interval
snapshotInterval = 0.3                 # seconds between robot-outline snapshots (visualisation)
smoothingIterations = 3
maxExpansions = 200000

# Obstacles editable by user (x0,y0,x1,y1) in meters, lower-left origin.
rectangularObstacles: List[Tuple[float, float, float, float]] = [
    # Example: (0.9, 1.0, 1.6, 1.5)
]

attachedImagePath = "/mnt/data/e623a063-3410-45af-b5d3-5dd807aa1647.png"
useRandomObstaclesIfNone = True
randomObstacleCount = 6
randomSeed = 42

# Output files
outputImagePath = "saved_path.png"
wheelLogPath = "wheel_log.csv"

# Derived constants
robotHalf = robotSize / 2.0
inflationRadius = robotHalf + clearance
gridSize = int(math.ceil(fieldSize / resolution))
forwardSpeed = forwardSpeedNominal
strafeSpeed = strafeFactor * forwardSpeed
diagSpeed = diagFactor * forwardSpeed

# ---------------------------- Utility (snake_case) ----------------------------
def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

def meters_to_grid(xy: Tuple[float, float]) -> Tuple[int, int]:
    """Convert (x,y) in meters (origin lower-left) to (col,row) indices (origin top-left)."""
    x, y = xy
    col = int(x / resolution)
    row = gridSize - 1 - int(y / resolution)
    col = int(clamp(col, 0, gridSize - 1))
    row = int(clamp(row, 0, gridSize - 1))
    return col, row

def grid_to_meters(col: int, row: int) -> Tuple[float, float]:
    """Convert grid indices (col,row) to center (x,y) meters (origin lower-left)."""
    x = (col + 0.5) * resolution
    y = ((gridSize - 1 - row) + 0.5) * resolution
    return x, y

def sample_line(a: Tuple[float,float], b: Tuple[float,float], step: float) -> List[Tuple[float,float]]:
    """Return samples on segment a->b with spacing <= step (inclusive endpoints)."""
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return [a]
    steps = max(1, int(math.ceil(dist / step)))
    return [(a[0] + dx * t / steps, a[1] + dy * t / steps) for t in range(steps+1)]

def rect_inflated(rect: Tuple[float,float,float,float], infl: float) -> Tuple[float,float,float,float]:
    x0,y0,x1,y1 = rect
    return (x0-infl, y0-infl, x1+infl, y1+infl)

def rect_to_grid(rect: Tuple[float,float,float,float]) -> Tuple[int,int,int,int]:
    """Return grid cell bounds (c0,r0,c1,r1) covering rectangle (meters)."""
    x0,y0,x1,y1 = rect
    c0,r1 = meters_to_grid((x0, y0))
    c1,r0 = meters_to_grid((x1, y1))
    cmin,cmax = min(c0,c1), max(c0,c1)
    rmin,rmax = min(r0,r1), max(r0,r1)
    return cmin, rmin, cmax, rmax

# ---------------------------- Occupancy build ----------------------------
def build_occupancy(startPose: Tuple[float,float], goalPose: Tuple[float,float]) -> np.ndarray:
    """
    Build occupancy boolean grid (shape gridSize x gridSize). True => occupied.
    Uses attached image if available, overlays rectangularObstacles, optionally generates random obstacles.
    Then inflates obstacles by inflationRadius.
    """
    occ = np.zeros((gridSize, gridSize), dtype=bool)

    # 1) image-based obstacles if present
    if os.path.exists(attachedImagePath):
        try:
            img = Image.open(attachedImagePath).convert("L")
            img = img.resize((gridSize, gridSize), resample=Image.NEAREST)
            arr = np.array(img)
            occ_img = arr < 128
            occ = np.logical_or(occ, occ_img)
        except Exception as e:
            print("Warning: image load failed; using rectangles only:", e)

    # 2) overlay rectangularObstacles (meters)
    localRects = list(rectangularObstacles)  # copy to avoid mutating global list
    if len(localRects) == 0 and useRandomObstaclesIfNone:
        random.seed(randomSeed)
        for _ in range(randomObstacleCount):
            w = random.uniform(0.18, 0.6)
            h = random.uniform(0.18, 0.6)
            x0 = random.uniform(0.05, fieldSize - 0.05 - w)
            y0 = random.uniform(0.05, fieldSize - 0.05 - h)
            localRects.append((x0, y0, x0 + w, y0 + h))

    for rect in localRects:
        x0,y0,x1,y1 = rect
        x0 = clamp(x0, 0.0, fieldSize); x1 = clamp(x1, 0.0, fieldSize)
        y0 = clamp(y0, 0.0, fieldSize); y1 = clamp(y1, 0.0, fieldSize)
        c0,r0,c1,r1 = rect_to_grid((x0,y0,x1,y1))
        occ[r0:r1+1, c0:c1+1] = True

    # 3) inflate by inflationRadius (box kernel)
    infl_cells = int(math.ceil(inflationRadius / resolution))
    if infl_cells > 0:
        kernel = np.ones((2*infl_cells + 1, 2*infl_cells + 1), dtype=bool)
        occ = binary_dilation(occ, structure=kernel)

    # 4) ensure start and goal are not inside obstacles (we'll project later if they are)
    return occ

# ---------------------------- Collision & LOS checks ----------------------------
def point_is_free(occupancy: np.ndarray, p: Tuple[float,float]) -> bool:
    """True if point p (meters) is inside field and not in occupied cell."""
    x,y = p
    if not (0.0 <= x <= fieldSize and 0.0 <= y <= fieldSize):
        return False
    c,r = meters_to_grid((x,y))
    return not occupancy[r,c]

def line_of_sight_free(occupancy: np.ndarray, a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    """Sample segment at step <= min(resolution, uncertainty/2) and ensure all samples free."""
    step = min(resolution, uncertainty / 2.0)
    for s in sample_line(a,b,step):
        if not point_is_free(occupancy, s):
            return False
    return True

# ---------------------------- Time-cost model ----------------------------
def time_cost_between(a: Tuple[float,float], b: Tuple[float,float], currentHeading: float) -> Tuple[float, float]:
    """
    Estimate time (seconds) to traverse a->b from currentHeading.
    Returns (timeSeconds, resultingHeading).
    Evaluates: maintain-heading (strafe/drive) vs rotate-then-forward.
    """
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0.0:
        return 0.0, currentHeading
    moveAngle = math.atan2(dy, dx)
    # minimal angular difference in [0, pi]
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

# ---------------------------- Node graph + Theta*-style search ----------------------------
def build_node_list(occupancy: np.ndarray, spacing: float) -> Tuple[List[Tuple[float,float,int,int]], Dict[Tuple[int,int],int]]:
    """
    Create candidate nodes spaced at 'spacing' meters (centers).
    Returns (nodePositions list of (x,y,ix,iy), nodeMap {(ix,iy)->index}).
    Excludes nodes too close to borders or inside obstacles.
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
            if point_is_free(occupancy, (x,y)):
                nodePositions.append((x,y,ix,iy))
                nodeMap[(ix,iy)] = idx
                idx += 1
    return nodePositions, nodeMap

def nearest_node_index(nodePositions: List[Tuple[float,float,int,int]], p: Tuple[float,float]) -> Optional[int]:
    """Return index of nearest node center to p (meters)."""
    best = None; bestd = 1e9
    for i,(x,y,_,_) in enumerate(nodePositions):
        d = math.hypot(x - p[0], y - p[1])
        if d < bestd:
            bestd = d; best = i
    return best

def theta_star_search(occupancy: np.ndarray, startPose: Tuple[float,float], goalPose: Tuple[float,float]) -> Optional[List[Tuple[float,float,float]]]:
    """
    Theta*-style search returning path as list of (x,y,heading).
    Heading values are approximate and used for time-cost decisions.
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

    def heuristic_time(pos):
        gx, gy, _, _ = nodePositions[goalIdx]
        return math.hypot(pos[0] - gx, pos[1] - gy) / forwardSpeed

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
            print("Search expansion cap reached.")
            break

        nodeIdx, headingIdx = state
        nodeX, nodeY, _, _ = nodePositions[nodeIdx]

        # goal condition (close to goal node)
        gx, gy, _, _ = nodePositions[goalIdx]
        if nodeIdx == goalIdx or math.hypot(nodeX - gx, nodeY - gy) <= nodeSpacing * 1.1:
            # reconstruct path
            sequence = []
            cur = state
            while cur is not None:
                nidx, hidx = cur
                px, py, _, _ = nodePositions[nidx]
                sequence.append((px, py, heading_idx_to_angle(hidx)))
                cur = parent[cur]
            sequence.reverse()
            # attach exact start and goal if needed
            if math.hypot(sequence[0][0] - startPose[0], sequence[0][1] - startPose[1]) > 1e-6:
                sequence.insert(0, (startPose[0], startPose[1], startHeading))
            if math.hypot(sequence[-1][0] - goalPose[0], sequence[-1][1] - goalPose[1]) > 1e-6:
                sequence.append((goalPose[0], goalPose[1], sequence[-1][2]))
            tend = time.time()
            print(f"Search time: {tend - tstart:.2f}s, expansions: {expansions}, nodes: {len(nodePositions)}")
            return sequence

        currentExact = parentExactPos[state]
        currentHeading = heading_idx_to_angle(headingIdx)

        for dx, dy in neighborOffsets:
            succX = nodeX + dx; succY = nodeY + dy
            if not (0.0 <= succX <= fieldSize and 0.0 <= succY <= fieldSize): continue
            ix = int(succX / nodeSpacing); iy = int(succY / nodeSpacing)
            if (ix,iy) not in nodeMap: continue
            succIdx = nodeMap[(ix,iy)]
            succPos = (nodePositions[succIdx][0], nodePositions[succIdx][1])

            # Theta* relaxation: connect from parentExact if LOS
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

# ---------------------------- Path smoothing ----------------------------
def smooth_path(path: List[Tuple[float,float,float]], occupancy: np.ndarray) -> List[Tuple[float,float,float]]:
    """Shortcut smoothing with LOS checks; preserves clearance using occupancy checks."""
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
        if k < len(coords) - 1:
            nx, ny = coords[k+1]
            heading = math.atan2(ny - y, nx - x)
        else:
            heading = out[-1][2] if out else 0.0
        out.append((x,y,heading))
    return out

# ---------------------------- Trajectory => Wheel log ----------------------------
def generate_wheel_log(path: List[Tuple[float,float,float]], occupancy: np.ndarray, startPose: Tuple[float,float]) -> pd.DataFrame:
    """
    Convert path into time-parameterized motion segments and sample wheel RPS per timeStep.
    Output columns: time_start, time_end, FR, FL, BR, BL (RPS).
    """
    logs = []   # segments: (t0,t1, vx_world, vy_world, wz, heading)
    t = 0.0
    currentHeading = path[0][2]
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
            vx_world = dx / timeOption1
            vy_world = dy / timeOption1
            logs.append((t, t + timeOption1, vx_world, vy_world, 0.0, currentHeading))
            t += timeOption1
        else:
            rot_dir = 1.0 if ((moveAngle - currentHeading + 2*math.pi) % (2*math.pi)) < math.pi else -1.0
            logs.append((t, t + rotateTime, 0.0, 0.0, rot_dir * rotationMax, currentHeading))
            t += rotateTime
            logs.append((t, t + dist / forwardSpeed, dx / (dist / forwardSpeed), dy / (dist / forwardSpeed), 0.0, moveAngle))
            t += dist / forwardSpeed
            currentHeading = moveAngle

    # sample logs at intervals
    numSteps = int(math.ceil(t / timeStep)) + 1
    rows = []
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
        # world -> body frame
        ch = math.cos(-headingAt); sh = math.sin(-headingAt)
        bodyVx = vx * ch - vy * sh
        bodyVy = vx * sh + vy * ch
        lx = ly = robotSize / 2.0
        lsum = lx + ly
        r = wheelRadius
        wfl = (1.0/r) * (bodyVx - bodyVy - lsum * wz)
        wfr = (1.0/r) * (bodyVx + bodyVy + lsum * wz)
        wrl = (1.0/r) * (bodyVx + bodyVy - lsum * wz)
        wrr = (1.0/r) * (bodyVx - bodyVy + lsum * wz)
        # rad/s -> RPS
        rpsFL = wfl / (2 * math.pi); rpsFR = wfr / (2 * math.pi)
        rpsRL = wrl / (2 * math.pi); rpsRR = wrr / (2 * math.pi)
        maxAbs = max(abs(rpsFL), abs(rpsFR), abs(rpsRL), abs(rpsRR), 1e-9)
        if maxAbs > motorMaxRps:
            scale = motorMaxRps / maxAbs
            rpsFL *= scale; rpsFR *= scale; rpsRL *= scale; rpsRR *= scale
        rows.append({
            "time_start": round(t0, 6),
            "time_end": round(t1, 6),
            "FR": round(rpsFR, 6),
            "FL": round(rpsFL, 6),
            "BR": round(rpsRR, 6),
            "BL": round(rpsRL, 6)
        })
    df = pd.DataFrame(rows)
    df.to_csv(wheelLogPath, index=False)
    print(f"Wheel log saved: {wheelLogPath} (total time {t:.2f}s, intervals {len(rows)})")
    return df

# ---------------------------- Visualization ----------------------------
def visualize_and_save(occupancy: np.ndarray, path: List[Tuple[float,float,float]], outputPath: str, snapInterval: float = snapshotInterval) -> str:
    """
    Save PNG with:
      - Field border
      - Obstacle fills and outlines
      - Path centerline (red)
      - Robot outline snapshots with heading arrows every snapInterval seconds
    """
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    ax.set_xlim(0, fieldSize); ax.set_ylim(0, fieldSize)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Field border
    ax.add_patch(patches.Rectangle((0,0), fieldSize, fieldSize, fill=False, edgecolor='black', linewidth=2, zorder=1))

    # Draw obstacle bounding boxes from occupancy components for clear outlines
    occImg = occupancy.astype(np.uint8)
    labeled, ncomp = label(occImg)
    objs = find_objects(labeled)
    for sl in objs:
        if sl is None: continue
        row0, row1 = sl[0].start, sl[0].stop - 1
        col0, col1 = sl[1].start, sl[1].stop - 1
        x0, y0 = grid_to_meters(col0, row1)
        x1, y1 = grid_to_meters(col1, row0)
        width = x1 - x0; height = y1 - y0
        rect = patches.Rectangle((x0, y0), width, height, facecolor='#cfeefe', edgecolor='#1177bb', linewidth=1.5, zorder=2)
        ax.add_patch(rect)

    # Path centerline
    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, color='red', linewidth=2, zorder=3)

    # Recompute simple time segments to place snapshots (reuse same policy as wheel log)
    # Build segments
    segments = []
    t = 0.0
    currentHeading = path[0][2]
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
            segments.append((t, t+timeOption1, a, b, currentHeading))
            t += timeOption1
        else:
            segments.append((t, t+rotateTime, a, a, currentHeading))
            t += rotateTime
            segments.append((t, t + dist / forwardSpeed, a, b, moveAngle))
            t += dist / forwardSpeed
            currentHeading = moveAngle

    totalTime = t
    if totalTime <= 0:
        # static drawing if no time-based motion
        sampleTimes = [0.0]
    else:
        sampleTimes = list(np.arange(0.0, totalTime + 1e-9, snapInterval))

    for st in sampleTimes:
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
            alpha = (st - t0) / max(1e-9, (t1 - t0))
            pos = (a[0] * (1-alpha) + b[0] * alpha, a[1] * (1-alpha) + b[1] * alpha)
        cx, cy = pos
        half = robotSize / 2.0
        rect = patches.Rectangle((cx-half, cy-half), robotSize, robotSize, linewidth=1.6,
                                 edgecolor='black', facecolor='none', zorder=5,
                                 transform=plt.matplotlib.transforms.Affine2D().rotate_around(cx, cy, heading) + ax.transData)
        ax.add_patch(rect)
        ax.arrow(cx, cy, 0.25 * math.cos(heading), 0.25 * math.sin(heading), head_width=0.08, head_length=0.08, fc='black', ec='black', zorder=6)

    ax.set_title("Planned Path with Robot Outlines and Heading")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout()
    fig.savefig(outputPath, dpi=200)
    plt.close(fig)
    return outputPath

# ---------------------------- Helpers: safe projection ----------------------------
def find_nearest_free(occupancy: np.ndarray, pose: Tuple[float,float], maxRadiusCells: int = 50) -> Optional[Tuple[float,float]]:
    """If pose is occupied, search outward for nearest free cell center and return meters coord."""
    col0, row0 = meters_to_grid(pose)
    for r in range(maxRadiusCells+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                c = col0 + dx; rr = row0 + dy
                if 0 <= c < gridSize and 0 <= rr < gridSize and not occupancy[rr, c]:
                    x,y = grid_to_meters(c, rr)
                    if point_is_free(occupancy, (x,y)):
                        return (x,y)
    return None

# ---------------------------- Main (no global reassign inside functions) ----------------------------
def plan_and_run(startPose: Tuple[float,float], goalPose: Tuple[float,float]):
    """
    Top-level function. Builds occupancy, projects start/goal if necessary, runs planner,
    smooths result, generates wheel log CSV, saves visualization PNG.
    """
    occupancy = build_occupancy(startPose, goalPose)

    # Ensure start/goal safe; attempt projection if not
    if not point_is_free(occupancy, startPose):
        proj = find_nearest_free(occupancy, startPose)
        if proj is None:
            print("Start pose unsafe and no nearby free cell found. Aborting.")
            return
        print("Projected startPose to nearest safe point:", proj)
        startPose = proj

    if not point_is_free(occupancy, goalPose):
        proj = find_nearest_free(occupancy, goalPose)
        if proj is None:
            print("Goal pose unsafe and no nearby free cell found. Aborting.")
            return
        print("Projected goalPose to nearest safe point:", proj)
        goalPose = proj

    path = theta_star_search(occupancy, startPose, goalPose)
    if path is None:
        print("Planner failed to find a path.")
        return

    pathSmoothed = smooth_path(path, occupancy)
    # final safety check: ensure all path points are free
    for (x,y,_) in pathSmoothed:
        if not point_is_free(occupancy, (x,y)):
            print("Collision detected in final path. Aborting.")
            return

    df = generate_wheel_log(pathSmoothed, occupancy, startPose)
    imagePath = visualize_and_save(occupancy, pathSmoothed, outputImagePath)
    print("Plan complete. Path points:", len(pathSmoothed))
    print("Saved wheel log:", wheelLogPath)
    print("Saved visualization:", imagePath)
    print(df.head(10).to_string(index=False))

# ---------------------------- Example usage ----------------------------
if __name__ == "__main__":
    # Example start/goal (meters). Replace with desired coordinates.
    start = (0.3, 0.3)
    goal = (3.3666667, 3.3666667)
    plan_and_run(start, goal)
