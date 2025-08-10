"""
Optimised Theta*-style any-angle planner for a 3.6666667 m square field,
designed to meet your updated dot-points while staying fast (<3 min).

Key design choices (keeps your requirements):
 - Uses a coarse node graph (adaptive nodeSpacing) for fast search, then performs
   local high-resolution shortcut smoothing only along the found path (not entire map).
 - Occupancy built at resolution (default 5 mm) for correctness; node graph spacing
   is chosen so total nodes remain reasonable. If node count would be huge, spacing
   is increased automatically to meet runtime limits.
 - Collision checking during search uses inflated occupancy (robot OBB approximated
   by circle for node validity) for speed. Final smoothing uses oriented-rectangle (OBB)
   collision checks sampled only along candidate shortcuts.
 - Time-prioritised cost model (strafe vs rotate+forward) used in edge costs.
 - Visualization saved to PNG with field border, obstacle outlines, path centerline,
   robot OBB snapshots and heading arrows. Demo places a 1m x 1m central obstacle and
   moves robot from one corner to opposite as requested.
 - Produces wheel RPS CSV (FR, FL, BR, BL) sampled at timeStep intervals.
 - All variables/lists/arrays use camelCase; all functions use snake_case.
 - No globals reassigned inside functions: main() passes parameters explicitly.

Run: python3 planner.py  (script runs plan_and_run() with demo start/goal by default).
"""

import os, math, time, heapq, random
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from scipy.ndimage import binary_dilation, label, find_objects

# ----------------------------- USER PARAMETERS (camelCase) -----------------------------
fieldSize = 3.6666667                  # meters
uncertainty = 0.005                    # meters (5 mm)
baseResolution = 0.005                 # occupancy cell size (m)
initialNodeSpacing = 0.03             # start coarse node spacing (m) to keep search small
minNodeSpacing = 0.015                # minimum node spacing allowed (higher runtime)
maxNodeCount = 120000                 # safety cap on node count (controls runtime)
robotSize = 0.40                       # m x m robot
clearance = 0.01                       # m (1 cm)
inflationRadius = robotSize / 2.0 + clearance
wheelDiameter = 0.08
wheelRadius = wheelDiameter / 2.0
motorMaxRps = 2.5
rotationMax = 1.0
forwardSpeedNominal = min(0.6, wheelRadius * motorMaxRps * 2 * math.pi)
strafeFactor = 0.8
diagFactor = 0.9

timeStep = 0.1                         # seconds per CSV entry
snapshotInterval = 0.3                 # seconds between robot-outline snapshots
smoothingIterations = 4
maxExpansions = 200000

# Obstacles (editable). Axis-aligned rectangles (x0,y0,x1,y1) in meters.
rectangularObstacles: List[Tuple[float,float,float,float]] = []
attachedImagePath = "/mnt/data/e623a063-3410-45af-b5d3-5dd807aa1647.png"

# Demo: center 1m obstacle and corner-to-corner task
centerObstacleSize = 1.0
centerObs = (fieldSize/2 - 0.5, fieldSize/2 - 0.5, fieldSize/2 + 0.5, fieldSize/2 + 0.5)
startPoseDefault = (0.05 + robotSize/2, 0.05 + robotSize/2)
goalPoseDefault = (fieldSize - (0.05 + robotSize/2), fieldSize - (0.05 + robotSize/2))

outputImagePath = "saved_path.png"
wheelLogPath = "wheel_log.csv"

# Derived
robotHalf = robotSize / 2.0
forwardSpeed = forwardSpeedNominal
strafeSpeed = strafeFactor * forwardSpeed

# ----------------------------- Utilities (snake_case) -----------------------------
def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

def meters_to_grid(x: float, y: float, resolution: float, gridSize: int) -> Tuple[int,int]:
    col = int(clamp(int(x / resolution), 0, gridSize-1))
    row = int(clamp(gridSize - 1 - int(y / resolution), 0, gridSize-1))
    return col, row

def grid_to_meters(col: int, row: int, resolution: float, gridSize: int) -> Tuple[float,float]:
    x = (col + 0.5) * resolution
    y = ((gridSize - 1 - row) + 0.5) * resolution
    return x, y

def sample_line(a: Tuple[float,float], b: Tuple[float,float], step: float) -> List[Tuple[float,float]]:
    dx = b[0] - a[0]; dy = b[1] - a[1]; dist = math.hypot(dx, dy)
    if dist == 0.0: return [a]
    steps = max(1, int(math.ceil(dist / step)))
    return [(a[0] + dx * t / steps, a[1] + dy * t / steps) for t in range(steps+1)]

# ----------------------------- Occupancy (build & inflate) -----------------------------
def build_occupancy_grid(resolution: float, includeCenterObs: bool=True,
                         rects: Optional[List[Tuple[float,float,float,float]]] = None,
                         attachedImage: str = attachedImagePath) -> Tuple[np.ndarray,int]:
    """
    Build occupancy boolean array (shape NxN) from attached image (if present)
    and rectangle list. Inflate by inflationRadius (box dilation).
    Returns occupancy, gridSize.
    """
    gridSize = int(math.ceil(fieldSize / resolution))
    occ = np.zeros((gridSize, gridSize), dtype=bool)

    # 1) image-based obstacles
    if os.path.exists(attachedImage):
        try:
            im = Image.open(attachedImage).convert("L")
            im = im.resize((gridSize, gridSize), resample=Image.NEAREST)
            arr = np.array(im)
            occ_img = arr < 128
            occ = np.logical_or(occ, occ_img)
        except Exception:
            pass

    # 2) rectangles
    useRects = list(rects) if rects else []
    if includeCenterObs:
        useRects.append(centerObs)
    for (x0,y0,x1,y1) in useRects:
        x0 = clamp(x0, 0.0, fieldSize); y0 = clamp(y0, 0.0, fieldSize)
        x1 = clamp(x1, 0.0, fieldSize); y1 = clamp(y1, 0.0, fieldSize)
        c0, r1 = meters_to_grid(x0, y0, resolution, gridSize)
        c1, r0 = meters_to_grid(x1, y1, resolution, gridSize)
        cmin, cmax = min(c0,c1), max(c0,c1)
        rmin, rmax = min(r0,r1), max(r0,r1)
        occ[rmin:rmax+1, cmin:cmax+1] = True

    # 3) inflate via binary dilation
    infl_cells = int(math.ceil(inflationRadius / resolution))
    if infl_cells > 0:
        kernel = np.ones((2*infl_cells + 1, 2*infl_cells + 1), dtype=bool)
        occ = binary_dilation(occ, structure=kernel)

    return occ, gridSize

# ----------------------------- Fast collision checks for nodes -----------------------------
def center_point_free(occ: np.ndarray, p: Tuple[float,float], resolution: float, gridSize: int) -> bool:
    """Fast check: is point center free? (used for node validity)."""
    if not (0.0 <= p[0] <= fieldSize and 0.0 <= p[1] <= fieldSize): return False
    c,r = meters_to_grid(p[0], p[1], resolution, gridSize)
    return not occ[r,c]

# ----------------------------- Conservative OBB collision check (for smoothing only) -----------------------------
def robot_corners(center: Tuple[float,float], heading: float, half: float) -> List[Tuple[float,float]]:
    cx,cy = center
    corners_local = [(-half,-half), (-half, half), (half, half), (half, -half)]
    cosH = math.cos(heading); sinH = math.sin(heading)
    return [(cx + (lx * cosH - ly * sinH), cy + (lx * sinH + ly * cosH)) for (lx,ly) in corners_local]

def obb_collides(occ: np.ndarray, center: Tuple[float,float], heading: float, half: float,
                 resolution: float, gridSize: int) -> bool:
    """
    Conservative OBB collision check: sample inside OBB bounding box at spacing <= min(resolution, uncertainty/2).
    Only used on short lists (path smoothing), so slower checks are acceptable.
    """
    corners = robot_corners(center, heading, half)
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    minx = clamp(min(xs) - resolution, 0.0, fieldSize); maxx = clamp(max(xs) + resolution, 0.0, fieldSize)
    miny = clamp(min(ys) - resolution, 0.0, fieldSize); maxy = clamp(max(ys) + resolution, 0.0, fieldSize)
    step = min(resolution, uncertainty / 2.0)
    nx = max(1, int(math.ceil((maxx - minx) / step)))
    ny = max(1, int(math.ceil((maxy - miny) / step)))
    # point-in-rotated-rect test (convex)
    poly = corners
    def point_in_poly(px, py):
        sign = None
        for i in range(len(poly)):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
            cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
            if abs(cross) < 1e-12: continue
            curSign = cross > 0
            if sign is None: sign = curSign
            elif sign != curSign: return False
        return True
    for i in range(nx+1):
        for j in range(ny+1):
            px = minx + (i / max(1,nx)) * (maxx - minx)
            py = miny + (j / max(1,ny)) * (maxy - miny)
            if not point_in_poly(px, py): continue
            if not center_point_free(occ, (px,py), resolution, gridSize):
                return True
    return False

# ----------------------------- Time-cost model -----------------------------
def time_cost(a: Tuple[float,float], b: Tuple[float,float], currentHeading: float) -> Tuple[float, float]:
    """Return (timeSeconds, resultingHeading), choosing strafe vs rotate+forward."""
    dx = b[0] - a[0]; dy = b[1] - a[1]; dist = math.hypot(dx, dy)
    if dist == 0.0: return 0.0, currentHeading
    moveAngle = math.atan2(dy, dx)
    angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
    cosF = abs(math.cos(angDiff))
    speed1 = cosF * forwardSpeed + (1 - cosF) * strafeSpeed
    t1 = dist / max(1e-9, speed1)
    rotT = angDiff / max(1e-9, rotationMax)
    t2 = rotT + dist / forwardSpeed
    if t1 <= t2: return t1, currentHeading
    else: return t2, moveAngle

# ----------------------------- Node graph + Theta* search (optimized) -----------------------------
def build_node_graph(occ: np.ndarray, resolution: float, nodeSpacing: float) -> Tuple[List[Tuple[float,float,int,int]], Dict[Tuple[int,int],int], int]:
    """Create node positions at spacing nodeSpacing. Use center-point free test (fast)."""
    nx = int(math.floor(fieldSize / nodeSpacing))
    ny = nx
    nodePositions = []
    nodeMap: Dict[Tuple[int,int], int] = {}
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            x = (ix + 0.5) * nodeSpacing
            y = (iy + 0.5) * nodeSpacing
            # respect field border clearance
            if x < (robotHalf + clearance) or x > (fieldSize - (robotHalf + clearance)): continue
            if y < (robotHalf + clearance) or y > (fieldSize - (robotHalf + clearance)): continue
            if center_point_free(occ, (x,y), resolution, occ.shape[0]):
                nodePositions.append((x,y,ix,iy))
                nodeMap[(ix,iy)] = idx; idx += 1
    return nodePositions, nodeMap, idx

def theta_star(occ: np.ndarray, resolution: float, startPose: Tuple[float,float], goalPose: Tuple[float,float],
               nodeSpacingStart: float) -> Optional[List[Tuple[float,float,float]]]:
    """
    Adaptive Theta*-style search: increase nodeSpacing if node count would be too large.
    Returns path (x,y,heading) or None.
    """
    # choose nodeSpacing adaptively to keep node count <= maxNodeCount
    nodeSpacing = nodeSpacingStart
    while True:
        nodePositions, nodeMap, nodeCount = build_node_graph(occ, resolution, nodeSpacing)
        if nodeCount == 0:
            # if grid too coarse, reduce spacing (but not below minNodeSpacing)
            if nodeSpacing <= minNodeSpacing:
                return None
            nodeSpacing = max(minNodeSpacing, nodeSpacing / 1.5)
            continue
        if nodeCount > maxNodeCount:
            nodeSpacing = min(nodeSpacing * 1.5, 0.5)
            if nodeSpacing > 0.5:
                return None
            continue
        break

    # neighbor offsets (8 directions + extra angles)
    angleSamples = 16
    neighborOffsets = [(math.cos(2*math.pi*i/angleSamples)*nodeSpacing, math.sin(2*math.pi*i/angleSamples)*nodeSpacing) for i in range(angleSamples)]

    def heuristic(npos):
        gx,gy,_,_ = nodePositions[goalIdx]
        return math.hypot(npos[0] - gx, npos[1] - gy) / forwardSpeed

    startIdx = None; goalIdx = None
    # find nearest node indices
    def nearest_idx(p):
        best=None; bd=1e9
        for i,(x,y,_,_) in enumerate(nodePositions):
            d = math.hypot(x - p[0], y - p[1])
            if d < bd: bd=d; best=i
        return best

    startIdx = nearest_idx(startPose)
    goalIdx = nearest_idx(goalPose)
    if startIdx is None or goalIdx is None: return None

    # heading quantization
    headingCount = 32
    def idx_to_angle(i): return (2*math.pi*i)/headingCount
    def angle_to_idx(a): return int(round((a % (2*math.pi)) / (2*math.pi) * headingCount)) % headingCount

    startHeading = 0.0
    startState = (startIdx, angle_to_idx(startHeading))
    gscore = {startState: 0.0}
    parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {startState: None}
    parentPos: Dict[Tuple[int,int], Tuple[float,float]] = {startState: (nodePositions[startIdx][0], nodePositions[startIdx][1])}
    openHeap = [(heuristic(nodePositions[startIdx]), startState)]
    closed = set()
    expansions = 0
    t0 = time.time()

    while openHeap:
        _, state = heapq.heappop(openHeap)
        if state in closed: continue
        closed.add(state)
        expansions += 1
        if expansions > maxExpansions:
            break
        nodeIdx, headingIdx = state
        nodeX, nodeY, _, _ = nodePositions[nodeIdx]
        # goal test: proximity to goal node
        gx,gy,_,_ = nodePositions[goalIdx]
        if nodeIdx == goalIdx or math.hypot(nodeX - gx, nodeY - gy) <= nodeSpacing * 1.1:
            # reconstruct
            seq = []
            cur = state
            while cur is not None:
                nidx, hidx = cur
                px,py,_,_ = nodePositions[nidx]
                seq.append((px,py, idx_to_angle(hidx)))
                cur = parent[cur]
            seq.reverse()
            # attach exact start & goal
            if math.hypot(seq[0][0] - startPose[0], seq[0][1] - startPose[1]) > 1e-6:
                seq.insert(0, (startPose[0], startPose[1], startHeading))
            if math.hypot(seq[-1][0] - goalPose[0], seq[-1][1] - goalPose[1]) > 1e-6:
                seq.append((goalPose[0], goalPose[1], seq[-1][2]))
            return seq

        curExact = parentPos[state]
        curHeading = idx_to_angle(headingIdx)

        for dx,dy in neighborOffsets:
            sx = nodeX + dx; sy = nodeY + dy
            if not (0.0 <= sx <= fieldSize and 0.0 <= sy <= fieldSize): continue
            ix = int(sx / nodeSpacing); iy = int(sy / nodeSpacing)
            if (ix,iy) not in nodeMap: continue
            succIdx = nodeMap[(ix,iy)]
            succPos = (nodePositions[succIdx][0], nodePositions[succIdx][1])

            # Theta* relaxation: try connecting from parentPos if LOS (fast sample_line with step)
            useParent = False
            parState = parent[state]
            if parState is not None and line_of_sight_sample(occ, parentPos[state], succPos, baseStep=min(baseResolution, uncertainty/2.0)):
                basePos = parentPos[state]; baseG = gscore.get(parState, 0.0); baseHeading = idx_to_angle(parState[1]); useParent = True
            else:
                basePos = (nodeX, nodeY); baseG = gscore[state]; baseHeading = curHeading

            moveT, resHeading = time_cost(basePos, succPos, baseHeading)
            tentative = baseG + moveT
            succHeadingIdx = angle_to_idx(resHeading)
            succState = (succIdx, succHeadingIdx)
            if tentative < gscore.get(succState, float('inf')):
                gscore[succState] = tentative
                parent[succState] = state if not useParent else parState
                parentPos[succState] = basePos
                heapq.heappush(openHeap, (tentative + heuristic(succPos), succState))
    return None

# helper LOS sampling using occupancy centers only (faster)
def line_of_sight_sample(occ: np.ndarray, a: Tuple[float,float], b: Tuple[float,float], baseStep: float) -> bool:
    step = baseStep
    for p in sample_line(a, b, step):
        # test center only (nodes already inflated occupancy), sufficient for LOS check here
        c,r = meters_to_grid(p[0], p[1], baseResolution, occ.shape[0])
        if occ[r,c]:
            return False
    return True

# ----------------------------- Smoothing (local, high-res checks) -----------------------------
def smooth_path(path: List[Tuple[float,float,float]], occ: np.ndarray, resolution: float) -> List[Tuple[float,float,float]]:
    if not path or len(path) < 3: return path
    coords = [(p[0], p[1]) for p in path]
    for _ in range(smoothingIterations):
        i = 0
        while i < len(coords) - 2:
            j = len(coords) - 1
            shortened = False
            while j > i + 1:
                a = coords[i]; b = coords[j]
                # full OBB-aware LOS sample only along segment a->b (samples * OBB checks) but limited to this few checks
                if los_with_obb(occ, a, b, resolution):
                    # remove intermediates
                    del coords[i+1:j]
                    shortened = True
                    break
                j -= 1
            if not shortened:
                i += 1
    # rebuild headings
    out = []
    for k in range(len(coords)):
        x,y = coords[k]
        if k < len(coords)-1:
            nx,ny = coords[k+1]; heading = math.atan2(ny - y, nx - x)
        else:
            heading = out[-1][2] if out else 0.0
        out.append((x,y,heading))
    return out

def los_with_obb(occ: np.ndarray, a: Tuple[float,float], b: Tuple[float,float], resolution: float) -> bool:
    """
    Check LOS between a and b where at each sample we check oriented robot OBB collision (conservative).
    This is only called on candidate shortcuts (few calls), so it's acceptable.
    """
    step = min(resolution, uncertainty/2.0)
    for p in sample_line(a, b, step):
        heading = math.atan2(b[1]-a[1], b[0]-a[0]) if (b[0]-a[0] or b[1]-a[1]) else 0.0
        if obb_collides(occ, p, heading, robotHalf, resolution, occ.shape[0]):
            return False
    return True

# ----------------------------- Trajectory -> wheel CSV -----------------------------
def trajectory_to_wheel_log(path: List[Tuple[float,float,float]], occ: np.ndarray, resolution: float) -> pd.DataFrame:
    logs = []  # (t0,t1,vx,vy,wz,heading)
    t = 0.0
    currentHeading = path[0][2]
    for i in range(len(path)-1):
        a = (path[i][0], path[i][1]); b = (path[i+1][0], path[i+1][1])
        dist = math.hypot(b[0]-a[0], b[1]-a[1])
        if dist < 1e-9: continue
        moveT, resHeading = time_cost(a,b,currentHeading)
        dx = b[0] - a[0]; dy = b[1] - a[1]
        moveAngle = math.atan2(dy, dx)
        angDiff = abs(((moveAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
        cosF = abs(math.cos(angDiff))
        speed1 = cosF * forwardSpeed + (1 - cosF) * strafeSpeed
        t1 = dist / max(1e-9, speed1)
        rotT = angDiff / max(1e-9, rotationMax)
        t2 = rotT + dist / forwardSpeed
        if t1 <= t2:
            vx = dx / t1; vy = dy / t1
            logs.append((t, t + t1, vx, vy, 0.0, currentHeading)); t += t1
        else:
            rotDir = 1.0 if ((moveAngle - currentHeading + 2*math.pi) % (2*math.pi)) < math.pi else -1.0
            logs.append((t, t + rotT, 0.0, 0.0, rotDir * rotationMax, currentHeading)); t += rotT
            vx = dx / (dist / forwardSpeed); vy = dy / (dist / forwardSpeed)
            logs.append((t, t + dist/forwardSpeed, vx, vy, 0.0, moveAngle)); t += dist / forwardSpeed
            currentHeading = moveAngle

    # sample and compute wheel RPS
    numSteps = int(math.ceil(t / timeStep)) + 1
    rows = []
    for s in range(numSteps):
        t0 = s * timeStep; t1 = min(t0 + timeStep, t); tm = (t0 + t1) / 2.0
        vx = vy = wz = 0.0; headingAt = 0.0
        for seg in logs:
            if seg[0] <= tm <= seg[1] + 1e-9:
                vx, vy, wz, headingAt = seg[2], seg[3], seg[4], seg[5]; break
        ch = math.cos(-headingAt); sh = math.sin(-headingAt)
        bodyVx = vx * ch - vy * sh; bodyVy = vx * sh + vy * ch
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
        rows.append({"time_start": round(t0,6), "time_end": round(t1,6),
                     "FR": round(rpsFR,6), "FL": round(rpsFL,6),
                     "BR": round(rpsRR,6), "BL": round(rpsRL,6)})
    df = pd.DataFrame(rows); df.to_csv(wheelLogPath, index=False)
    return df

# ----------------------------- Visualization -----------------------------
def visualize(occ: np.ndarray, resolution: float, path: List[Tuple[float,float,float]], outputPath: str,
              snapInterval: float = snapshotInterval):
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    ax.set_xlim(0, fieldSize); ax.set_ylim(0, fieldSize); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.add_patch(patches.Rectangle((0,0), fieldSize, fieldSize, fill=False, edgecolor='black', linewidth=2, zorder=1))
    occImg = occ.astype(np.uint8)
    labeled, ncomp = label(occImg)
    objs = find_objects(labeled)
    for sl in objs:
        if sl is None: continue
        row0,row1 = sl[0].start, sl[0].stop-1; col0,col1 = sl[1].start, sl[1].stop-1
        x0,y0 = grid_to_meters(col0, row1, resolution, occ.shape[0]); x1,y1 = grid_to_meters(col1, row0, resolution, occ.shape[0])
        ax.add_patch(patches.Rectangle((x0,y0), x1-x0, y1-y0, facecolor='#cfeefe', edgecolor='#1177bb', linewidth=1.5, zorder=2))
    if path:
        xs = [p[0] for p in path]; ys = [p[1] for p in path]; ax.plot(xs, ys, color='red', linewidth=2, zorder=3)
    # recreate simple time segments for snapshots (reuse time_cost)
    segments = []
    t = 0.0; currentHeading = path[0][2] if path else 0.0
    for i in range(len(path)-1):
        a=(path[i][0], path[i][1]); b=(path[i+1][0], path[i+1][1])
        dist = math.hypot(b[0]-a[0], b[1]-a[1]); 
        if dist < 1e-9: continue
        moveT, resH = time_cost(a,b,currentHeading)
        dx=b[0]-a[0]; dy=b[1]-a[1]; mAngle = math.atan2(dy,dx)
        angDiff = abs(((mAngle - currentHeading + math.pi) % (2*math.pi)) - math.pi)
        cosF = abs(math.cos(angDiff)); speed1 = cosF * forwardSpeed + (1 - cosF) * strafeSpeed
        t1 = dist / max(1e-9, speed1); rotT = angDiff / max(1e-9, rotationMax); t2 = rotT + dist / forwardSpeed
        if t1 <= t2:
            segments.append((t, t+t1, a, b, currentHeading)); t += t1
        else:
            segments.append((t, t+rotT, a, a, currentHeading)); t += rotT
            segments.append((t, t + dist/forwardSpeed, a, b, mAngle)); t += dist / forwardSpeed; currentHeading = mAngle
    totalTime = t
    sampleTimes = [0.0] if totalTime <= 0 else list(np.arange(0.0, totalTime + 1e-9, snapInterval))
    for st in sampleTimes:
        seg = None
        for s in segments:
            if s[0] <= st <= s[1] + 1e-9: seg = s; break
        if seg is None: continue
        _, _, a, b, heading = seg
        if seg[1] - seg[0] < 1e-9: pos = a
        else:
            alpha = (st - seg[0]) / max(1e-9, (seg[1] - seg[0])); pos = (a[0]*(1-alpha)+b[0]*alpha, a[1]*(1-alpha)+b[1]*alpha)
        cx,cy = pos
        rect = patches.Rectangle((cx - robotHalf, cy - robotHalf), robotSize, robotSize,
                                 edgecolor='black', facecolor='none', linewidth=1.6,
                                 transform=plt.matplotlib.transforms.Affine2D().rotate_around(cx, cy, heading) + ax.transData, zorder=5)
        ax.add_patch(rect)
        ax.arrow(cx, cy, 0.25 * math.cos(heading), 0.25 * math.sin(heading), head_width=0.08, head_length=0.08, fc='black', ec='black', zorder=6)
    plt.title("Planned Path with Robot Outlines and Heading")
    plt.tight_layout(); fig.savefig(outputPath, dpi=200); plt.close(fig)
    return outputPath

# ----------------------------- Main orchestration -----------------------------
def plan_and_run(startPose: Tuple[float,float] = startPoseDefault,
                 goalPose: Tuple[float,float] = goalPoseDefault,
                 rects: Optional[List[Tuple[float,float,float,float]]] = None):
    """
    Main entrypoint. Returns path and DataFrame if successful.
    """
    rects = rects if rects is not None else rectangularObstacles
    occ, gridSize = build_occupancy_grid(baseResolution, includeCenterObs=True, rects=rects)
    # quick projection for start/goal if inside inflated occ
    if not center_point_free(occ, startPose, baseResolution, gridSize):
        # find nearest free grid cell (simple BFS outward)
        found = None
        c0, r0 = meters_to_grid(startPose[0], startPose[1], baseResolution, gridSize)
        for radius in range(0, 60):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    c = c0 + dx; r = r0 + dy
                    if 0 <= c < gridSize and 0 <= r < gridSize and not occ[r,c]:
                        p = grid_to_meters(c, r, baseResolution, gridSize)
                        if not obb_collides(occ, p, 0.0, robotHalf, baseResolution, gridSize):
                            found = p; break
                if found: break
            if found: break
        if not found:
            print("Unsafe start, no nearby free point. Aborting.")
            return None
        print("Projected startPose ->", found); startPose = found
    if not center_point_free(occ, goalPose, baseResolution, gridSize):
        found = None
        c0, r0 = meters_to_grid(goalPose[0], goalPose[1], baseResolution, gridSize)
        for radius in range(0, 60):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    c = c0 + dx; r = r0 + dy
                    if 0 <= c < gridSize and 0 <= r < gridSize and not occ[r,c]:
                        p = grid_to_meters(c, r, baseResolution, gridSize)
                        if not obb_collides(occ, p, 0.0, robotHalf, baseResolution, gridSize):
                            found = p; break
                if found: break
            if found: break
        if not found:
            print("Unsafe goal, no nearby free point. Aborting.")
            return None
        print("Projected goalPose ->", found); goalPose = found

    # adaptively pick nodeSpacing to keep search feasible
    path = theta_star(occ, baseResolution, startPose, goalPose, initialNodeSpacing)
    if path is None:
        print("Planner failed to find a path. Try increasing initialNodeSpacing to simplify graph.")
        return None

    # smoothing (local) with high-res OBB checks
    pathSmoothed = smooth_path(path, occ, baseResolution)

    # verify final path OBB collision-free
    for x,y,h in pathSmoothed:
        if obb_collides(occ, (x,y), h, robotHalf, baseResolution, gridSize):
            print("Collision detected after smoothing. Aborting.")
            return None

    # generate wheel CSV
    df = trajectory_to_wheel_log(pathSmoothed, occ, baseResolution)
    # save visualization
    img = visualize(occ, baseResolution, pathSmoothed, outputImagePath)
    print("Done. Path points:", len(pathSmoothed))
    print("Saved:", outputImagePath, wheelLogPath)
    return pathSmoothed, df

# ----------------------------- Run example -----------------------------
if __name__ == "__main__":
    random.seed(42)
    # ensure demo center obstacle included
    rectangularObstacles = []  # empty to rely on centerObs inside build_occupancy_grid
    res = plan_and_run(startPoseDefault, goalPoseDefault, rectangularObstacles)
    if res is None:
        print("No valid plan generated.")
    else:
        path, df = res
        print(df.head(10).to_string(index=False))
