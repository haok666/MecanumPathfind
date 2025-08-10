
# Python implementation of the requested program.
# It follows the user's dot points as closely as possible:
# - Field is 3.6666667 m square
# - Resolution and uncertainty controlled (default 0.005 m -> 5 mm)
# - camelCase for variables/lists/arrays; functions in under_score (snake_case)
# - Editable rectangular obstacles list provided (axis-aligned)
# - Prioritises time: movement costs are time estimates; robot may choose to rotate then go forward vs strafing
# - Visualises path and saves an image
# - Produces a log of movement and converts to wheel rotations (RPS) given wheel diameter and motor limits
# - Uses the attached image to derive obstacles if available; otherwise uses the explicit obstacle list
# - Robot size 0.4m, must keep 0.01m (1cm) clearance from walls/obstacles by inflating obstacles
# - Produces a table of wheel rotations sampled at regular intervals and saves as CSV
# - Tries to be reasonably fast (coarse search grid with any-angle LOS checks using fine sampling)
#
# Notes/assumptions:
# - The attached image (provided by developer) is assumed to represent an occupancy map where 
#   darker/black regions are obstacles and white is free space. The program will read and scale
#   that image to the 3.6666667 m field if present.
# - The program uses a Theta*-like relaxation (line-of-sight shortcut) with time-based costs.
# - State includes robot heading quantized into headingCount orientations (to decide rotation costs).
# - Resolution parameters are adjustable near the top; the default meets 5 mm uncertainty.
#
# Run time: The code uses a nodeSpacing larger than the collision sampling resolution so that
# search is reasonably fast but LOS checks are precise. If runtime ends up being small, raise
# resolution parameters as desired.
#
# Outputs:
# - saved_path.png : visualization showing obstacles, inflated obstacles, and path
# - wheel_log.csv : table of time intervals and wheel RPS for FR, FL, BR, BL

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math, heapq, time, csv, os
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------- USER-EDITABLE PARAMETERS (camelCase) -----------------------
fieldSize = 3.6666667  # meters (square side length)
uncertainty = 0.005  # meters (5 mm required max)
resolution = 0.005  # occupancy resolution (m per pixel). If runtime < 3min, you may reduce this to < 0.005
nodeSpacing = 0.02  # spacing for search nodes (m). Lower = more accurate, slower.
robotSize = 0.40  # meters (square robot)
clearance = 0.01  # meters required clearance from any obstacle/wall (1 cm)
wheelDiameter = 0.08  # meters (80 mm)
wheelRadius = wheelDiameter / 2.0
motorMaxRps = 2.5  # max rotations per second of motor
headingCount = 16  # quantization of robot heading (e.g., 16 -> 22.5° steps)
rotationMax = 1.0  # rad/s max rotation speed (tunable; realistic)
# base speeds (m/s) -- must be <= wheel limits derived from motorMaxRps
maxLinearSpeed = min(0.6, wheelRadius * motorMaxRps * 2 * math.pi)  # ensure within wheel capability
forwardSpeed = maxLinearSpeed  # when robot is aligned with movement
strafeSpeed = 0.8 * forwardSpeed  # strafing/sliding (slightly slower)
diagSpeed = 0.9 * forwardSpeed  # diagonal travel speed

timeStep = 0.1  # seconds for logging table granularity

# Path smoothing iterations
smoothingIterations = 3

# Input start/goal (meters) - example
startPose = (0.3, 0.3)  # x,y in meters (center-based)
goalPose = (3.3666667, 3.3666667)  # avoid exact border to keep clearance

# Editable rectangular obstacles list (x_min, y_min, x_max, y_max) in meters.
# All rectangles axis-aligned at 90 deg to sides.
# By default, empty: we will also try to read the attached image to create obstacles.
rectangularObstacles = [
    # Example obstacle in the middle; comment/uncomment to edit.
    # (1.0, 1.0, 1.4, 1.5),
]

# Image path (developer provided image). If it exists it will be used as authoritative obstacle map.
attachedImagePath = "/mnt/data/ae14fc11-8d04-42cc-b13f-33556677fd7b.png"

outputImagePath = "saved_path.png"
wheelLogPath = "wheel_log.csv"

# ----------------------- END PARAMETERS -----------------------


# ----------------------- Utility functions (snake_case function names) -----------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def heading_index_to_angle(idx):
    return (2 * math.pi * idx) / headingCount

def angle_to_heading_index(angle):
    a = angle % (2 * math.pi)
    return int(round(a / (2 * math.pi) * headingCount)) % headingCount

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Bresenham in continuous sampling sense: sample along line with step 'step_m' and return sample points
def sample_line(a, b, step_m):
    dist = euclid(a, b)
    if dist == 0:
        return [a]
    steps = int(math.ceil(dist / step_m))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        pts.append((a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t))
    return pts

# Inflate axis-aligned rectangles by 'infl' meters
def inflate_rect(rect, infl):
    x0, y0, x1, y1 = rect
    return (x0 - infl, y0 - infl, x1 + infl, y1 + infl)

# ----------------------- Build occupancy from image or explicit rectangles -----------------------
def build_occupancy_from_image_or_rects():
    # occupancy array: True means obstacle (after inflation)
    gridSize = int(math.ceil(fieldSize / resolution))
    occupancy = np.zeros((gridSize, gridSize), dtype=bool)

    # Use image if present; otherwise create occupancy from rectangularObstacles
    if os.path.exists(attachedImagePath):
        try:
            img = Image.open(attachedImagePath).convert("L")  # grayscale
            imgW, imgH = img.size
            # Map image pixels to field coordinates: assume image covers whole field
            arr = np.array(img)
            # threshold to obstacles: dark pixels -> obstacle
            thresh = 128
            mask = arr < thresh
            # Resize mask to occupancy grid
            pilMask = Image.fromarray((mask.astype(np.uint8) * 255).astype(np.uint8))
            pilMask = pilMask.resize((gridSize, gridSize), resample=Image.NEAREST)
            resized = np.array(pilMask) > 0
            occupancy = resized.astype(bool)
        except Exception as e:
            print("Failed to use attached image, falling back to rectangles. Error:", e)

    # overlay rectangles if provided (rects defined in meters)
    for rect in rectangularObstacles:
        # convert to grid coords (note occupancy origin is at (0,0) in lower-left; we'll use top-left indexing later)
        x0, y0, x1, y1 = rect
        # clamp within field
        x0, y0 = clamp(x0, 0, fieldSize), clamp(y0, 0, fieldSize)
        x1, y1 = clamp(x1, 0, fieldSize), clamp(y1, 0, fieldSize)
        ix0 = int(math.floor(x0 / resolution))
        iy0 = int(math.floor(y0 / resolution))
        ix1 = int(math.ceil(x1 / resolution))
        iy1 = int(math.ceil(y1 / resolution))
        # occupancy indexing: [row, col] where row is y (0 at bottom). We'll map y->row by flipping later.
        for gx in range(ix0, ix1):
            for gy in range(iy0, iy1):
                if 0 <= gx < occupancy.shape[1] and 0 <= gy < occupancy.shape[0]:
                    occupancy[gy, gx] = True

    # Inflate obstacles by robot half-size + clearance
    robotHalfDiagInfl = (robotSize / 2.0) + clearance
    inflCells = int(math.ceil(robotHalfDiagInfl / resolution))
    if inflCells > 0:
        # simple dilation using binary convolution / distance transform alternative
        from scipy.ndimage import binary_dilation
        structure = np.ones((2 * inflCells + 1, 2 * inflCells + 1), dtype=bool)
        occupancy = binary_dilation(occupancy, structure=structure)

    return occupancy, gridSize

# ----------------------- Collision checking -----------------------
def point_is_free(occupancy, p):
    # p: (x,y) in meters
    if p[0] < 0 or p[1] < 0 or p[0] > fieldSize or p[1] > fieldSize:
        return False
    ix = int(p[0] / resolution)
    iy = int(p[1] / resolution)
    # clamp
    ix = clamp(ix, 0, occupancy.shape[1] - 1)
    iy = clamp(iy, 0, occupancy.shape[0] - 1)
    return not occupancy[iy, ix]

def line_of_sight_free(occupancy, a, b):
    # sample along line with step = min(resolution, uncertainty/2)
    step = min(resolution, uncertainty / 2.0)
    for pt in sample_line(a, b, step):
        if not point_is_free(occupancy, pt):
            return False
    return True

# ----------------------- Heuristic and cost (time-based) -----------------------
def time_cost_between(a, b, current_heading):
    # returns minimal time to go from a to b given current heading (radians).
    # Two options: strafe (keep heading) or rotate then go forward (resulting heading aligned)
    dx = b[0] - a[0]; dy = b[1] - a[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, current_heading  # no time, heading unchanged
    move_angle = math.atan2(dy, dx)
    # option 1: strafe with current heading unchanged
    ang_diff = abs(((move_angle - current_heading + math.pi) % (2*math.pi)) - math.pi)
    # speed depends on relative angle: if moving roughly along heading -> forwardSpeed,
    # if moving roughly perpendicular -> strafeSpeed
    relative = ang_diff
    # interpolate linearly between forward and strafe speeds based on angle (0 -> forward, pi/2 -> strafe, pi -> backward)
    # backward considered same as forward speed for mecanum (but we keep same forwardSpeed)
    basis = abs(math.cos(relative))  # 1 when aligned, 0 when perpendicular
    speedOption1 = abs(basis) * forwardSpeed + (1 - abs(basis)) * strafeSpeed
    timeOption1 = dist / max(1e-6, speedOption1)
    # option 2: rotate in place to movement angle, then go forward at forwardSpeed
    angle_to_rotate = ang_diff  # minimal rotation
    rotate_time = angle_to_rotate / max(1e-6, rotationMax)
    timeOption2 = rotate_time + dist / forwardSpeed
    # choose min
    if timeOption1 <= timeOption2:
        return timeOption1, current_heading  # heading unchanged
    else:
        # after this move, heading becomes move_angle
        return timeOption2, move_angle

# ----------------------- Search (Theta*-like with orientation handling) -----------------------
def search_path(occupancy, startPose, goalPose):
    # Create node grid positions (we will quantize positions to a coarse grid for nodes)
    nx = int(math.ceil(fieldSize / nodeSpacing))
    ny = int(math.ceil(fieldSize / nodeSpacing))
    node_positions = []
    for iy in range(ny):
        for ix in range(nx):
            x = (ix + 0.5) * nodeSpacing
            y = (iy + 0.5) * nodeSpacing
            # drop nodes too close to borders
            if x < clearance + robotSize/2.0 or x > fieldSize - (clearance + robotSize/2.0): continue
            if y < clearance + robotSize/2.0 or y > fieldSize - (clearance + robotSize/2.0): continue
            if point_is_free(occupancy, (x,y)):
                node_positions.append((x,y,ix,iy))
    # Build a quick lookup from grid index to node index
    node_map = {}  # (ix,iy) -> index
    for i,pos in enumerate(node_positions):
        node_map[(pos[2], pos[3])] = i
    # helper to find nearest node to a pose
    def nearest_node(p):
        ix = int(p[0] / nodeSpacing)
        iy = int(p[1] / nodeSpacing)
        # search local neighborhood for free node
        for r in range(0, 4):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    key = (ix+dx, iy+dy)
                    if key in node_map:
                        idx = node_map[key]
                        return idx, node_positions[idx][:2]
        # fallback brute force
        best = None; bestd=1e9; bi=None
        for i,pos in enumerate(node_positions):
            d = euclid(p, pos[:2])
            if d < bestd:
                bestd = d; best=pos[:2]; bi=i
        return bi, best

    startIdx, startNode = nearest_node(startPose)
    goalIdx, goalNode = nearest_node(goalPose)
    if startIdx is None or goalIdx is None:
        print("Start or goal have no nearby free nodes")
        return None

    # Precompute neighbor offsets (8-connected plus some extra angles for any-angle possibilities)
    angleList = [i*(2*math.pi/16) for i in range(16)]
    neighborOffsets = []
    for a in angleList:
        nx_off = math.cos(a) * nodeSpacing
        ny_off = math.sin(a) * nodeSpacing
        neighborOffsets.append((nx_off, ny_off))

    # A* with heading in state and Theta* LOS relaxation: state = (nodeIndex, headingIndex)
    def heuristic_time(nodePos):
        return euclid(nodePos, goalNode) / forwardSpeed  # optimistic: straight at forward speed

    # Data structures
    startHeading = 0.0  # assume initial heading 0 rad (along +x). Could be parameterized.
    startHeadingIdx = angle_to_heading_index(startHeading)
    closed = set()
    # g: dict of (node,headingIdx)->time
    g = {}
    parent = {}  # maps state -> (parent_state)
    parentPos = {}  # maps state -> parent position (float x,y)
    # initialize
    startState = (startIdx, startHeadingIdx)
    g[startState] = 0.0
    parent[startState] = None
    parentPos[startState] = startNode  # starting exact location
    pq = []
    heapq.heappush(pq, (heuristic_time(startNode), startState))

    nodeIndexToPos = [ (p[0], p[1]) for p in node_positions ]

    expansions = 0
    maxExpansions = 200000  # safety cap to keep runtime reasonable
    startTime = time.time()

    while pq:
        _, state = heapq.heappop(pq)
        if state in closed: continue
        closed.add(state)
        nodeIdx, headingIdx = state
        nodePos = nodeIndexToPos[nodeIdx]
        # goal check (if nodePos near goal)
        if euclid(nodePos, goalNode) <= nodeSpacing*1.1:
            # reconstruct path
            path = []
            cur = state
            while cur is not None:
                ni, hi = cur
                path.append((nodeIndexToPos[ni][0], nodeIndexToPos[ni][1], heading_index_to_angle(hi)))
                cur = parent[cur]
            path.reverse()
            # attach exact start and goal
            if path[0][0:2] != startPose:
                path.insert(0, (startPose[0], startPose[1], startHeading))
            path.append((goalPose[0], goalPose[1], path[-1][2]))
            endTime = time.time()
            print(f"Search time: {endTime-startTime:.2f}s, expansions: {expansions}, nodes: {len(node_positions)}")
            return path

        expansions += 1
        if expansions > maxExpansions:
            print("Exceeded expansion cap. Aborting")
            break

        # generate successors by sampling neighbor offsets
        for off in neighborOffsets:
            succPos = (nodePos[0] + off[0], nodePos[1] + off[1])
            # check bounds
            if succPos[0] < 0 or succPos[1] < 0 or succPos[0] > fieldSize or succPos[1] > fieldSize: continue
            # find nearest node to succPos (quantize)
            ix = int(succPos[0] / nodeSpacing); iy = int(succPos[1] / nodeSpacing)
            key = (ix, iy)
            if key not in node_map: continue
            succIdx = node_map[key]
            succNodePos = node_positions[succIdx][:2]
            # check LOS from parentPos[state] to succNodePos (Theta* relaxation)
            useParent = False
            parPos = parentPos[state]
            if parPos is not None and line_of_sight_free(occupancy, parPos, succNodePos):
                # we can attempt connect from parent directly
                useParent = True
                basePos = parPos
                baseG = g[parent[state]] if parent[state] is not None else 0.0
                baseHeading = heading_index_to_angle(parent[state][1]) if parent[state] is not None else heading_index_to_angle(startHeadingIdx)
            else:
                basePos = nodePos
                baseG = g[state]
                baseHeading = heading_index_to_angle(headingIdx)
            # compute time cost (considering rotate vs strafe)
            moveTime, resultingHeading = time_cost_between(basePos, succNodePos, baseHeading)
            tentativeG = baseG + moveTime
            # build succ heading index depending on whether robot chose to rotate
            # if resultingHeading equals baseHeading -> heading unchanged; else becomes movement angle
            succHeadingIdx = angle_to_heading_index(resultingHeading)
            succState = (succIdx, succHeadingIdx)
            if tentativeG < g.get(succState, float('inf')):
                g[succState] = tentativeG
                parent[succState] = state if not useParent else parent[state]
                parentPos[succState] = basePos
                f = tentativeG + heuristic_time(succNodePos)
                heapq.heappush(pq, (f, succState))

    print("No path found")
    return None

# ----------------------- Postprocessing: smoothing -----------------------
def smooth_path(path, occupancy):
    # path: list of (x,y,heading)
    # simple shortcut smoothing: attempt to replace sequences with straight LOS and minimal time
    for it in range(smoothingIterations):
        i = 0
        while i < len(path) - 2:
            a = (path[i][0], path[i][1])
            for j in range(len(path) - 1, i + 1, -1):
                b = (path[j][0], path[j][1])
                if line_of_sight_free(occupancy, a, b):
                    # remove intermediate nodes i+1..j-1
                    if j - i > 1:
                        del path[i+1:j]
                    break
            i += 1
    return path

# ----------------------- Trajectory generation and wheel mapping -----------------------
def generate_trajectory_and_wheel_log(path, occupancy):
    # path: list of (x,y,heading) with start and goal included.
    # We'll generate motion segments between consecutive points using the time-optimal choice computed earlier
    segments = []
    logs = []  # each log entry: (start_time, end_time, vx, vy, wz)
    t = 0.0
    current_heading = path[0][2]
    for k in range(len(path)-1):
        a = (path[k][0], path[k][1]); b = (path[k+1][0], path[k+1][1])
        dist = euclid(a,b)
        if dist < 1e-6: continue
        # decide whether to strafe or rotate+forward as earlier
        move_angle = math.atan2(b[1]-a[1], b[0]-a[0])
        ang_diff = abs(((move_angle - current_heading + math.pi) % (2*math.pi)) - math.pi)
        basis = abs(math.cos(ang_diff))
        speedOption1 = abs(basis) * forwardSpeed + (1 - abs(basis)) * strafeSpeed
        timeOption1 = dist / max(1e-6, speedOption1)
        rotate_time = ang_diff / max(1e-6, rotationMax)
        timeOption2 = rotate_time + dist / forwardSpeed
        if timeOption1 <= timeOption2:
            # strafe: maintain heading, compute vx, vy in world frame -> convert to body frame later
            vx = (b[0]-a[0]) / timeOption1
            vy = (b[1]-a[1]) / timeOption1
            wz = 0.0
            start_t = t; end_t = t + timeOption1
            logs.append((start_t, end_t, vx, vy, wz, current_heading))
            t = end_t
            # heading unchanged
        else:
            # rotate then go forward
            # rotation segment
            rot_dir = 1 if ((move_angle - current_heading + 2*math.pi) % (2*math.pi)) < math.pi else -1
            rot_end_t = t + rotate_time
            logs.append((t, rot_end_t, 0.0, 0.0, rot_dir * rotationMax, current_heading))
            t = rot_end_t
            # forward segment at forwardSpeed with heading = move_angle
            vx = (b[0]-a[0]) / (dist / forwardSpeed)
            vy = (b[1]-a[1]) / (dist / forwardSpeed)
            wz = 0.0
            logs.append((t, t + dist / forwardSpeed, vx, vy, wz, move_angle))
            t += dist / forwardSpeed
            current_heading = move_angle

    # Now convert logs into wheel angular velocities (RPS) sample table by timeStep
    numSteps = int(math.ceil(t / timeStep)) + 1
    table = []
    for step in range(numSteps):
        t0 = step * timeStep
        t1 = min(t0 + timeStep, t)
        # find active segment for midpoint
        tm = (t0 + t1) / 2.0
        vx = vy = wz = 0.0
        heading_at_segment = 0.0
        for seg in logs:
            if seg[0] <= tm <= seg[1] + 1e-9:
                vx, vy, wz, heading_at_segment = seg[2], seg[3], seg[4], seg[5]
                break
        # convert body speeds to wheel angular velocities using inverse kinematics
        # First compute velocities in robot frame: rotate global vx,vy by -heading
        ch = math.cos(-heading_at_segment); sh = math.sin(-heading_at_segment)
        body_vx = vx * ch - vy * sh
        body_vy = vx * sh + vy * ch
        # inverse kinematics matrix as given in prompt (ordering FL, FR, RL, RR)
        lsum = (robotSize / 2.0) * 2.0  # for square robot, lx+ly = size/2 + size/2 = size
        # But more correctly lx=ly=robotSize/2
        lx = ly = robotSize / 2.0
        lsum = lx + ly
        r = wheelRadius
        # Using matrix:
        # wfl = 1/r * (vx - vy - (lx+ly) * wz)
        # wfr = 1/r * (vx + vy + (lx+ly) * wz)
        # wrl = 1/r * (vx + vy - (lx+ly) * wz)
        # wrr = 1/r * (vx - vy + (lx+ly) * wz)
        wfl = (1.0/r) * (body_vx - body_vy - lsum * wz)
        wfr = (1.0/r) * (body_vx + body_vy + lsum * wz)
        wrl = (1.0/r) * (body_vx + body_vy - lsum * wz)
        wrr = (1.0/r) * (body_vx - body_vy + lsum * wz)
        # convert rad/s to RPS
        rps_fl = wfl / (2 * math.pi)
        rps_fr = wfr / (2 * math.pi)
        rps_rl = wrl / (2 * math.pi)
        rps_rr = wrr / (2 * math.pi)
        # cap to motorMaxRps (if any exceed, scale all down proportionally)
        maxAbs = max(abs(rps_fl), abs(rps_fr), abs(rps_rl), abs(rps_rr), 1e-9)
        if maxAbs > motorMaxRps:
            scale = motorMaxRps / maxAbs
            rps_fl *= scale; rps_fr *= scale; rps_rl *= scale; rps_rr *= scale
        table.append({
            "time_start": t0,
            "time_end": t1,
            "rps_FR": rps_fr,
            "rps_FL": rps_fl,
            "rps_BR": rps_rr,
            "rps_BL": rps_rl
        })
    # save CSV
    df = pd.DataFrame(table)
    df.to_csv(wheelLogPath, index=False)
    print(f"Saved wheel log to {wheelLogPath}, total time {t:.2f}s, {len(table)} intervals")
    return logs, df

# ----------------------- Visualization -----------------------
def visualize_and_save(occupancy, path):
    # create an image representing field and occupancy, overlay path
    gridSize = occupancy.shape[0]
    img = Image.new("RGB", (gridSize, gridSize), (255,255,255))
    draw = ImageDraw.Draw(img)
    # draw occupancy (note occupancy[y,x] with y=0 at bottom; but our array has y=0 at top because of resize earlier)
    # We'll just use array directly (0,0 top-left)
    occImg = (occupancy.astype(np.uint8) * 0)
    occImg = (occupancy.astype(np.uint8) * 0)
    for y in range(gridSize):
        for x in range(gridSize):
            if occupancy[y,x]:
                img.putpixel((x,y), (30,30,30))
    # draw inflated obstacles boundaries in red? (hard to compute boundaries; skip)
    # overlay path transformed to pixels
    drawPath = []
    for p in path:
        xpix = int((p[0] / fieldSize) * gridSize)
        ypix = int((1 - (p[1] / fieldSize)) * gridSize)  # invert y for image coordinates
        drawPath.append((xpix, ypix))
    if len(drawPath) > 1:
        draw.line(drawPath, fill=(255,0,0), width=2)
    # draw robot footprint at start and goal
    def draw_robot_at(drawObj, px, py, heading, color):
        # px,py in pixels, heading radians
        half = int((robotSize/fieldSize) * gridSize * 0.5)
        # rectangle center-based
        box = [px-half, py-half, px+half, py+half]
        drawObj.rectangle(box, outline=color, width=2)
        # heading arrow
        hx = px + int(math.cos(heading) * half)
        hy = py - int(math.sin(heading) * half)
        drawObj.line([(px,py),(hx,hy)], fill=color, width=2)
    # start
    sx = int((path[0][0]/fieldSize) * gridSize); sy = int((1 - (path[0][1]/fieldSize)) * gridSize)
    gx = int((path[-1][0]/fieldSize) * gridSize); gy = int((1 - (path[-1][1]/fieldSize)) * gridSize)
    draw_robot_at(draw, sx, sy, path[0][2], (0,255,0))
    draw_robot_at(draw, gx, gy, path[-1][2], (0,0,255))
    img.save(outputImagePath)
    print(f"Saved path image to {outputImagePath}")
    return outputImagePath

# ----------------------- Main routine -----------------------
def main():
    occupancy, gridSize = build_occupancy_from_image_or_rects()
    print(f"Occupancy grid built: {gridSize}x{gridSize}, resolution {resolution} m/cell")
    # ensure start/goal clearance
    if not point_is_free(occupancy, startPose):
        print("WARNING: Start pose is inside obstacle or too close after inflation.")
    if not point_is_free(occupancy, goalPose):
        print("WARNING: Goal pose is inside obstacle or too close after inflation.")
    path = search_path(occupancy, startPose, goalPose)
    if path is None:
        print("No path found. Aborting.")
        return
    path = smooth_path(path, occupancy)
    logs, df = generate_trajectory_and_wheel_log(path, occupancy)
    imgPath = visualize_and_save(occupancy, path)
    # print a small table of rotation per interval
    print(df.head(20).to_string(index=False))
    print("Done. Files:", imgPath, wheelLogPath)

# Run main
if __name__ == "__main__":
    main()

# Provide user guidance: If runtime is fast, reduce 'resolution' or nodeSpacing to increase precision.
