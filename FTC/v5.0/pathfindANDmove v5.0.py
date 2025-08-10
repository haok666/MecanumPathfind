"""
Mecanum Theta* pathfinder and wheel-output generator.
Single-file program. Outputs visualization PNG and wheel rotation CSV.

Conventions:
- Variables use camelCase.
- Functions use snake_case.

Edit obstacles or imagePath then run: python mecanum_theta_star.py
"""

import math
import heapq
import time
import csv
from collections import deque
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ------------------- USER EDITABLE PARAMETERS -------------------
fieldSize = 3.6666667  # meters (square side)
marginError = 0.005  # meters (5 mm)
cellSize = 0.02  # meters per grid cell. adjust for speed/precision.

robotWidth = 0.40  # m
robotLength = 0.40  # m
robotHalfDiagonal = math.hypot(robotWidth, robotLength) / 2.0
clearanceMargin = 0.01  # 1 cm required clearance from walls/obstacles

wheelDiameter = 0.08  # m
wheelRadius = wheelDiameter / 2.0
maxRps = 2.5  # max revolutions per second per motor

# directional speed factors
forwardSpeedFactor = 1.00
strafeSpeedFactor = 0.95

# visualization and outputs
outputImagePath = 'mecanum_path_visualization.png'
outputCsvPath = 'wheel_rotations.csv'

# control sampling
dt = 0.1

# optional image map (black=obstacle). Set to None to use obstacles list below.
imagePath = None

# Obstacles: axis-aligned rectangles. each entry: (x_min, y_min, width, height) in metres.
obstacles = [
    # 1m x 1m square centered in field
    ((fieldSize - 1.0) / 2.0, (fieldSize - 1.0) / 2.0, 1.0, 1.0),
    # example extras
    (0.5, 0.3, 0.3, 0.7),
    (2.2, 2.6, 0.5, 0.2),
]

# start and goal positions (robot center). Corner to corner test.
startPose = (robotHalfDiagonal + clearanceMargin + 0.001, robotHalfDiagonal + clearanceMargin + 0.001)
goalPose = (fieldSize - (robotHalfDiagonal + clearanceMargin + 0.001), fieldSize - (robotHalfDiagonal + clearanceMargin + 0.001))
startOrientation = 0.0  # radians

# inflation retry parameters
inflateFactor = 1.0
minInflateFactor = 0.5
inflateSteps = 6

# derived
gridSizeX = int(math.ceil(fieldSize / cellSize))
gridSizeY = gridSizeX
maxLinearSpeed = 2.0 * math.pi * wheelRadius * maxRps  # m/s

# geometry helpers
halfLength = robotLength / 2.0
halfWidth = robotWidth / 2.0

def world_to_grid(x, y):
    gx = int(round(x / cellSize))
    gy = int(round(y / cellSize))
    gx = max(0, min(gridSizeX - 1, gx))
    gy = max(0, min(gridSizeY - 1, gy))
    return gx, gy

def grid_to_world(gx, gy):
    x = gx * cellSize
    y = gy * cellSize
    return x, y

# collision geometry
def rect_polygon(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

def oriented_rect(cx, cy, w, h, theta):
    hw = w / 2.0
    hh = h / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    out = []
    for (dx, dy) in corners:
        x = cx + dx * cosT - dy * sinT
        y = cy + dx * sinT + dy * cosT
        out.append((x, y))
    return out

def polygon_intersect(polyA, polyB):
    def projection(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)
    def norm(dx, dy):
        L = math.hypot(dx, dy)
        if L == 0:
            return (0.0, 0.0)
        return (dx / L, dy / L)
    for poly in (polyA, polyB):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            edge = (x2 - x1, y2 - y1)
            axis = norm(-edge[1], edge[0])
            minA, maxA = projection(polyA, axis)
            minB, maxB = projection(polyB, axis)
            if maxA < minB or maxB < minA:
                return False
    return True

def robot_collision(cx, cy, theta, obstacleRects):
    polyRobot = oriented_rect(cx, cy, robotWidth, robotLength, theta)
    if cx - robotHalfDiagonal - clearanceMargin < 0 or cy - robotHalfDiagonal - clearanceMargin < 0 or cx + robotHalfDiagonal + clearanceMargin > fieldSize or cy + robotHalfDiagonal + clearanceMargin > fieldSize:
        return True
    for (ox, oy, ow, oh) in obstacleRects:
        polyObs = rect_polygon(ox, oy, ow, oh)
        if polygon_intersect(polyRobot, polyObs):
            return True
    return False

# occupancy grid builder
def build_obstacle_map():
    grid = np.zeros((gridSizeX, gridSizeY), dtype=bool)
    minAllowed = clearanceMargin + robotHalfDiagonal
    minCell = int(math.floor(minAllowed / cellSize))
    maxCell = gridSizeX - minCell - 1
    for i in range(gridSizeX):
        for j in range(gridSizeY):
            if i < minCell or j < minCell or i > maxCell or j > maxCell:
                grid[i, j] = True
    if imagePath:
        img = Image.open(imagePath).convert('L')
        img = img.resize((gridSizeX, gridSizeY), Image.NEAREST)
        arr = np.array(img)
        grid |= (arr < 128)
    else:
        inflate = (robotHalfDiagonal + clearanceMargin) * inflateFactor
        for (ox, oy, ow, oh) in obstacles:
            x0 = max(0.0, ox - inflate)
            y0 = max(0.0, oy - inflate)
            x1 = min(fieldSize, ox + ow + inflate)
            y1 = min(fieldSize, oy + oh + inflate)
            gx0, gy0 = world_to_grid(x0, y0)
            gx1, gy1 = world_to_grid(x1, y1)
            grid[gx0:gx1+1, gy0:gy1+1] = True
    return grid

# fast LOS using grid occupancy
def line_of_sight_grid(p1, p2, gridObstacles):
    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    steps = max(2, int(math.ceil(dist / (cellSize * 0.5))))
    for s in range(steps+1):
        t = s / float(steps)
        x = p1[0] + (p2[0]-p1[0]) * t
        y = p1[1] + (p2[1]-p1[1]) * t
        gx, gy = world_to_grid(x, y)
        if gridObstacles[gx, gy]:
            return False
    return True

def save_grid_debug(gridObstacles, path=None):
    img = np.ones((gridSizeY, gridSizeX, 3), dtype=np.uint8) * 255
    for i in range(gridSizeX):
        for j in range(gridSizeY):
            if gridObstacles[i, j]:
                img[j, i] = (200, 0, 0)
    sgx, sgy = world_to_grid(*startPose)
    ggx, ggy = world_to_grid(*goalPose)
    img[sgy, sgx] = (0, 200, 0)
    img[ggy, ggx] = (0, 0, 200)
    if path:
        for (x, y) in path:
            gx, gy = world_to_grid(x, y)
            img[gy, gx] = (0, 0, 0)
    im = Image.fromarray(img)
    im = im.resize((gridSizeX*4, gridSizeY*4), Image.NEAREST)
    im.save('grid_debug.png')
    print('Saved grid_debug.png (red=blocked, green=start, blue=goal)')

def grid_reachable(startNode, goalNode, gridObstacles):
    sx, sy = startNode
    gx, gy = goalNode
    if gridObstacles[sx, sy] or gridObstacles[gx, gy]:
        return False, 0
    q = deque()
    q.append((sx, sy))
    visited = np.zeros_like(gridObstacles, dtype=bool)
    visited[sx, sy] = True
    count = 1
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        cx, cy = q.popleft()
        if (cx, cy) == (gx, gy):
            return True, count
        for dx, dy in neigh:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < gridSizeX and 0 <= ny < gridSizeY and not visited[nx, ny] and not gridObstacles[nx, ny]:
                visited[nx, ny] = True
                q.append((nx, ny))
                count += 1
    return False, count

# Theta* implementation using occupancy LOS
def theta_star(start, goal, gridObstacles):
    startGx, startGy = world_to_grid(*start)
    goalGx, goalGy = world_to_grid(*goal)
    startNode = (startGx, startGy)
    goalNode = (goalGx, goalGy)

    def heuristic(a, b):
        ax, ay = grid_to_world(a[0], a[1])
        bx, by = grid_to_world(b[0], b[1])
        euclid = math.hypot(bx-ax, by-ay)
        return euclid / maxLinearSpeed

    openSet = []
    entryCount = 0
    gScore = {startNode: 0.0}
    parent = {startNode: None}
    fScore = {startNode: heuristic(startNode, goalNode)}
    heapq.heappush(openSet, (fScore[startNode], entryCount, startNode))

    neighborOffsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    maxIter = gridSizeX * gridSizeY * 8
    iters = 0
    while openSet and iters < maxIter:
        iters += 1
        _, _, current = heapq.heappop(openSet)
        if current == goalNode:
            break
        for dx, dy in neighborOffsets:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or ny < 0 or nx >= gridSizeX or ny >= gridSizeY:
                continue
            if gridObstacles[nx, ny]:
                continue
            neighbor = (nx, ny)
            if parent[current] is not None:
                parentNode = parent[current]
                pWorld = grid_to_world(parentNode[0], parentNode[1])
                nWorld = grid_to_world(neighbor[0], neighbor[1])
                if line_of_sight_grid(pWorld, nWorld, gridObstacles):
                    tentative_g = gScore[parentNode] + math.hypot(nWorld[0]-pWorld[0], nWorld[1]-pWorld[1]) / maxLinearSpeed
                    if neighbor not in gScore or tentative_g < gScore[neighbor]:
                        gScore[neighbor] = tentative_g
                        parent[neighbor] = parentNode
                        fScore[neighbor] = tentative_g + heuristic(neighbor, goalNode)
                        entryCount += 1
                        heapq.heappush(openSet, (fScore[neighbor], entryCount, neighbor))
                    continue
            cWorld = grid_to_world(current[0], current[1])
            nWorld = grid_to_world(neighbor[0], neighbor[1])
            dist = math.hypot(nWorld[0]-cWorld[0], nWorld[1]-cWorld[1])
            tentative_g = gScore[current] + dist / maxLinearSpeed
            if neighbor not in gScore or tentative_g < gScore[neighbor]:
                if not line_of_sight_grid(cWorld, nWorld, gridObstacles):
                    continue
                gScore[neighbor] = tentative_g
                parent[neighbor] = current
                fScore[neighbor] = tentative_g + heuristic(neighbor, goalNode)
                entryCount += 1
                heapq.heappush(openSet, (fScore[neighbor], entryCount, neighbor))
    if goalNode not in parent:
        raise RuntimeError('Path not found')
    path = []
    cur = goalNode
    while cur is not None:
        path.append(grid_to_world(cur[0], cur[1]))
        cur = parent[cur]
    path.reverse()
    path[0] = start
    path[-1] = goal
    return path

# smoothing using occupancy LOS
def smooth_path(path, gridObstacles):
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path)-1:
        j = len(path)-1
        success = False
        while j > i:
            if line_of_sight_grid(path[i], path[j], gridObstacles):
                smoothed.append(path[j])
                i = j
                success = True
                break
            j -= 1
        if not success:
            smoothed.append(path[i+1])
            i += 1
    return smoothed

# convert path to wheel rotation timeline
def path_to_wheel_table(path, startOrientation):
    timeline = []
    tNow = 0.0
    currentTheta = startOrientation
    L = robotLength
    W = robotWidth
    r = wheelRadius
    k = (L + W) / 2.0
    for segIdx in range(1, len(path)):
        p0 = path[segIdx-1]
        p1 = path[segIdx]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue
        angleGlobal = math.atan2(dy, dx)
        relAngle = angleGlobal - currentTheta
        vx = maxLinearSpeed * math.cos(relAngle) * forwardSpeedFactor
        vy = -maxLinearSpeed * math.sin(relAngle) * strafeSpeedFactor
        nominalSpeed = max(1e-6, math.hypot(vx, vy))
        tSeg = dist / nominalSpeed
        numSteps = max(1, int(math.ceil(tSeg / dt)))
        stepDt = tSeg / numSteps
        for s in range(numSteps):
            omega = 0.0
            frRad = (vx - vy - k*omega) / r
            flRad = (vx + vy + k*omega) / r
            brRad = (vx + vy - k*omega) / r
            blRad = (vx - vy + k*omega) / r
            frRps = frRad / (2.0 * math.pi)
            flRps = flRad / (2.0 * math.pi)
            brRps = brRad / (2.0 * math.pi)
            blRps = blRad / (2.0 * math.pi)
            maxMag = max(abs(frRps), abs(flRps), abs(brRps), abs(blRps), 1e-12)
            if maxMag > maxRps:
                scale = maxRps / maxMag
                frRps *= scale
                flRps *= scale
                brRps *= scale
                blRps *= scale
            timeline.append((tNow, tNow + stepDt, frRps, flRps, brRps, blRps))
            tNow += stepDt
    return timeline

# visualization
def draw_visualization(path, obstacleRects):
    dpi = 200
    figSize = 6
    fig = plt.figure(figsize=(figSize, figSize))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(0, fieldSize)
    ax.set_ylim(0, fieldSize)
    ax.plot([0, fieldSize, fieldSize, 0, 0], [0, 0, fieldSize, fieldSize, 0], color='black')
    for (ox, oy, ow, oh) in obstacleRects:
        rect = plt.Rectangle((ox, oy), ow, oh, fill=False, edgecolor='red')
        ax.add_patch(rect)
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, '-o', linewidth=1)
    for idx, p in enumerate(path):
        theta = startOrientation
        poly = oriented_rect(p[0], p[1], robotWidth, robotLength, theta)
        polyX = [v[0] for v in poly] + [poly[0][0]]
        polyY = [v[1] for v in poly] + [poly[0][1]]
        ax.plot(polyX, polyY, color='blue', alpha=0.6)
        headingX = p[0] + 0.5 * robotLength * math.cos(theta)
        headingY = p[1] + 0.5 * robotLength * math.sin(theta)
        ax.arrow(p[0], p[1], headingX - p[0], headingY - p[1], head_width=0.02, head_length=0.03)
    plt.title('Mecanum Theta* path')
    plt.savefig(outputImagePath, dpi=dpi)
    plt.close(fig)

def write_wheel_csv(timeline):
    with open(outputCsvPath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_start', 'time_end', 'FR_rps', 'FL_rps', 'BR_rps', 'BL_rps'])
        for row in timeline:
            writer.writerow([f'{row[0]:.3f}', f'{row[1]:.3f}', f'{row[2]:.6f}', f'{row[3]:.6f}', f'{row[4]:.6f}', f'{row[5]:.6f}'])
    print('Wrote wheel table to', outputCsvPath)

# main with inflation retry and diagnostics
def main():
    global inflateFactor
    t0 = time.time()
    found = False
    tried = []
    for step in range(inflateSteps):
        inflateFactor = 1.0 - step * (1.0 - minInflateFactor) / max(1, inflateSteps-1)
        gridObstacles = build_obstacle_map()
        obstacleRects = list(obstacles)
        obstacleRects.append(( -1.0, -1.0, fieldSize + 2.0, 1.0))
        obstacleRects.append(( -1.0, fieldSize, fieldSize + 2.0, 1.0))
        obstacleRects.append(( -1.0, -1.0, 1.0, fieldSize + 2.0))
        obstacleRects.append((fieldSize, -1.0, 1.0, fieldSize + 2.0))

        sgx, sgy = world_to_grid(*startPose)
        ggx, ggy = world_to_grid(*goalPose)
        print(f'inflateFactor={inflateFactor:.3f} startGrid=({sgx},{sgy}) blocked={gridObstacles[sgx,sgy]} goalGrid=({ggx},{ggy}) blocked={gridObstacles[ggx,ggy]}')
        save_grid_debug(gridObstacles)
        reachable, regionSize = grid_reachable((sgx, sgy), (ggx, ggy), gridObstacles)
        print('Grid connectivity test: reachable=', reachable, 'reachableCellCount=', regionSize)
        if not reachable:
            tried.append(inflateFactor)
            print('Occupancy grid disconnected. Reducing inflation and retrying.')
            continue
        if robot_collision(startPose[0], startPose[1], startOrientation, obstacleRects):
            raise RuntimeError('Start pose collides geometrically')
        if robot_collision(goalPose[0], goalPose[1], startOrientation, obstacleRects):
            raise RuntimeError('Goal pose collides geometrically')
        try:
            path = theta_star(startPose, goalPose, gridObstacles)
            print('Path found with inflateFactor=', inflateFactor)
            found = True
            break
        except RuntimeError:
            tried.append(inflateFactor)
            print('theta_star failed at inflateFactor=', inflateFactor)
            continue
    if not found:
        raise RuntimeError(f'Path not found after trying inflate factors {tried}')
    smoothed = smooth_path(path, gridObstacles)
    draw_visualization(smoothed, obstacleRects)
    timeline = path_to_wheel_table(smoothed, startOrientation)
    write_wheel_csv(timeline)
    print('Total runtime (s):', time.time() - t0)

if __name__ == '__main__':
    main()
