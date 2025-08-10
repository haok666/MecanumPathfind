import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sqrt, atan2, radians, degrees, pi, cos, sin
from scipy.signal import convolve2d
import time
from tqdm import tqdm
import os

# --- configuration constants (camelCase) ---
fieldSize      = 3.6667      # metres
maxRps         = 4.0         # revolutions per second
robotLength    = 0.40        # metres
robotWidth     = 0.40        # metres
resolution     = 0.005       # metres per grid cell
wheelRadius    = 0.03        # metres
rotationSpeed  = 1 * pi      # rad s-1

# ------------------------------------------------------------------------------#
#                               robot controller                                #
# ------------------------------------------------------------------------------#
class RobotController:
    def __init__(self):
        self.commandHistory = []
        self.currentTime = 0.0

    def move_motor(self, motorName, rpm, duration):
        """Log motor movement command."""
        cmd = {
            "motor": motorName,
            "rpm": rpm,
            "start": self.currentTime,
            "end": self.currentTime + duration,
        }
        self.commandHistory.append(cmd)
        # (send to physical motor in a real robot)
        time.sleep(duration * 0.1)  # simulate at 10 % speed
        self.currentTime += duration


# ------------------------------------------------------------------------------#
#                            low-level helper routines                          #
# ------------------------------------------------------------------------------#
def calculate_mecanum_speed(thetaDeg):
    """Return translational speed (m s-1) achievable in direction thetaDeg."""
    thetaRad = radians(thetaDeg)
    frictionLoss = 0.1 * abs(np.sin(thetaRad))
    efficiency = 0.7 + 0.3 * abs(np.cos(thetaRad))
    return 3.77 * (1 - frictionLoss) * efficiency  # ≈3.77 m s-1 at 4 RPS


def create_obstacle_map(obstacleList):
    """Convert obstacle list to a Boolean grid with clearance expansion."""
    gridSize = int(fieldSize / resolution)
    grid = np.zeros((gridSize, gridSize), dtype=bool)

    for x, y, side in obstacleList:
        minX = max(0, int((x - side / 2) / resolution))
        maxX = min(gridSize - 1, int((x + side / 2) / resolution))
        minY = max(0, int((y - side / 2) / resolution))
        maxY = min(gridSize - 1, int((y + side / 2) / resolution))
        grid[minX : maxX + 1, minY : maxY + 1] = True

    clearance = int(0.297 / resolution)  # 29.7 cm diagonal
    kernel = np.ones((2 * clearance + 1, 2 * clearance + 1))
    return convolve2d(grid, kernel, mode="same") > 0


def is_position_valid(x, y, angle, obstacleMap):
    """True if the robot rectangle fits at (x, y) with heading angle."""
    halfLen = robotLength / 2
    halfWid = robotWidth / 2
    gridSize = obstacleMap.shape[0]

    # four corners in world coordinates
    corners = [
        (
            x + halfLen * np.cos(angle) - halfWid * np.sin(angle),
            y + halfLen * np.sin(angle) + halfWid * np.cos(angle),
        ),
        (
            x + halfLen * np.cos(angle) + halfWid * np.sin(angle),
            y + halfLen * np.sin(angle) - halfWid * np.cos(angle),
        ),
        (
            x - halfLen * np.cos(angle) + halfWid * np.sin(angle),
            y - halfLen * np.sin(angle) - halfWid * np.cos(angle),
        ),
        (
            x - halfLen * np.cos(angle) - halfWid * np.sin(angle),
            y - halfLen * np.sin(angle) + halfWid * np.cos(angle),
        ),
    ]

    # field boundaries
    for cx, cy in corners:
        if not (0 <= cx <= fieldSize and 0 <= cy <= fieldSize):
            return False

    # obstacle check (sample corner cells)
    for cx, cy in corners:
        gx = min(gridSize - 1, max(0, int(cx / resolution)))
        gy = min(gridSize - 1, max(0, int(cy / resolution)))
        if obstacleMap[gx, gy]:
            return False

    return True


# ------------------------------------------------------------------------------#
#                           A* path-finding (grid)                              #
# ------------------------------------------------------------------------------#
def find_optimal_path(startPos, goalPos, obstacleMap, startAngle=0.0):
    """Return list of (gridX, gridY, heading) from start to goal, or None."""
    gridSize = obstacleMap.shape[0]
    startIdx = (int(startPos[0] / resolution), int(startPos[1] / resolution))
    goalIdx = (int(goalPos[0] / resolution), int(goalPos[1] / resolution))

    pq = []
    heappush(pq, (0, 0, *startIdx, startAngle, None))
    visited = set()

    # 8-connected moves
    dirs = [
        (1, 0, 0),
        (-1, 0, pi),
        (0, 1, pi / 2),
        (0, -1, 3 * pi / 2),
        (1, 1, pi / 4),
        (1, -1, 7 * pi / 4),
        (-1, 1, 3 * pi / 4),
        (-1, -1, 5 * pi / 4),
    ]

    with tqdm(total=gridSize * gridSize * 8, desc="A* search") as bar:
        while pq:
            _, cost, x, y, angle, parent = heappop(pq)
            bar.update(1)

            if (x, y) == goalIdx:
                path = []
                while parent:
                    path.append((x, y, angle))
                    x, y, angle, parent = parent
                path.append((x, y, angle))  # add start
                return path[::-1]

            if (x, y, angle) in visited:
                continue
            visited.add((x, y, angle))

            for dx, dy, moveAngle in dirs:
                nx, ny = x + dx, y + dy
                worldX = nx * resolution
                worldY = ny * resolution

                if not is_position_valid(worldX, worldY, moveAngle, obstacleMap):
                    continue

                moveDist = resolution * sqrt(dx ** 2 + dy ** 2)
                moveTime = moveDist / calculate_mecanum_speed(degrees(moveAngle))
                rotTime = abs((moveAngle - angle + pi) % (2 * pi) - pi) / rotationSpeed

                heappush(
                    pq,
                    (
                        cost + moveTime + rotTime,
                        cost + moveTime + rotTime,
                        nx,
                        ny,
                        moveAngle,
                        (x, y, angle, parent),
                    ),
                )
    return None


# ------------------------------------------------------------------------------#
#                       line-of-sight path smoothing                            #
# ------------------------------------------------------------------------------#
def _clear_segment(p1, p2, obstacleMap, step=resolution / 2):
    """True if every sampled point on segment p1-p2 is collision-free."""
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * resolution
    nSteps = max(2, int(dist / step))

    for i in range(1, nSteps):
        t = i / nSteps
        xi = (1 - t) * x1 + t * x2
        yi = (1 - t) * y1 + t * y2
        if not is_position_valid(
            xi * resolution, yi * resolution, 0.0, obstacleMap
        ):
            return False
    return True


def smooth_path(rawPath, obstacleMap):
    """Return pruned list of way-points."""
    if not rawPath:
        return rawPath
    smoothed = [rawPath[0]]
    i = 0
    while i < len(rawPath) - 1:
        j = len(rawPath) - 1
        while j > i + 1:
            if _clear_segment(rawPath[i], rawPath[j], obstacleMap):
                break
            j -= 1
        smoothed.append(rawPath[j])
        i = j
    return smoothed


# ------------------------------------------------------------------------------#
#                         robot motion execution                                #
# ------------------------------------------------------------------------------#
def execute_path(robot, pathCoords):
    """Translate path into motor commands and log them."""
    if len(pathCoords) <= 1:
        return

    robot.commandHistory.clear()
    robot.currentTime = 0.0
    prevX, prevY, prevAngle = pathCoords[0]

    for gx, gy, ang in pathCoords[1:]:
        dx = (gx - prevX) * resolution
        dy = (gy - prevY) * resolution
        dist = sqrt(dx ** 2 + dy ** 2)
        moveAngle = atan2(dy, dx)

        moveTime = dist / calculate_mecanum_speed(degrees(moveAngle))
        rotTime = abs((moveAngle - prevAngle + pi) % (2 * pi) - pi) / rotationSpeed

        # rotation
        if rotTime > 1e-3:
            sign = -1 if moveAngle > prevAngle else 1
            robot.move_motor("FR", sign * maxRps * 60, rotTime)
            robot.move_motor("FL", -sign * maxRps * 60, rotTime)
            robot.move_motor("BR", sign * maxRps * 60, rotTime)
            robot.move_motor("BL", -sign * maxRps * 60, rotTime)

        # translation
        vx, vy = np.cos(moveAngle), np.sin(moveAngle)
        robot.move_motor("FL", (vx - vy) * maxRps * 60, moveTime)
        robot.move_motor("FR", (vx + vy) * maxRps * 60, moveTime)
        robot.move_motor("BL", (vx + vy) * maxRps * 60, moveTime)
        robot.move_motor("BR", (vx - vy) * maxRps * 60, moveTime)

        prevX, prevY, prevAngle = gx, gy, ang


def generate_command_report(robot):
    """Print friendly summary of motor commands."""
    print("\nRobot Movement Log")
    print("{:<20} {:<12} {:<12} {:<12} {:<12}".format(
        "Time (s)", "Front R", "Front L", "Back R", "Back L"
    ))

    spans = {}
    for cmd in robot.commandHistory:
        key = (cmd["start"], cmd["end"])
        spans.setdefault(key, {})[cmd["motor"]] = cmd["rpm"]

    for (start, end), motors in sorted(spans.items()):
        line = "{:05.2f}–{:05.2f}".format(start, end)
        print("{:<20} {:<12.1f} {:<12.1f} {:<12.1f} {:<12.1f}".format(
            line,
            motors.get("FR", 0),
            motors.get("FL", 0),
            motors.get("BR", 0),
            motors.get("BL", 0),
        ))


# ------------------------------------------------------------------------------#
#                           visualisation                                       #
# ------------------------------------------------------------------------------#
def visualize_path(pathCoords, obstacleList, filename="path.png"):
    """Draw field, obstacles, path and robot footprint samples."""
    plt.figure(figsize=(10, 10))
    
    # Set up plot boundaries and grid
    plt.xlim(0, fieldSize)
    plt.ylim(0, fieldSize)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    
    # Draw field boundary
    plt.plot([0, fieldSize, fieldSize, 0, 0],
             [0, 0, fieldSize, fieldSize, 0], "k-", lw=2)
    
    # Draw obstacles with better visualization
    for x, y, side in obstacleList:
        plt.gca().add_patch(
            plt.Rectangle(
                (x - side/2, y - side/2), 
                side, side,
                color='#FF6B6B', alpha=0.3, 
                edgecolor='#FF2626', linewidth=1
            )
        )
    
    if pathCoords:
        # Convert grid coordinates to meters
        xs = [p[0] * resolution for p in pathCoords]
        ys = [p[1] * resolution for p in pathCoords]
        
        # Draw path with better styling
        plt.plot(xs, ys, '#4285F4', linewidth=1.5, alpha=0.9, label='Path Center')
        
        # Draw robot orientation markers
        halfLen = robotLength / 2
        halfWid = robotWidth / 2
        step = max(1, len(pathCoords) // 20)  # Show about 20 robot outlines
        
        for idx in range(0, len(pathCoords), step):
            gx, gy, ang = pathCoords[idx]
            cx, cy = gx * resolution, gy * resolution
            
            # Calculate robot corners
            corners = [
                (cx + halfLen * cos(ang) - halfWid * sin(ang),
                 cy + halfLen * sin(ang) + halfWid * cos(ang)),
                (cx + halfLen * cos(ang) + halfWid * sin(ang),
                 cy + halfLen * sin(ang) - halfWid * cos(ang)),
                (cx - halfLen * cos(ang) + halfWid * sin(ang),
                 cy - halfLen * sin(ang) - halfWid * cos(ang)),
                (cx - halfLen * cos(ang) - halfWid * sin(ang),
                 cy - halfLen * sin(ang) + halfWid * cos(ang)),
                (cx + halfLen * cos(ang) - halfWid * sin(ang),
                 cy + halfLen * sin(ang) + halfWid * cos(ang))
            ]
            
            # Plot robot outline
            outline_x, outline_y = zip(*corners)
            plt.plot(outline_x, outline_y, '#7D3C98', linewidth=0.8, alpha=0.4)
        
        # Mark start and end points
        if len(pathCoords) > 0:
            start_x, start_y = pathCoords[0][0] * resolution, pathCoords[0][1] * resolution
            end_x, end_y = pathCoords[-1][0] * resolution, pathCoords[-1][1] * resolution
            
            plt.scatter(start_x, start_y, color='#0F9D58', s=100, label='Start', 
                       edgecolors='black', linewidths=0.5)
            plt.scatter(end_x, end_y, color='#DB4437', s=100, label='End', 
                       edgecolors='black', linewidths=0.5)
    
    # Add labels and title
    plt.xlabel("X Position (meters)", fontsize=12)
    plt.ylabel("Y Position (meters)", fontsize=12)
    plt.title("Robot Navigation Path", pad=20, fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    
    # Save the figure
    saveDir = "C:/Users/Kenny/Desktop/Coding/FTC"
    os.makedirs(saveDir, exist_ok=True)
    filepath = os.path.join(saveDir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {filepath}")


# ------------------------------------------------------------------------------#
#                           main demo                                           #
# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    obstacleList = [
        (2.0, 1.8, 0.3),
        (1.2, 3.4, 0.4),
        (0.7, 0.9, 0.2),
        (3.3, 0.5, 0.3),
        (1.9, 0.25, 0.2),
    ]
    startPos = (0.25, 0.25)
    goalPos = (3.2, 2.8)
    initialAngle = 0.0  # rad

    print("Building obstacle map …")
    obstacleMap = create_obstacle_map(obstacleList)

    print("Planning …")
    rawPath = find_optimal_path(startPos, goalPos, obstacleMap, initialAngle)
    if not rawPath:
        print("No feasible path.")
        exit()

    path = smooth_path(rawPath, obstacleMap)
    print(f"Path has {len(path)} way-points after smoothing.")

    visualize_path(path, obstacleList)

    robot = RobotController()
    execute_path(robot, path)
    generate_command_report(robot)