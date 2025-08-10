import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sqrt, atan2, radians, degrees, pi
from scipy.signal import convolve2d
import time
from tqdm import tqdm

# Constants
FIELD_SIZE = 3.6667
MAX_RPS = 4.0
robotDims = (0.423, 0.415)  # camelCase for variables
resolution = 0.005
wheelRadius = 0.03
rotationSpeed = 2 * pi

class Robot:
    def movement(self, motor, rpm, duration):
        pass  # Hardware interface

def calculate_mecanum_speed(theta_deg):  # under_score for functions
    theta = radians(theta_deg)
    return 3.77 * (1 - 0.1*abs(np.sin(theta))) * (0.7 + 0.3*abs(np.cos(theta)))

def create_obstacle_map(obstacles):
    gridDim = int(FIELD_SIZE / resolution)
    obstacleMap = np.zeros((gridDim, gridDim), dtype=bool)
    
    with tqdm(total=len(obstacles), desc="Mapping obstacles") as pbar:
        for x, y, side in obstacles:
            minX = max(0, int((x-side/2)/resolution))
            maxX = min(gridDim-1, int((x+side/2)/resolution))
            minY = max(0, int((y-side/2)/resolution))
            maxY = min(gridDim-1, int((y+side/2)/resolution))
            obstacleMap[minX:maxX+1, minY:maxY+1] = True
            pbar.update(1)
    
    clearance = int(0.297 / resolution)
    kernel = np.ones((2*clearance+1, 2*clearance+1))
    return convolve2d(obstacleMap, kernel, mode='same') > 0

def is_robot_in_bounds(x, y, angle):
    halfLength = robotDims[0]/2
    halfWidth = robotDims[1]/2
    
    corners = [
        (x + halfLength*np.cos(angle) - halfWidth*np.sin(angle),
         y + halfLength*np.sin(angle) + halfWidth*np.cos(angle)),
        (x + halfLength*np.cos(angle) + halfWidth*np.sin(angle),
         y + halfLength*np.sin(angle) - halfWidth*np.cos(angle)),
        (x - halfLength*np.cos(angle) + halfWidth*np.sin(angle),
         y - halfLength*np.sin(angle) - halfWidth*np.cos(angle)),
        (x - halfLength*np.cos(angle) - halfWidth*np.sin(angle),
         y - halfLength*np.sin(angle) + halfWidth*np.cos(angle))
    ]
    
    for (cx, cy) in corners:
        if not (0 <= cx <= FIELD_SIZE and 0 <= cy <= FIELD_SIZE):
            return False
    return True

def find_optimal_path(start, goal, obstacleMap, startAngle=0):
    gridDim = obstacleMap.shape[0]
    startIdx = (int(start[0]/resolution), int(start[1]/resolution))
    goalIdx = (int(goal[0]/resolution), int(goal[1]/resolution))
    
    heap = []
    heappush(heap, (0, 0, *startIdx, startAngle, None))
    visited = {}
    
    directions = [(1,0,0), (-1,0,pi), (0,1,pi/2), (0,-1,3*pi/2),
                 (1,1,pi/4), (1,-1,7*pi/4), (-1,1,3*pi/4), (-1,-1,5*pi/4)]
    
    with tqdm(total=gridDim*gridDim*8, desc="Finding path") as pbar:
        while heap:
            _, cost, x, y, angle, parent = heappop(heap)
            pbar.update(1)
            
            if (x,y) == goalIdx:
                path = []
                while parent:
                    path.append((x, y, angle))
                    x, y, angle, parent = parent
                return path[::-1]
            
            if (x,y,angle) in visited:
                continue
            visited[(x,y,angle)] = True
            
            for dx, dy, moveAngle in directions:
                nx, ny = x+dx, y+dy
                if 0<=nx<gridDim and 0<=ny<gridDim and not obstacleMap[nx,ny]:
                    worldX = nx * resolution
                    worldY = ny * resolution
                    
                    if not is_robot_in_bounds(worldX, worldY, moveAngle):
                        continue
                        
                    moveDist = resolution * sqrt(dx**2 + dy**2)
                    moveTime = moveDist / calculate_mecanum_speed(degrees(moveAngle))
                    rotTime = abs((moveAngle - angle + pi) % (2*pi) - pi) / rotationSpeed
                    heappush(heap, (cost + moveTime + rotTime, 
                                  cost + moveTime + rotTime,
                                  nx, ny, moveAngle, (x, y, angle, parent)))

def execute_path(robot, path):
    if len(path) <= 1:
        return
    
    prevX, prevY, prevAngle = path[0]
    with tqdm(total=len(path)-1, desc="Executing") as pbar:
        for i, (x, y, angle) in enumerate(path[1:]):
            worldX = x * resolution
            worldY = y * resolution
            
            if not is_robot_in_bounds(worldX, worldY, angle):
                print(f"Emergency stop at step {i}: Boundary violation!")
                break
                
            dx = (x - prevX) * resolution
            dy = (y - prevY) * resolution
            distance = sqrt(dx**2 + dy**2)
            moveAngle = atan2(dy, dx)
            
            if abs(prevAngle - moveAngle) > 0.1:
                rotTime = abs(prevAngle - moveAngle) / rotationSpeed
                if moveAngle > prevAngle:
                    robot.movement('FR', -MAX_RPS*60, rotTime)
                    robot.movement('FL', MAX_RPS*60, rotTime)
                else:
                    robot.movement('FR', MAX_RPS*60, rotTime)
                    robot.movement('FL', -MAX_RPS*60, rotTime)
                time.sleep(rotTime)
            
            moveTime = distance / calculate_mecanum_speed(degrees(moveAngle))
            vx, vy = np.cos(moveAngle), np.sin(moveAngle)
            robot.movement('FL', (vx-vy)*MAX_RPS*60, moveTime)
            robot.movement('FR', (vx+vy)*MAX_RPS*60, moveTime)
            robot.movement('BL', (vx+vy)*MAX_RPS*60, moveTime)
            robot.movement('BR', (vx-vy)*MAX_RPS*60, moveTime)
            time.sleep(moveTime)
            
            prevX, prevY, prevAngle = x, y, angle
            pbar.update(1)

def visualize_path(path, obstacles):
    plt.figure(figsize=(10, 10))
    
    # Field boundary
    plt.plot([0, FIELD_SIZE, FIELD_SIZE, 0, 0], 
             [0, 0, FIELD_SIZE, FIELD_SIZE, 0], 'k-', linewidth=2)
    
    # Obstacles
    for x, y, side in obstacles:
        plt.gca().add_patch(plt.Rectangle((x-side/2, y-side/2), side, side,
                                        color='red', alpha=0.5))
    
    if path:
        # Center path
        xCoords = [p[0]*resolution for p in path]
        yCoords = [p[1]*resolution for p in path]
        plt.plot(xCoords, yCoords, 'b-', linewidth=1.5)
        
        # Robot outlines
        halfLength = robotDims[0]/2
        halfWidth = robotDims[1]/2
        
        for i in range(0, len(path), max(1, len(path)//20)):
            x, y, angle = path[i]
            worldX = x * resolution
            worldY = y * resolution
            
            corners = [
                (worldX + halfLength*np.cos(angle) - halfWidth*np.sin(angle),
                 worldY + halfLength*np.sin(angle) + halfWidth*np.cos(angle)),
                (worldX + halfLength*np.cos(angle) + halfWidth*np.sin(angle),
                 worldY + halfLength*np.sin(angle) - halfWidth*np.cos(angle)),
                (worldX - halfLength*np.cos(angle) + halfWidth*np.sin(angle),
                 worldY - halfLength*np.sin(angle) - halfWidth*np.cos(angle)),
                (worldX - halfLength*np.cos(angle) - halfWidth*np.sin(angle),
                 worldY - halfLength*np.sin(angle) + halfWidth*np.cos(angle))
            ]
            corners.append(corners[0])
            plt.plot([c[0] for c in corners], [c[1] for c in corners], 
                    'g-', linewidth=0.8, alpha=0.7)
            
            # Facing indicator
            frontX = worldX + halfLength * np.cos(angle)
            frontY = worldY + halfLength * np.sin(angle)
            plt.plot([worldX, frontX], [worldY, frontY], 'r-', linewidth=1)
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Path Visualization with Robot Outline")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    obstacles = [(1.85, 1.85, 1.0)]
    start = (0.25, 0.25)
    goal = (3.3, 2.6)
    
    obstacleMap = create_obstacle_map(obstacles)
    path = find_optimal_path(start, goal, obstacleMap)
    
    if path:
        visualize_path(path, obstacles)
        robot = Robot()
        execute_path(robot, path)
    else:
        print("No valid path found")