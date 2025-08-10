#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// ------------------- USER EDITABLE PARAMETERS -------------------
const double fieldSize = 3.6666667;  // meters (square side)
const double marginError = 0.005;    // meters (5 mm)
const double cellSize = 0.02;        // meters per grid cell. adjust for speed/precision.

const double robotWidth = 0.40;      // m
const double robotLength = 0.40;     // m
const double robotHalfDiagonal = std::hypot(robotWidth, robotLength) / 2.0;
const double clearanceMargin = 0.01; // 1 cm required clearance from walls/obstacles

const double wheelDiameter = 0.08;   // m
const double wheelRadius = wheelDiameter / 2.0;
const double maxRps = 2.5;           // max revolutions per second per motor

// directional speed factors
const double forwardSpeedFactor = 1.00;
const double strafeSpeedFactor = 0.95;

// visualization and outputs
const std::string outputImagePath = "mecanum_path_visualization.png";
const std::string outputCsvPath = "wheel_rotations.csv";

// control sampling
const double dt = 0.1;

// optional image map (black=obstacle). Set to empty to use obstacles list below.
const std::string imagePath = "";

// Obstacles: axis-aligned rectangles. each entry: (x_min, y_min, width, height) in metres.
const std::vector<std::vector<double>> obstacles = {
    // 1m x 1m square centered in field
    {(fieldSize - 1.0) / 2.0, (fieldSize - 1.0) / 2.0, 1.0, 1.0},
    // example extras
    {0.5, 0.3, 0.3, 0.7},
    {2.2, 2.6, 0.5, 0.2}
};

// start and goal positions (robot center). Corner to corner test.
const std::pair<double, double> startPose = {
    robotHalfDiagonal + clearanceMargin + 0.001, 
    robotHalfDiagonal + clearanceMargin + 0.001
};
const std::pair<double, double> goalPose = {
    fieldSize - (robotHalfDiagonal + clearanceMargin + 0.001),
    fieldSize - (robotHalfDiagonal + clearanceMargin + 0.001)
};
const double startOrientation = 0.0;  // radians

// inflation retry parameters
double inflateFactor = 1.0;
const double minInflateFactor = 0.5;
const int inflateSteps = 6;

// derived
const int gridSizeX = static_cast<int>(std::ceil(fieldSize / cellSize));
const int gridSizeY = gridSizeX;
const double maxLinearSpeed = 2.0 * M_PI * wheelRadius * maxRps;  // m/s

// geometry helpers
const double halfLength = robotLength / 2.0;
const double halfWidth = robotWidth / 2.0;

// ------------------------- HELPER FUNCTIONS -------------------------
std::pair<int, int> world_to_grid(double x, double y) {
    int gx = static_cast<int>(std::round(x / cellSize));
    int gy = static_cast<int>(std::round(y / cellSize));
    gx = std::max(0, std::min(gridSizeX - 1, gx));
    gy = std::max(0, std::min(gridSizeY - 1, gy));
    return {gx, gy};
}

std::pair<double, double> grid_to_world(int gx, int gy) {
    double x = gx * cellSize;
    double y = gy * cellSize;
    return {x, y};
}

// ----------------------- COLLISION DETECTION -----------------------
using Polygon = std::vector<std::pair<double, double>>;

Polygon rect_polygon(double x, double y, double w, double h) {
    return {{x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}};
}

Polygon oriented_rect(double cx, double cy, double w, double h, double theta) {
    double hw = w / 2.0;
    double hh = h / 2.0;
    std::vector<std::pair<double, double>> corners = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);
    Polygon out;
    for (const auto& [dx, dy] : corners) {
        double x = cx + dx * cosT - dy * sinT;
        double y = cy + dx * sinT + dy * cosT;
        out.emplace_back(x, y);
    }
    return out;
}

double dot(double ax, double ay, double bx, double by) {
    return ax * bx + ay * by;
}

std::pair<double, double> norm(double dx, double dy) {
    double L = std::hypot(dx, dy);
    if (L == 0) return {0.0, 0.0};
    return {dx / L, dy / L};
}

std::pair<double, double> projection(const Polygon& poly, double axisX, double axisY) {
    double minDot = std::numeric_limits<double>::max();
    double maxDot = std::numeric_limits<double>::lowest();
    for (const auto& [x, y] : poly) {
        double d = dot(x, y, axisX, axisY);
        minDot = std::min(minDot, d);
        maxDot = std::max(maxDot, d);
    }
    return {minDot, maxDot};
}

bool polygon_intersect(const Polygon& polyA, const Polygon& polyB) {
    auto check_axis = [&](double dx, double dy) {
        auto [minA, maxA] = projection(polyA, dx, dy);
        auto [minB, maxB] = projection(polyB, dx, dy);
        return (maxA >= minB) && (maxB >= minA);
    };

    for (size_t i = 0; i < polyA.size(); ++i) {
        double x1 = polyA[i].first;
        double y1 = polyA[i].second;
        double x2 = polyA[(i+1) % polyA.size()].first;
        double y2 = polyA[(i+1) % polyA.size()].second;
        double edgeX = x2 - x1;
        double edgeY = y2 - y1;
        auto [axisX, axisY] = norm(-edgeY, edgeX);
        if (!check_axis(axisX, axisY)) return false;
    }

    for (size_t i = 0; i < polyB.size(); ++i) {
        double x1 = polyB[i].first;
        double y1 = polyB[i].second;
        double x2 = polyB[(i+1) % polyB.size()].first;
        double y2 = polyB[(i+1) % polyB.size()].second;
        double edgeX = x2 - x1;
        double edgeY = y2 - y1;
        auto [axisX, axisY] = norm(-edgeY, edgeX);
        if (!check_axis(axisX, axisY)) return false;
    }

    return true;
}

bool robot_collision(double cx, double cy, double theta, 
                     const std::vector<std::vector<double>>& obstacleRects) {
    Polygon polyRobot = oriented_rect(cx, cy, robotWidth, robotLength, theta);
    
    // Boundary collision
    if (cx - robotHalfDiagonal - clearanceMargin < 0 || 
        cy - robotHalfDiagonal - clearanceMargin < 0 || 
        cx + robotHalfDiagonal + clearanceMargin > fieldSize || 
        cy + robotHalfDiagonal + clearanceMargin > fieldSize) {
        return true;
    }
    
    for (const auto& rect : obstacleRects) {
        Polygon polyObs = rect_polygon(rect[0], rect[1], rect[2], rect[3]);
        if (polygon_intersect(polyRobot, polyObs)) {
            return true;
        }
    }
    return false;
}

// ---------------------- OCCUPANCY GRID BUILDER ----------------------
cv::Mat build_obstacle_map() {
    cv::Mat grid = cv::Mat::zeros(gridSizeX, gridSizeY, CV_8U);
    double minAllowed = clearanceMargin + robotHalfDiagonal;
    int minCell = static_cast<int>(std::floor(minAllowed / cellSize));
    int maxCell = gridSizeX - minCell - 1;
    
    for (int i = 0; i < gridSizeX; ++i) {
        for (int j = 0; j < gridSizeY; ++j) {
            if (i < minCell || j < minCell || i > maxCell || j > maxCell) {
                grid.at<uint8_t>(i, j) = 1;
            }
        }
    }

    if (!imagePath.empty()) {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error loading image: " << imagePath << std::endl;
            return grid;
        }
        cv::resize(img, img, cv::Size(gridSizeX, gridSizeY), 0, 0, cv::INTER_NEAREST);
        for (int i = 0; i < gridSizeX; ++i) {
            for (int j = 0; j < gridSizeY; ++j) {
                if (img.at<uint8_t>(i, j) < 128) {
                    grid.at<uint8_t>(i, j) = 1;
                }
            }
        }
    } else {
        double inflate = (robotHalfDiagonal + clearanceMargin) * inflateFactor;
        for (const auto& obs : obstacles) {
            double x0 = std::max(0.0, obs[0] - inflate);
            double y0 = std::max(0.0, obs[1] - inflate);
            double x1 = std::min(fieldSize, obs[0] + obs[2] + inflate);
            double y1 = std::min(fieldSize, obs[1] + obs[3] + inflate);
            auto [gx0, gy0] = world_to_grid(x0, y0);
            auto [gx1, gy1] = world_to_grid(x1, y1);
            for (int i = gx0; i <= gx1; ++i) {
                for (int j = gy0; j <= gy1; ++j) {
                    if (i >= 0 && i < gridSizeX && j >= 0 && j < gridSizeY) {
                        grid.at<uint8_t>(i, j) = 1;
                    }
                }
            }
        }
    }
    return grid;
}

// ------------------------ LINE OF SIGHT CHECK -----------------------
bool line_of_sight_grid(const std::pair<double, double>& p1, 
                        const std::pair<double, double>& p2, 
                        const cv::Mat& gridObstacles) {
    double dx = p2.first - p1.first;
    double dy = p2.second - p1.second;
    double dist = std::hypot(dx, dy);
    int steps = std::max(2, static_cast<int>(std::ceil(dist / (cellSize * 0.5))));
    
    for (int s = 0; s <= steps; ++s) {
        double t = static_cast<double>(s) / steps;
        double x = p1.first + dx * t;
        double y = p1.second + dy * t;
        auto [gx, gy] = world_to_grid(x, y);
        if (gridObstacles.at<uint8_t>(gx, gy)) {
            return false;
        }
    }
    return true;
}

// ----------------------- GRID CONNECTIVITY CHECK --------------------
std::pair<bool, int> grid_reachable(const std::pair<int, int>& startNode, 
                                    const std::pair<int, int>& goalNode, 
                                    const cv::Mat& gridObstacles) {
    auto [sx, sy] = startNode;
    auto [gx, gy] = goalNode;
    if (gridObstacles.at<uint8_t>(sx, sy) || gridObstacles.at<uint8_t>(gx, gy)) {
        return {false, 0};
    }

    std::queue<std::pair<int, int>> q;
    q.push(startNode);
    cv::Mat visited = cv::Mat::zeros(gridSizeX, gridSizeY, CV_8U);
    visited.at<uint8_t>(sx, sy) = 1;
    int count = 1;

    std::vector<std::pair<int, int>> neighbors = {
        {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}
    };

    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();
        if (cx == gx && cy == gy) {
            return {true, count};
        }
        for (const auto& [dx, dy] : neighbors) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx >= 0 && ny >= 0 && nx < gridSizeX && ny < gridSizeY && 
                !visited.at<uint8_t>(nx, ny) && !gridObstacles.at<uint8_t>(nx, ny)) {
                visited.at<uint8_t>(nx, ny) = 1;
                q.emplace(nx, ny);
                ++count;
            }
        }
    }
    return {false, count};
}

// -------------------------- THETA* ALGORITHM -------------------------
struct Node {
    int x, y;
    double fScore;
    int entryCount;
    
    bool operator>(const Node& other) const {
        return fScore > other.fScore;
    }
};

std::vector<std::pair<double, double>> theta_star(
    const std::pair<double, double>& start, 
    const std::pair<double, double>& goal, 
    const cv::Mat& gridObstacles) {
    
    auto startGrid = world_to_grid(start.first, start.second);
    auto goalGrid = world_to_grid(goal.first, goal.second);
    std::pair<int, int> startNode = {startGrid.first, startGrid.second};
    std::pair<int, int> goalNode = {goalGrid.first, goalGrid.second};

    auto heuristic = [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        auto [ax, ay] = grid_to_world(a.first, a.second);
        auto [bx, by] = grid_to_world(b.first, b.second);
        return std::hypot(bx - ax, by - ay) / maxLinearSpeed;
    };

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openSet;
    int entryCount = 0;
    std::unordered_map<int, double> gScore;
    std::unordered_map<int, std::pair<int, int>> parent;
    std::unordered_map<int, double> fScore;

    auto hash = [](int x, int y) { return x * gridSizeY + y; };

    gScore[hash(startNode.first, startNode.second)] = 0.0;
    parent[hash(startNode.first, startNode.second)] = {-1, -1};
    fScore[hash(startNode.first, startNode.second)] = heuristic(startNode, goalNode);
    openSet.push({startNode.first, startNode.second, 
                 fScore[hash(startNode.first, startNode.second)], entryCount++});

    std::vector<std::pair<int, int>> neighborOffsets = {
        {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}
    };

    int maxIter = gridSizeX * gridSizeY * 8;
    int iters = 0;

    while (!openSet.empty() && iters < maxIter) {
        Node current = openSet.top();
        openSet.pop();
        std::pair<int, int> currentPos = {current.x, current.y};
        
        if (currentPos == goalNode) break;
        iters++;

        for (const auto& offset : neighborOffsets) {
            int nx = currentPos.first + offset.first;
            int ny = currentPos.second + offset.second;
            if (nx < 0 || ny < 0 || nx >= gridSizeX || ny >= gridSizeY) continue;
            if (gridObstacles.at<uint8_t>(nx, ny)) continue;
            
            std::pair<int, int> neighbor = {nx, ny};
            int neighborHash = hash(nx, ny);
            int currentHash = hash(currentPos.first, currentPos.second);

            if (parent.find(currentHash) != parent.end() && 
                parent[currentHash] != std::pair<int, int>{-1, -1}) {
                auto parentPos = parent[currentHash];
                auto parentWorld = grid_to_world(parentPos.first, parentPos.second);
                auto neighborWorld = grid_to_world(neighbor.first, neighbor.second);
                
                if (line_of_sight_grid(parentWorld, neighborWorld, gridObstacles)) {
                    double tentative_g = gScore[hash(parentPos.first, parentPos.second)] + 
                        std::hypot(neighborWorld.first - parentWorld.first, 
                                   neighborWorld.second - parentWorld.second) / maxLinearSpeed;
                    
                    if (gScore.find(neighborHash) == gScore.end() || 
                        tentative_g < gScore[neighborHash]) {
                        gScore[neighborHash] = tentative_g;
                        parent[neighborHash] = parentPos;
                        fScore[neighborHash] = tentative_g + heuristic(neighbor, goalNode);
                        openSet.push({nx, ny, fScore[neighborHash], entryCount++});
                    }
                    continue;
                }
            }

            auto currentWorld = grid_to_world(currentPos.first, currentPos.second);
            auto neighborWorld = grid_to_world(neighbor.first, neighbor.second);
            double dist = std::hypot(neighborWorld.first - currentWorld.first, 
                                     neighborWorld.second - currentWorld.second);
            double tentative_g = gScore[currentHash] + dist / maxLinearSpeed;
            
            if (gScore.find(neighborHash) == gScore.end() || 
                tentative_g < gScore[neighborHash]) {
                if (!line_of_sight_grid(currentWorld, neighborWorld, gridObstacles)) {
                    continue;
                }
                gScore[neighborHash] = tentative_g;
                parent[neighborHash] = currentPos;
                fScore[neighborHash] = tentative_g + heuristic(neighbor, goalNode);
                openSet.push({nx, ny, fScore[neighborHash], entryCount++});
            }
        }
    }

    if (parent.find(hash(goalNode.first, goalNode.second)) == parent.end()) {
        throw std::runtime_error("Path not found");
    }

    std::vector<std::pair<double, double>> path;
    std::pair<int, int> current = goalNode;
    while (current != std::pair<int, int>{-1, -1}) {
        path.push_back(grid_to_world(current.first, current.second));
        int currentHash = hash(current.first, current.second);
        current = parent.find(currentHash) != parent.end() ? parent[currentHash] : std::pair<int, int>{-1, -1};
    }
    std::reverse(path.begin(), path.end());
    path.front() = start;
    path.back() = goal;
    return path;
}

// ------------------------- PATH SMOOTHING -------------------------
std::vector<std::pair<double, double>> smooth_path(
    const std::vector<std::pair<double, double>>& path, 
    const cv::Mat& gridObstacles) {
    if (path.size() <= 2) return path;
    std::vector<std::pair<double, double>> smoothed;
    smoothed.push_back(path[0]);
    size_t i = 0;
    
    while (i < path.size() - 1) {
        size_t j = path.size() - 1;
        bool success = false;
        while (j > i) {
            if (line_of_sight_grid(path[i], path[j], gridObstacles)) {
                smoothed.push_back(path[j]);
                i = j;
                success = true;
                break;
            }
            j--;
        }
        if (!success) {
            smoothed.push_back(path[i+1]);
            i++;
        }
    }
    return smoothed;
}

// ------------------- WHEEL ROTATION GENERATION --------------------
struct WheelEntry {
    double tStart, tEnd;
    double frRps, flRps, brRps, blRps;
};

std::vector<WheelEntry> path_to_wheel_table(
    const std::vector<std::pair<double, double>>& path, 
    double startOrientation) {
    std::vector<WheelEntry> timeline;
    double tNow = 0.0;
    double currentTheta = startOrientation;
    const double L = robotLength;
    const double W = robotWidth;
    const double r = wheelRadius;
    const double k = (L + W) / 2.0;
    
    for (size_t segIdx = 1; segIdx < path.size(); ++segIdx) {
        const auto& p0 = path[segIdx-1];
        const auto& p1 = path[segIdx];
        double dx = p1.first - p0.first;
        double dy = p1.second - p0.second;
        double dist = std::hypot(dx, dy);
        if (dist < 1e-9) continue;
        
        double angleGlobal = std::atan2(dy, dx);
        double relAngle = angleGlobal - currentTheta;
        double vx = maxLinearSpeed * std::cos(relAngle) * forwardSpeedFactor;
        double vy = -maxLinearSpeed * std::sin(relAngle) * strafeSpeedFactor;
        double nominalSpeed = std::max(1e-6, std::hypot(vx, vy));
        double tSeg = dist / nominalSpeed;
        int numSteps = std::max(1, static_cast<int>(std::ceil(tSeg / dt)));
        double stepDt = tSeg / numSteps;
        
        for (int s = 0; s < numSteps; ++s) {
            double omega = 0.0;
            double frRad = (vx - vy - k * omega) / r;
            double flRad = (vx + vy + k * omega) / r;
            double brRad = (vx + vy - k * omega) / r;
            double blRad = (vx - vy + k * omega) / r;
            
            double frRps = frRad / (2.0 * M_PI);
            double flRps = flRad / (2.0 * M_PI);
            double brRps = brRad / (2.0 * M_PI);
            double blRps = blRad / (2.0 * M_PI);
            
            double maxMag = std::max({
                std::abs(frRps), std::abs(flRps), 
                std::abs(brRps), std::abs(blRps), 1e-12
            });
            
            if (maxMag > maxRps) {
                double scale = maxRps / maxMag;
                frRps *= scale;
                flRps *= scale;
                brRps *= scale;
                blRps *= scale;
            }
            
            timeline.push_back({
                tNow, tNow + stepDt,
                frRps, flRps, brRps, blRps
            });
            tNow += stepDt;
        }
    }
    return timeline;
}

// ----------------------- VISUALIZATION ----------------------------
void draw_visualization(
    const std::vector<std::pair<double, double>>& path,
    const std::vector<std::vector<double>>& obstacleRects) {
    
    const int imgSize = 800;
    const double scale = imgSize / fieldSize;
    cv::Mat img(imgSize, imgSize, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw field boundary
    cv::rectangle(img, 
                 cv::Point(0, 0), 
                 cv::Point(imgSize-1, imgSize-1), 
                 cv::Scalar(0, 0, 0), 1);
    
    // Draw obstacles
    for (const auto& rect : obstacleRects) {
        cv::Point topLeft(rect[0] * scale, rect[1] * scale);
        cv::Point bottomRight((rect[0] + rect[2]) * scale, 
                             (rect[1] + rect[3]) * scale);
        cv::rectangle(img, topLeft, bottomRight, cv::Scalar(0, 0, 255), 1);
    }
    
    // Draw path
    std::vector<cv::Point> cvPath;
    for (const auto& [x, y] : path) {
        cvPath.emplace_back(x * scale, y * scale);
    }
    cv::polylines(img, cvPath, false, cv::Scalar(0, 0, 0), 1);
    
    // Draw robot positions
    for (const auto& [x, y] : path) {
        Polygon poly = oriented_rect(x, y, robotWidth, robotLength, startOrientation);
        std::vector<cv::Point> cvPoly;
        for (const auto& [px, py] : poly) {
            cvPoly.emplace_back(px * scale, py * scale);
        }
        cv::polylines(img, std::vector<std::vector<cv::Point>>{cvPoly}, true, 
                     cv::Scalar(255, 0, 0), 1);
        
        // Draw heading arrow
        cv::Point start(x * scale, y * scale);
        cv::Point end((x + 0.5 * robotLength * std::cos(startOrientation)) * scale,
                     (y + 0.5 * robotLength * std::sin(startOrientation)) * scale);
        cv::arrowedLine(img, start, end, cv::Scalar(0, 255, 0), 1);
    }
    
    cv::imwrite(outputImagePath, img);
    std::cout << "Saved visualization to " << outputImagePath << std::endl;
}

void write_wheel_csv(const std::vector<WheelEntry>& timeline) {
    std::ofstream csvFile(outputCsvPath);
    csvFile << "time_start,time_end,FR_rps,FL_rps,BR_rps,BL_rps\n";
    for (const auto& entry : timeline) {
        csvFile << std::fixed << std::setprecision(6)
                << entry.tStart << "," << entry.tEnd << ","
                << entry.frRps << "," << entry.flRps << ","
                << entry.brRps << "," << entry.blRps << "\n";
    }
    std::cout << "Wrote wheel table to " << outputCsvPath << std::endl;
}

// --------------------------- MAIN PROGRAM ---------------------------
int main() {
    std::clock_t t0 = std::clock();
    bool found = false;
    std::vector<double> tried;
    std::vector<std::pair<double, double>> finalPath;
    cv::Mat finalGrid;
    std::vector<std::vector<double>> finalObstacles;
    
    for (int step = 0; step < inflateSteps; ++step) {
        inflateFactor = 1.0 - step * (1.0 - minInflateFactor) / std::max(1, inflateSteps-1);
        cv::Mat gridObstacles = build_obstacle_map();
        auto obstacleRects = obstacles;
        obstacleRects.push_back({-1.0, -1.0, fieldSize + 2.0, 1.0});
        obstacleRects.push_back({-1.0, fieldSize, fieldSize + 2.0, 1.0});
        obstacleRects.push_back({-1.0, -1.0, 1.0, fieldSize + 2.0});
        obstacleRects.push_back({fieldSize, -1.0, 1.0, fieldSize + 2.0});
        
        auto startGrid = world_to_grid(startPose.first, startPose.second);
        auto goalGrid = world_to_grid(goalPose.first, goalPose.second);
        std::cout << "inflateFactor=" << inflateFactor 
                  << " startGrid=(" << startGrid.first << "," << startGrid.second << ")"
                  << " blocked=" << (gridObstacles.at<uint8_t>(startGrid.first, startGrid.second) 
                  << " goalGrid=(" << goalGrid.first << "," << goalGrid.second << ")"
                  << " blocked=" << (gridObstacles.at<uint8_t>(goalGrid.first, goalGrid.second)) 
                  << std::endl;
        
        auto [reachable, regionSize] = grid_reachable(startGrid, goalGrid, gridObstacles);
        std::cout << "Grid connectivity test: reachable=" << reachable 
                  << " reachableCellCount=" << regionSize << std::endl;
        
        if (!reachable) {
            tried.push_back(inflateFactor);
            std::cout << "Occupancy grid disconnected. Reducing inflation and retrying." << std::endl;
            continue;
        }
        
        if (robot_collision(startPose.first, startPose.second, startOrientation, obstacleRects)) {
            throw std::runtime_error("Start pose collides geometrically");
        }
        
        if (robot_collision(goalPose.first, goalPose.second, startOrientation, obstacleRects)) {
            throw std::runtime_error("Goal pose collides geometrically");
        }
        
        try {
            finalPath = theta_star(startPose, goalPose, gridObstacles);
            std::cout << "Path found with inflateFactor=" << inflateFactor << std::endl;
            finalGrid = gridObstacles.clone();
            finalObstacles = obstacleRects;
            found = true;
            break;
        } catch (const std::exception& e) {
            tried.push_back(inflateFactor);
            std::cout << "theta_star failed at inflateFactor=" << inflateFactor << std::endl;
        }
    }
    
    if (!found) {
        std::string errorMsg = "Path not found after trying inflate factors:";
        for (double factor : tried) errorMsg += " " + std::to_string(factor);
        throw std::runtime_error(errorMsg);
    }
    
    auto smoothed = smooth_path(finalPath, finalGrid);
    draw_visualization(smoothed, finalObstacles);
    auto timeline = path_to_wheel_table(smoothed, startOrientation);
    write_wheel_csv(timeline);
    
    double runtime = (std::clock() - t0) / static_cast<double>(CLOCKS_PER_SEC);
    std::cout << "Total runtime (s): " << runtime << std::endl;
    
    return 0;
}