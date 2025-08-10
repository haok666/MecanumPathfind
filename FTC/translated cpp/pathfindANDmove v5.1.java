import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class MecanumThetaStar {

    // ------------------- USER EDITABLE PARAMETERS -------------------
    static final double fieldSize = 3.6666667;  // meters (square side)
    static final double marginError = 0.005;    // meters (5 mm)
    static final double cellSize = 0.02;        // meters per grid cell

    static final double robotWidth = 0.40;      // m
    static final double robotLength = 0.40;     // m
    static final double robotHalfDiagonal = Math.hypot(robotWidth, robotLength) / 2.0;
    static final double clearanceMargin = 0.01; // 1 cm clearance

    static final double wheelDiameter = 0.08;   // m
    static final double wheelRadius = wheelDiameter / 2.0;
    static final double maxRps = 2.5;           // max revolutions per second

    // Directional speed factors
    static final double forwardSpeedFactor = 1.00;
    static final double strafeSpeedFactor = 0.95;

    // Output paths
    static final String outputImagePath = "mecanum_path_visualization.png";
    static final String outputCsvPath = "wheel_rotations.csv";

    // Control sampling
    static final double dt = 0.1;

    // Optional image map (set to null to use obstacles list)
    static final String imagePath = null;

    // Obstacles: (x_min, y_min, width, height) in metres
    static final List<double[]> obstacles = Arrays.asList(
        new double[]{(fieldSize - 1.0) / 2.0, (fieldSize - 1.0) / 2.0, 1.0, 1.0},
        new double[]{0.5, 0.3, 0.3, 0.7},
        new double[]{2.2, 2.6, 0.5, 0.2}
    );

    // Start and goal positions
    static final double[] startPose = {
        robotHalfDiagonal + clearanceMargin + 0.001,
        robotHalfDiagonal + clearanceMargin + 0.001
    };
    static final double[] goalPose = {
        fieldSize - (robotHalfDiagonal + clearanceMargin + 0.001),
        fieldSize - (robotHalfDiagonal + clearanceMargin + 0.001)
    };
    static final double startOrientation = 0.0;  // radians

    // Inflation retry parameters
    static double inflateFactor = 1.0;
    static final double minInflateFactor = 0.5;
    static final int inflateSteps = 6;

    // Derived constants
    static final int gridSizeX = (int) Math.ceil(fieldSize / cellSize);
    static final int gridSizeY = gridSizeX;
    static final double maxLinearSpeed = 2.0 * Math.PI * wheelRadius * maxRps;

    // Geometry helpers
    static final double halfLength = robotLength / 2.0;
    static final double halfWidth = robotWidth / 2.0;

    // ------------------------- HELPER FUNCTIONS -------------------------
    static int[] worldToGrid(double x, double y) {
        int gx = (int) Math.round(x / cellSize);
        int gy = (int) Math.round(y / cellSize);
        gx = Math.max(0, Math.min(gridSizeX - 1, gx));
        gy = Math.max(0, Math.min(gridSizeY - 1, gy));
        return new int[]{gx, gy};
    }

    static double[] gridToWorld(int gx, int gy) {
        double x = gx * cellSize;
        double y = gy * cellSize;
        return new double[]{x, y};
    }

    // ----------------------- COLLISION DETECTION -----------------------
    static class Point {
        double x, y;
        Point(double x, double y) { this.x = x; this.y = y; }
    }

    static List<Point> rectPolygon(double x, double y, double w, double h) {
        return Arrays.asList(
            new Point(x, y),
            new Point(x + w, y),
            new Point(x + w, y + h),
            new Point(x, y + h)
        );
    }

    static List<Point> orientedRect(double cx, double cy, double w, double h, double theta) {
        double hw = w / 2.0;
        double hh = h / 2.0;
        Point[] corners = {
            new Point(-hw, -hh), new Point(hw, -hh),
            new Point(hw, hh), new Point(-hw, hh)
        };
        
        double cosT = Math.cos(theta);
        double sinT = Math.sin(theta);
        List<Point> out = new ArrayList<>();
        for (Point p : corners) {
            double x = cx + p.x * cosT - p.y * sinT;
            double y = cy + p.x * sinT + p.y * cosT;
            out.add(new Point(x, y));
        }
        return out;
    }

    static double dot(Point a, Point b) {
        return a.x * b.x + a.y * b.y;
    }

    static Point norm(double dx, double dy) {
        double L = Math.hypot(dx, dy);
        if (L == 0) return new Point(0, 0);
        return new Point(dx / L, dy / L);
    }

    static double[] projection(List<Point> poly, Point axis) {
        double minDot = Double.MAX_VALUE;
        double maxDot = -Double.MAX_VALUE;
        for (Point p : poly) {
            double d = dot(p, axis);
            minDot = Math.min(minDot, d);
            maxDot = Math.max(maxDot, d);
        }
        return new double[]{minDot, maxDot};
    }

    static boolean polygonIntersect(List<Point> polyA, List<Point> polyB) {
        // Check axes from polyA
        for (int i = 0; i < polyA.size(); i++) {
            Point p1 = polyA.get(i);
            Point p2 = polyA.get((i + 1) % polyA.size());
            Point edge = new Point(p2.x - p1.x, p2.y - p1.y);
            Point axis = norm(-edge.y, edge.x);
            double[] projA = projection(polyA, axis);
            double[] projB = projection(polyB, axis);
            if (projA[1] < projB[0] || projB[1] < projA[0]) {
                return false;
            }
        }
        // Check axes from polyB
        for (int i = 0; i < polyB.size(); i++) {
            Point p1 = polyB.get(i);
            Point p2 = polyB.get((i + 1) % polyB.size());
            Point edge = new Point(p2.x - p1.x, p2.y - p1.y);
            Point axis = norm(-edge.y, edge.x);
            double[] projA = projection(polyA, axis);
            double[] projB = projection(polyB, axis);
            if (projA[1] < projB[0] || projB[1] < projA[0]) {
                return false;
            }
        }
        return true;
    }

    static boolean robotCollision(double cx, double cy, double theta, List<double[]> obstacleRects) {
        List<Point> polyRobot = orientedRect(cx, cy, robotWidth, robotLength, theta);
        
        // Boundary collision
        if (cx - robotHalfDiagonal - clearanceMargin < 0 || 
            cy - robotHalfDiagonal - clearanceMargin < 0 || 
            cx + robotHalfDiagonal + clearanceMargin > fieldSize || 
            cy + robotHalfDiagonal + clearanceMargin > fieldSize) {
            return true;
        }
        
        for (double[] rect : obstacleRects) {
            List<Point> polyObs = rectPolygon(rect[0], rect[1], rect[2], rect[3]);
            if (polygonIntersect(polyRobot, polyObs)) {
                return true;
            }
        }
        return false;
    }

    // ---------------------- OCCUPANCY GRID BUILDER ----------------------
    static boolean[][] buildObstacleMap() {
        boolean[][] grid = new boolean[gridSizeX][gridSizeY];
        double minAllowed = clearanceMargin + robotHalfDiagonal;
        int minCell = (int) Math.floor(minAllowed / cellSize);
        int maxCell = gridSizeX - minCell - 1;
        
        // Boundary obstacles
        for (int i = 0; i < gridSizeX; i++) {
            for (int j = 0; j < gridSizeY; j++) {
                if (i < minCell || j < minCell || i > maxCell || j > maxCell) {
                    grid[i][j] = true;
                }
            }
        }

        if (imagePath != null) {
            try {
                BufferedImage img = javax.imageio.ImageIO.read(new File(imagePath));
                for (int i = 0; i < gridSizeX; i++) {
                    for (int j = 0; j < gridSizeY; j++) {
                        int pixel = img.getRGB(i, j);
                        int r = (pixel >> 16) & 0xff;
                        int g = (pixel >> 8) & 0xff;
                        int b = pixel & 0xff;
                        if ((r + g + b) / 3 < 128) {
                            grid[i][j] = true;
                        }
                    }
                }
            } catch (IOException e) {
                System.err.println("Error loading image: " + e.getMessage());
            }
        } else {
            double inflate = (robotHalfDiagonal + clearanceMargin) * inflateFactor;
            for (double[] obs : obstacles) {
                double x0 = Math.max(0.0, obs[0] - inflate);
                double y0 = Math.max(0.0, obs[1] - inflate);
                double x1 = Math.min(fieldSize, obs[0] + obs[2] + inflate);
                double y1 = Math.min(fieldSize, obs[1] + obs[3] + inflate);
                int[] g0 = worldToGrid(x0, y0);
                int[] g1 = worldToGrid(x1, y1);
                for (int i = g0[0]; i <= g1[0]; i++) {
                    for (int j = g0[1]; j <= g1[1]; j++) {
                        if (i >= 0 && i < gridSizeX && j >= 0 && j < gridSizeY) {
                            grid[i][j] = true;
                        }
                    }
                }
            }
        }
        return grid;
    }

    // ------------------------ LINE OF SIGHT CHECK -----------------------
    static boolean lineOfSightGrid(Point p1, Point p2, boolean[][] gridObstacles) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double dist = Math.hypot(dx, dy);
        int steps = Math.max(2, (int) Math.ceil(dist / (cellSize * 0.5)));
        
        for (int s = 0; s <= steps; s++) {
            double t = s / (double) steps;
            double x = p1.x + dx * t;
            double y = p1.y + dy * t;
            int[] gridPos = worldToGrid(x, y);
            if (gridObstacles[gridPos[0]][gridPos[1]]) {
                return false;
            }
        }
        return true;
    }

    // ----------------------- GRID CONNECTIVITY CHECK --------------------
    static class GridNode {
        int x, y;
        GridNode(int x, int y) { this.x = x; this.y = y; }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            GridNode gridNode = (GridNode) o;
            return x == gridNode.x && y == gridNode.y;
        }
        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }
    }

    static int[] gridReachable(int[] startGrid, int[] goalGrid, boolean[][] gridObstacles) {
        int sx = startGrid[0], sy = startGrid[1];
        int gx = goalGrid[0], gy = goalGrid[1];
        if (gridObstacles[sx][sy] || gridObstacles[gx][gy]) {
            return new int[]{0, 0};  // Not reachable
        }

        Queue<GridNode> queue = new LinkedList<>();
        queue.add(new GridNode(sx, sy));
        boolean[][] visited = new boolean[gridSizeX][gridSizeY];
        visited[sx][sy] = true;
        int count = 1;

        int[][] neighbors = {
            {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}
        };

        while (!queue.isEmpty()) {
            GridNode current = queue.poll();
            if (current.x == gx && current.y == gy) {
                return new int[]{1, count};  // Reachable with count
            }
            for (int[] offset : neighbors) {
                int nx = current.x + offset[0];
                int ny = current.y + offset[1];
                if (nx >= 0 && ny >= 0 && nx < gridSizeX && ny < gridSizeY && 
                    !visited[nx][ny] && !gridObstacles[nx][ny]) {
                    visited[nx][ny] = true;
                    queue.add(new GridNode(nx, ny));
                    count++;
                }
            }
        }
        return new int[]{0, count};  // Not reachable
    }

    // -------------------------- THETA* ALGORITHM -------------------------
    static class Node implements Comparable<Node> {
        int x, y;
        double fScore;
        Node(int x, int y, double fScore) {
            this.x = x; this.y = y; this.fScore = fScore;
        }
        @Override
        public int compareTo(Node other) {
            return Double.compare(this.fScore, other.fScore);
        }
    }

    static List<Point> thetaStar(Point start, Point goal, boolean[][] gridObstacles) {
        int[] startGrid = worldToGrid(start.x, start.y);
        int[] goalGrid = worldToGrid(goal.x, goal.y);
        GridNode startNode = new GridNode(startGrid[0], startGrid[1]);
        GridNode goalNode = new GridNode(goalGrid[0], goalGrid[1]);

        // Heuristic function
        java.util.function.Function<GridNode, Double> heuristic = (node) -> {
            double[] world = gridToWorld(node.x, node.y);
            double dx = goal.x - world[0];
            double dy = goal.y - world[1];
            return Math.hypot(dx, dy) / maxLinearSpeed;
        };

        PriorityQueue<Node> openSet = new PriorityQueue<>();
        Map<GridNode, Double> gScore = new HashMap<>();
        Map<GridNode, GridNode> parent = new HashMap<>();
        Map<GridNode, Double> fScore = new HashMap<>();

        gScore.put(startNode, 0.0);
        parent.put(startNode, null);
        double startFScore = heuristic.apply(startNode);
        fScore.put(startNode, startFScore);
        openSet.add(new Node(startNode.x, startNode.y, startFScore));

        int[][] neighborOffsets = {
            {-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}
        };

        int maxIter = gridSizeX * gridSizeY * 8;
        int iters = 0;

        while (!openSet.isEmpty() && iters < maxIter) {
            Node current = openSet.poll();
            GridNode currentGrid = new GridNode(current.x, current.y);
            iters++;

            if (currentGrid.equals(goalNode)) break;

            for (int[] offset : neighborOffsets) {
                int nx = current.x + offset[0];
                int ny = current.y + offset[1];
                if (nx < 0 || ny < 0 || nx >= gridSizeX || ny >= gridSizeY) continue;
                if (gridObstacles[nx][ny]) continue;
                
                GridNode neighbor = new GridNode(nx, ny);
                Point neighborWorld = new Point(gridToWorld(nx, ny)[0], gridToWorld(nx, ny)[1]);

                // Case 1: Check line-of-sight to parent
                if (parent.get(currentGrid) != null) {
                    GridNode parentNode = parent.get(currentGrid);
                    Point parentWorld = new Point(gridToWorld(parentNode.x, parentNode.y)[0], 
                                              gridToWorld(parentNode.x, parentNode.y)[1]);
                    
                    if (lineOfSightGrid(parentWorld, neighborWorld, gridObstacles)) {
                        double tentativeG = gScore.get(parentNode) + 
                            Math.hypot(neighborWorld.x - parentWorld.x, 
                                       neighborWorld.y - parentWorld.y) / maxLinearSpeed;
                        
                        if (!gScore.containsKey(neighbor) || tentativeG < gScore.get(neighbor)) {
                            gScore.put(neighbor, tentativeG);
                            parent.put(neighbor, parentNode);
                            double newFScore = tentativeG + heuristic.apply(neighbor);
                            fScore.put(neighbor, newFScore);
                            openSet.add(new Node(nx, ny, newFScore));
                        }
                        continue;
                    }
                }

                // Case 2: Standard A* move
                Point currentWorld = new Point(gridToWorld(current.x, current.y)[0], 
                                           gridToWorld(current.x, current.y)[1]);
                double dist = Math.hypot(neighborWorld.x - currentWorld.x, 
                                         neighborWorld.y - currentWorld.y);
                double tentativeG = gScore.get(currentGrid) + dist / maxLinearSpeed;
                
                if (!gScore.containsKey(neighbor) || tentativeG < gScore.get(neighbor)) {
                    if (!lineOfSightGrid(currentWorld, neighborWorld, gridObstacles)) continue;
                    gScore.put(neighbor, tentativeG);
                    parent.put(neighbor, currentGrid);
                    double newFScore = tentativeG + heuristic.apply(neighbor);
                    fScore.put(neighbor, newFScore);
                    openSet.add(new Node(nx, ny, newFScore));
                }
            }
        }

        if (!parent.containsKey(goalNode)) {
            throw new RuntimeException("Path not found");
        }

        // Reconstruct path
        List<Point> path = new ArrayList<>();
        GridNode current = goalNode;
        while (current != null) {
            path.add(new Point(gridToWorld(current.x, current.y)[0], 
                           gridToWorld(current.x, current.y)[1]));
            current = parent.get(current);
        }
        Collections.reverse(path);
        path.set(0, start);
        path.set(path.size() - 1, goal);
        return path;
    }

    // ------------------------- PATH SMOOTHING -------------------------
    static List<Point> smoothPath(List<Point> path, boolean[][] gridObstacles) {
        if (path.size() <= 2) return path;
        List<Point> smoothed = new ArrayList<>();
        smoothed.add(path.get(0));
        int i = 0;
        
        while (i < path.size() - 1) {
            int j = path.size() - 1;
            boolean success = false;
            while (j > i) {
                if (lineOfSightGrid(path.get(i), path.get(j), gridObstacles)) {
                    smoothed.add(path.get(j));
                    i = j;
                    success = true;
                    break;
                }
                j--;
            }
            if (!success) {
                smoothed.add(path.get(i + 1));
                i++;
            }
        }
        return smoothed;
    }

    // ------------------- WHEEL ROTATION GENERATION --------------------
    static class WheelEntry {
        double tStart, tEnd;
        double frRps, flRps, brRps, blRps;
        WheelEntry(double tStart, double tEnd, double fr, double fl, double br, double bl) {
            this.tStart = tStart; this.tEnd = tEnd;
            this.frRps = fr; this.flRps = fl; this.brRps = br; this.blRps = bl;
        }
    }

    static List<WheelEntry> pathToWheelTable(List<Point> path, double startTheta) {
        List<WheelEntry> timeline = new ArrayList<>();
        double tNow = 0.0;
        double currentTheta = startTheta;
        double L = robotLength;
        double W = robotWidth;
        double r = wheelRadius;
        double k = (L + W) / 2.0;
        
        for (int segIdx = 1; segIdx < path.size(); segIdx++) {
            Point p0 = path.get(segIdx - 1);
            Point p1 = path.get(segIdx);
            double dx = p1.x - p0.x;
            double dy = p1.y - p0.y;
            double dist = Math.hypot(dx, dy);
            if (dist < 1e-9) continue;
            
            double angleGlobal = Math.atan2(dy, dx);
            double relAngle = angleGlobal - currentTheta;
            double vx = maxLinearSpeed * Math.cos(relAngle) * forwardSpeedFactor;
            double vy = -maxLinearSpeed * Math.sin(relAngle) * strafeSpeedFactor;
            double nominalSpeed = Math.max(1e-6, Math.hypot(vx, vy));
            double tSeg = dist / nominalSpeed;
            int numSteps = Math.max(1, (int) Math.ceil(tSeg / dt));
            double stepDt = tSeg / numSteps;
            
            for (int s = 0; s < numSteps; s++) {
                double omega = 0.0;
                double frRad = (vx - vy - k * omega) / r;
                double flRad = (vx + vy + k * omega) / r;
                double brRad = (vx + vy - k * omega) / r;
                double blRad = (vx - vy + k * omega) / r;
                
                double frRps = frRad / (2.0 * Math.PI);
                double flRps = flRad / (2.0 * Math.PI);
                double brRps = brRad / (2.0 * Math.PI);
                double blRps = blRad / (2.0 * Math.PI);
                
                double maxMag = Math.max(Math.max(
                    Math.abs(frRps), Math.abs(flRps)),
                    Math.max(Math.abs(brRps), Math.max(Math.abs(blRps), 1e-12))
                );
                
                if (maxMag > maxRps) {
                    double scale = maxRps / maxMag;
                    frRps *= scale;
                    flRps *= scale;
                    brRps *= scale;
                    blRps *= scale;
                }
                
                timeline.add(new WheelEntry(
                    tNow, tNow + stepDt, frRps, flRps, brRps, blRps
                ));
                tNow += stepDt;
            }
        }
        return timeline;
    }

    // ----------------------- VISUALIZATION ----------------------------
    static void drawVisualization(List<Point> path, List<double[]> obstacleRects) {
        int imgSize = 800;
        double scale = imgSize / fieldSize;
        BufferedImage image = new BufferedImage(imgSize, imgSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        
        // White background
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, imgSize, imgSize);
        
        // Field boundary
        g.setColor(Color.BLACK);
        g.drawRect(0, 0, imgSize - 1, imgSize - 1);
        
        // Draw obstacles
        g.setColor(Color.RED);
        for (double[] rect : obstacleRects) {
            int x = (int) (rect[0] * scale);
            int y = (int) (rect[1] * scale);
            int w = (int) (rect[2] * scale);
            int h = (int) (rect[3] * scale);
            g.drawRect(x, y, w, h);
        }
        
        // Draw path
        g.setColor(Color.BLUE);
        for (int i = 1; i < path.size(); i++) {
            Point p0 = path.get(i - 1);
            Point p1 = path.get(i);
            int x0 = (int) (p0.x * scale);
            int y0 = (int) (p0.y * scale);
            int x1 = (int) (p1.x * scale);
            int y1 = (int) (p1.y * scale);
            g.drawLine(x0, y0, x1, y1);
            g.fillOval(x1 - 3, y1 - 3, 6, 6);
        }
        
        // Draw robot positions
        g.setColor(new Color(0, 0, 255, 100));
        for (Point p : path) {
            List<Point> poly = orientedRect(p.x, p.y, robotWidth, robotLength, startOrientation);
            int[] xPoints = new int[poly.size()];
            int[] yPoints = new int[poly.size()];
            for (int i = 0; i < poly.size(); i++) {
                xPoints[i] = (int) (poly.get(i).x * scale);
                yPoints[i] = (int) (poly.get(i).y * scale);
            }
            g.drawPolygon(xPoints, yPoints, poly.size());
            
            // Draw heading arrow
            int cx = (int) (p.x * scale);
            int cy = (int) (p.y * scale);
            int hx = (int) ((p.x + 0.5 * robotLength * Math.cos(startOrientation)) * scale);
            int hy = (int) ((p.y + 0.5 * robotLength * Math.sin(startOrientation)) * scale);
            g.drawLine(cx, cy, hx, hy);
            g.fillOval(hx - 2, hy - 2, 4, 4);
        }
        
        try {
            javax.imageio.ImageIO.write(image, "png", new File(outputImagePath));
            System.out.println("Saved visualization to " + outputImagePath);
        } catch (IOException e) {
            System.err.println("Error saving image: " + e.getMessage());
        }
    }

    static void writeWheelCsv(List<WheelEntry> timeline) {
        try (FileWriter writer = new FileWriter(outputCsvPath)) {
            writer.write("time_start,time_end,FR_rps,FL_rps,BR_rps,BL_rps\n");
            for (WheelEntry entry : timeline) {
                writer.write(String.format("%.3f,%.3f,%.6f,%.6f,%.6f,%.6f%n",
                    entry.tStart, entry.tEnd,
                    entry.frRps, entry.flRps, entry.brRps, entry.blRps
                ));
            }
            System.out.println("Wrote wheel table to " + outputCsvPath);
        } catch (IOException e) {
            System.err.println("Error writing CSV: " + e.getMessage());
        }
    }

    // --------------------------- MAIN PROGRAM ---------------------------
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        boolean found = false;
        List<Double> triedFactors = new ArrayList<>();
        List<Point> finalPath = null;
        boolean[][] finalGrid = null;
        List<double[]> finalObstacles = null;
        
        for (int step = 0; step < inflateSteps; step++) {
            inflateFactor = 1.0 - step * (1.0 - minInflateFactor) / Math.max(1, inflateSteps - 1);
            boolean[][] gridObstacles = buildObstacleMap();
            List<double[]> obstacleRects = new ArrayList<>(obstacles);
            
            // Add boundary obstacles
            obstacleRects.add(new double[]{-1.0, -1.0, fieldSize + 2.0, 1.0});
            obstacleRects.add(new double[]{-1.0, fieldSize, fieldSize + 2.0, 1.0});
            obstacleRects.add(new double[]{-1.0, -1.0, 1.0, fieldSize + 2.0});
            obstacleRects.add(new double[]{fieldSize, -1.0, 1.0, fieldSize + 2.0});
            
            int[] startGrid = worldToGrid(startPose[0], startPose[1]);
            int[] goalGrid = worldToGrid(goalPose[0], goalPose[1]);
            System.out.printf("inflateFactor=%.3f startGrid=(%d,%d) blocked=%s goalGrid=(%d,%d) blocked=%s%n",
                inflateFactor, startGrid[0], startGrid[1], gridObstacles[startGrid[0]][startGrid[1]],
                goalGrid[0], goalGrid[1], gridObstacles[goalGrid[0]][goalGrid[1]]);
            
            int[] reachResult = gridReachable(startGrid, goalGrid, gridObstacles);
            boolean reachable = reachResult[0] == 1;
            System.out.printf("Grid connectivity test: reachable=%s reachableCellCount=%d%n",
                reachable, reachResult[1]);
            
            if (!reachable) {
                triedFactors.add(inflateFactor);
                System.out.println("Occupancy grid disconnected. Reducing inflation and retrying.");
                continue;
            }
            
            if (robotCollision(startPose[0], startPose[1], startOrientation, obstacleRects)) {
                throw new RuntimeException("Start pose collides geometrically");
            }
            
            if (robotCollision(goalPose[0], goalPose[1], startOrientation, obstacleRects)) {
                throw new RuntimeException("Goal pose collides geometrically");
            }
            
            try {
                Point startPoint = new Point(startPose[0], startPose[1]);
                Point goalPoint = new Point(goalPose[0], goalPose[1]);
                finalPath = thetaStar(startPoint, goalPoint, gridObstacles);
                System.out.printf("Path found with inflateFactor=%.3f%n", inflateFactor);
                finalGrid = gridObstacles;
                finalObstacles = obstacleRects;
                found = true;
                break;
            } catch (RuntimeException e) {
                triedFactors.add(inflateFactor);
                System.out.println("theta_star failed: " + e.getMessage());
            }
        }
        
        if (!found) {
            throw new RuntimeException("Path not found after trying inflate factors: " + triedFactors);
        }
        
        List<Point> smoothed = smoothPath(finalPath, finalGrid);
        drawVisualization(smoothed, finalObstacles);
        List<WheelEntry> timeline = pathToWheelTable(smoothed, startOrientation);
        writeWheelCsv(timeline);
        
        double runtime = (System.currentTimeMillis() - startTime) / 1000.0;
        System.out.printf("Total runtime: %.3f seconds%n", runtime);
    }
}