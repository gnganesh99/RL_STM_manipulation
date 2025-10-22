import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ============================
# Geometry helpers
# ============================
@dataclass
class Rect:
    """A simple axis-aligned rectangular obstacle."""

    x: float
    y: float
    w: float
    h: float
    padding:Optional[float] = 0.0 # Optional padding for collision checks



    def contains(self, px: float, py: float) -> bool:
        """Return True if point (px,py) lies inside the rectangle."""
        return self.x - self.padding <= px <= self.x + self.w + self.padding and self.y - self.padding <= py <= self.y + self.h + self.padding

    def edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return a list of 4 edges (as point pairs) defining the rectangle."""
        x0, y0, x1, y1 = self.x - self.padding, self.y - self.padding, self.x + (self.w + self.padding), self.y + (self.h + self.padding)
        return [((x0, y0), (x1, y0)),
                ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)),
                ((x0, y1), (x0, y0))]

# Circular obstacles
@dataclass
class Circle:

    x: float
    y: float
    r: float
    padding: Optional[float] = 0.0  # Optional padding for collision checks


    def contains(self, px: float, py: float) -> bool:
        """Return True if point (px,py) lies inside the circle."""
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= (self.r + self.padding) ** 2



def orientation(a, b, c) -> int:
    """Helper for segment intersection: orientation of triplet (a,b,c)."""

    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def on_segment(a, b, c) -> bool:
    """Return True if point b lies on segment a–c."""
    return (min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12 and
            min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12)


def segments_intersect(p1, q1, p2, q2) -> bool:
    """Check if line segments p1–q1 and p2–q2 intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False


def segment_hits_rect(p, q, rect: Rect) -> bool:
    """Return True if segment p–q intersects rectangle or lies inside it."""
    # Check if segment endpoints are inside the rectangle
    if rect.contains(*p) or rect.contains(*q):
        return True

    # Check for intersections with rectangle edges
    for e in rect.edges():
        if segments_intersect(p, q, e[0], e[1]):
            return True

    return False





def segment_hits_circle(p, q, circle:Circle, interpolate = 100) -> bool:

    if p[0] != q[0]:
        slope = (p[1] - q[1]) / (p[0] - q[0])

    # Check if the segment intersects the circle
    x = np.linspace(p[0], q[0], num=interpolate) if p[0] != q[0] else np.ones(interpolate) * p[0]
    y = slope * (x - p[0]) + p[1] if p[0] != q[0] else np.linspace(p[1], q[1], num=interpolate)

    for px, py in zip(x, y):
        if circle.contains(px, py):
            return True

    return False





# ============================
# RRT* core
# ============================
@dataclass
class Node:
    """A node in the RRT* tree."""
    x: float
    y: float
    parent: Optional[int]  # index of parent node
    cost: float            # cost-to-come (path length from start)


class RRTStar:
    """
    RRT* path planner in 2D with rectangular obstacles.

    Algorithm overview:
    - Grow a tree from the start state by random sampling.
    - Each new node is connected to the best parent among neighbors
      (minimizing path cost).
    - Rewiring step improves nearby nodes if they become cheaper.
    - Terminates when a node close enough to the goal is found.
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        bounds: Tuple[float, float, float, float],
        obstacles: List[Rect],
        step_size: float = 12.0,
        goal_radius: float = 18.0,
        goal_sample_rate: float = 0.07,
        max_iters: int = 10_000,
        neighbor_radius: Optional[float] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize RRT*.

        Args:
            start: Starting point (x,y).
            goal: Goal point (x,y).
            bounds: (xmin, ymin, xmax, ymax) defining the planning region.
            obstacles: List of rectangular obstacles.
            step_size: Maximum step size per tree expansion.
            goal_radius: Distance threshold for accepting goal.
            goal_sample_rate: Probability of sampling goal directly (bias).
            max_iters: Maximum iterations of tree growth.
            neighbor_radius: Fixed neighbor radius; if None, use adaptive rule.
            seed: Random seed for reproducibility.
        """
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.start = Node(*start, parent=None, cost=0.0)
        self.goal_point = goal
        self.obstacles = obstacles
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.goal_sample_rate = goal_sample_rate
        self.max_iters = max_iters
        self.neighbor_radius = neighbor_radius
        self.nodes: List[Node] = [self.start]
        self.best_goal_idx: Optional[int] = None
        self.iter = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Area of free space (used in adaptive radius)
        self.free_area = (self.xmax - self.xmin) * (self.ymax - self.ymin)

    # ---- helpers ----
    def sample_free(self) -> Tuple[float, float]:
        """Randomly sample a free point; with small probability, return goal."""

        if self.iter == 0: # first sample is biased towards goal
            return self.goal_point
        
        if random.random() < self.goal_sample_rate:
            return self.goal_point
        
        return (random.uniform(self.xmin, self.xmax),
                random.uniform(self.ymin, self.ymax))

    def nearest(self, point: Tuple[float, float]) -> int:
        """Return index of nearest node to given point."""
        px, py = point
        best_i, best_d2 = 0, float("inf")
        for i, n in enumerate(self.nodes):
            d2 = (n.x - px) ** 2 + (n.y - py) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i

    def steer(self, from_node: Node, to_point: Tuple[float, float]) -> Tuple[float, float]:
        """Move from from_node toward to_point by at most step_size."""
        dx, dy = to_point[0] - from_node.x, to_point[1] - from_node.y
        dist = math.hypot(dx, dy)
        if dist <= self.step_size:
            return to_point
        ux, uy = dx / dist, dy / dist
        return (from_node.x + self.step_size * ux, from_node.y + self.step_size * uy)

    def collision_free(self, p: Tuple[float, float], q: Tuple[float, float]) -> bool:
        """Return True if straight line p–q is within bounds and obstacle-free."""
        # Check if the point is within bounds
        for coord in [p, q]:
            if not (self.xmin <= coord[0] <= self.xmax and self.ymin <= coord[1] <= self.ymax):
                return False

        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            if isinstance(obstacle, Rect):
                if segment_hits_rect(p, q, obstacle):
                    return False
                
            if isinstance(obstacle, Circle):
                if segment_hits_circle(p, q, obstacle, interpolate = int(self.step_size)):
                    return False
        return True

    

    def distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reached_goal(self, point: Tuple[float, float]) -> bool:
        """Return True if point lies within goal_radius of goal."""
        return self.distance(point, self.goal_point) <= self.goal_radius

    def neighbors(self, point: Tuple[float, float], n: int) -> List[int]:
        """
        Return indices of neighbors within radius r.

        If neighbor_radius is None, compute radius adaptively:
            r_n ∝ sqrt((log n) / n).
        """
        if self.neighbor_radius is not None:
            r = self.neighbor_radius
        else:
            gamma = 50.0 * math.sqrt(self.free_area)  # tunable constant
            r = min(gamma * math.sqrt(max(math.log(max(n, 2)) / n, 1e-9)), 100.0)

        px, py = point
        idxs = []
        r2 = r * r
        for i, nd in enumerate(self.nodes):
            if (nd.x - px) ** 2 + (nd.y - py) ** 2 <= r2:
                idxs.append(i)
        return idxs

    # ---- main ----
    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """
        Run the RRT* algorithm.

        Returns:
            A list of waypoints [(x,y), ...] from start to goal if found,
            otherwise None.
        """
        for self.iter in range(self.max_iters):
            x_rand = self.sample_free()
            idx_nearest = self.nearest(x_rand)
            x_new = self.steer(self.nodes[idx_nearest], x_rand)

            if not self.collision_free((self.nodes[idx_nearest].x, self.nodes[idx_nearest].y), x_new):
                continue

            # Best parent among neighbors
            neighs = self.neighbors(x_new, len(self.nodes))
            best_parent = idx_nearest
            best_cost = self.nodes[idx_nearest].cost + self.distance((self.nodes[idx_nearest].x, self.nodes[idx_nearest].y), x_new)

            for j in neighs:
                nd = self.nodes[j]
                new_cost = nd.cost + self.distance((nd.x, nd.y), x_new)
                if new_cost < best_cost and self.collision_free((nd.x, nd.y), x_new):
                    best_parent = j
                    best_cost = new_cost

            # Add new node
            new_idx = len(self.nodes)
            self.nodes.append(Node(x_new[0], x_new[1], parent=best_parent, cost=best_cost))

            # Rewire neighbors
            for j in neighs:
                if j == best_parent or j == new_idx:
                    continue
                nd = self.nodes[j]
                edge_cost = self.distance((self.nodes[new_idx].x, self.nodes[new_idx].y), (nd.x, nd.y))
                if self.nodes[new_idx].cost + edge_cost < nd.cost and self.collision_free((self.nodes[new_idx].x, self.nodes[new_idx].y), (nd.x, nd.y)):
                    self.nodes[j].parent = new_idx
                    self.propagate_cost_update(j)

            # Check goal
            if self.reached_goal(x_new) and self.collision_free(x_new, self.goal_point):
                self.best_goal_idx = new_idx

        if self.best_goal_idx is not None:
            return self.extract_path(self.best_goal_idx)
        return None

    def propagate_cost_update(self, idx: int):
        """Update cost-to-come of descendants after a rewiring event."""
        from collections import deque
        q = deque([idx])
        while q:
            i = q.popleft()
            node = self.nodes[i]
            if node.parent is not None:
                parent = self.nodes[node.parent]
                node.cost = parent.cost + self.distance((parent.x, parent.y), (node.x, node.y))
            for k, child in enumerate(self.nodes):
                if child.parent == i:
                    q.append(k)

    def extract_path(self, idx: int) -> List[Tuple[float, float]]:
        """Reconstruct path from given node index back to the start."""
        path = []
        cur = idx
        while cur is not None:
            n = self.nodes[cur]
            path.append((n.x, n.y))
            cur = n.parent
        path.reverse()
        if self.collision_free(path[-1], self.goal_point):
            path.append(self.goal_point)
        return path


def run_rrt_star(rect_array, start, goal, pixels = 256, max_iters = 5000, padding = 3, rrt_above_d = 0):
    """Run a demo of RRT* in a 2D world with obstacles and plot the result."""
    bounds = (0, 0, pixels, pixels)
    start = (int(start[0]), int(start[1]))
    goal  = (int(goal[0]), int(goal[1]))
    padding = int(padding)
    obstacles = []

    for rect in rect_array:
        obstacles.append(Rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), padding = padding))



    # if the straight path is collision free, return the direct path
    _is_collision_free = True
    
    for obstacle in obstacles:
        if segment_hits_rect(start, goal, obstacle):
            _is_collision_free = False
            break
        
    direct_distance = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)

    if _is_collision_free or direct_distance < rrt_above_d:
        path = np.array([start, goal])
        p_label = "Direct path"
        return path, p_label

    planner = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=5.0,
        goal_radius=8.0,
        goal_sample_rate=0.2,
        max_iters=max_iters,
        neighbor_radius=None,
        seed=None,
    )

    print("RRT* planning...")
    path = planner.plan()
    print("Found path?", path is not None)

    if path is not None: 
        path = np.array(list(dict.fromkeys(map(tuple,path))), dtype = float) #removes duplicates while preserving order. (np.unique does not preserve order)
        p_label = "RRT* path"
    else:
        path = np.array([start])
        p_label = "Path not found"


    # # Plot
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.set_xlim(bounds[0], bounds[2])
    # ax.set_ylim(bounds[1], bounds[3])
    # ax.set_aspect('equal')
    # for ob in obstacles:
    #     r = plt.Rectangle((ob.x, ob.y), ob.w, ob.h, color='gray')
    #     ax.add_patch(r)
    
    # if path is not None:
    #     px, py = zip(*path)
    #     ax.plot(px, py, '-r', linewidth=2, label='RRT* path')
    #     ax.plot(start[0], start[1], 'og', markersize=8, label='Start')
    #     ax.plot(goal[0], goal[1], 'xb', markersize=8, label='Goal')
    #     ax.legend()
        
        #rrt_img_path = 'rrt_path.jpg'
        #plt.savefig(rrt_img_path, bbox_inches = 'tight', pad_inches = 0)


    return path, p_label
