"""Warehouse maps used by the Search lab (Task 3 test cases)."""

# The original warehouse supplied in the laboratory (Task 3, Test 1).
# This is the map printed in the Search lab PDF.
ORIGINAL = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""

# An additionally open warehouse used to make the BFS/A* comparison
# illustrative.  The original lab map is essentially a single corridor, so
# BFS and A* expand the same set of cells.  Here the warehouse has two open
# rooms joined by a door at the bottom.  The start is in the LEFT room and the
# goal in the RIGHT room.  BFS floods the left room (away from the goal) before
# reaching the door; A*'s Manhattan heuristic pulls the search towards the door
# and the goal, so it expands strictly fewer states for the same optimal path.
OPEN_WAREHOUSE = """\
######################
#..........#.........#
#..........#.........#
#..........#.........#
#....S.....#....G....#
#..........#.........#
#..........#.........#
#..........#.........#
#..........#.........#
#..........#.........#
#..........+.........#
######################
""".replace("+", ".")  # the '+' marks the single door in the dividing wall


# Test 2: trivial case — goal immediately adjacent to the start.
TRIVIAL = """\
#####
#SG##
#####
"""

# Test 3: no solution — the goal is walled off completely.
NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######
"""

# Test 4: alternative paths — a loop giving two routes of equal shortest length.
ALTERNATIVE = """\
#######
#S...G#
#.###.#
#.....#
#######
"""

# A map on which greedy best-first search is lured into a longer route while
# A* still finds the shortest path.  Used to illustrate that balancing g and h
# (A*) keeps optimality, whereas chasing h alone (greedy) does not.
GREEDY_TRAP = """\
#########
#S..#...#
#.......#
#.#.....#
#....#..#
#....#.G#
#########
"""
