Your program should have the following input:
1. FOON (A .txt file)
2. A goal object to search (An object name and its state)
3. List of ingredients and utensils available in the kitchen (A .txt file)
FOON, sample goal objects and a kitchen file will be found in the provided starter code. The
starter code and recorded video on this assignment will be found on Canvas module.


Tasks:
1. Implement two search algorithms.
    a. Iterative Deepening Search: Go through the class lectures to learn how this
algorithm works. While you explore the nodes, you may find that there are
multiple ways to prepare an object. Ideally, we need to explore all possible paths
to find the optimal solution. But to make it simple, you can just take the first path
that you find. You need to keep increasing the depth until you find the solution. A
task tree is considered a solution if the leaf nodes are available in the kitchen.


    b. Greedy Best-First Search: To explore a node, instead of choosing a path
randomly from various options, choose a path based on the heuristic function.
You need to implement two different search method using the following two
heuristic function.
i. h(n) = success rate of the motion
ii. h(n) = number of input objects in the function unit
In case of heuristic 1, if you have multiple path with different motions, choose the
path that gives higher success rate of executing the motion successfully. For
example, a robot has higher success rate of pouring sliced onion compared to
slicing a whole onion. The success rate of each motion is provided in the
motion.txt file.
For heuristic 2, if you find that scrambled egg can be prepared with either {egg,
oil, cheese, onion} or {egg, oil, salt}, take the path that require {egg, oil, salt}.
Because, in this path, you need fewer input objects.


2. Visualize the retrieved task trees and check if they make sense. You should use it to
debug your program.


3. Save three task trees (one for iterative deepening search, one for heuristic 1 and one for
heuristic 2) in three separate .txt files. If the goal node does not exist in FOON, print that
“The goal node does not exist”.
Output:
Three task trees saved in three separate .txt files
