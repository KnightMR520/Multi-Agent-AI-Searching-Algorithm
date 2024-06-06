import random
import networkx as nx
from GraphGenerator import GraphGenerator
from collections import deque
import time

# Project by Andrew Galifi and Nick Russo

def main(agent_selected):
    g, agent, target, shortest_paths, observed_node, probabilities = GraphGenerator()

    target_movement_range = [1]

    step = 0
    examinations = 0
    while agent != target:
        neighbors = list(g.neighbors(target))
        next_node = random.choice(neighbors)

        target = next_node
        no_step = False
        match agent_selected:
            case 0:
                pass
            case 1:
                agent, no_step = agentOneModel(agent, target, g)
            case 2:
                agent, no_step = agentTwoModel(agent, target, g, shortest_paths)
            case 3:
                agent = observed_node
            case 4:
                agent, probabilities, e = agentFourModel(target, g, probabilities, target_movement_range)
                examinations += e
            case 5:
                agent, probabilities, e = agentFiveModel(target, g, probabilities)
                examinations += e
            case 6:
                agent, probabilities, e, no_step = agentSixModel(agent, target, g, probabilities, target_movement_range)
                examinations += e
            case 7:
                agent, probabilities, e, no_step = agentSevenModel(agent, target, g, probabilities, shortest_paths)
                examinations += e

        if no_step:
            step -= 1
        step += 1

        if agent == target:
            if agent_selected == 4 or agent_selected == 5:
                return examinations
            if agent_selected == 6 or agent_selected == 7:
                return examinations, step
            return step


# this uses dijkstra's to find the shortest path between the agent's neighbors and the target,
# picking between the most optimal step options randomly,
# and then makes the first step towards it (unless its on target already)
def agentOneModel(agent, target, g):
    agent_path = []
    shortest = 999
    neighbors = g.neighbors(agent)

    if agent != target:
        for node in neighbors:
            length = nx.dijkstra_path_length(g, node, target)
            # "randomly" pick move out of optimal options
            if length <= shortest:
                shortest = length
                agent_path = nx.dijkstra_path(g, node, target)
        no_step = False
        return agent_path[0], no_step

    else:
        no_step = True
        return agent, no_step


# this uses astar with the manhattan heuristic to find the shortest path between the agent and the target,
# and then makes the first step towards it (unless its on target already)
# On average just slightly faster than agent1 by a step or two
def agentTwoModel(agent, target, g, sp):
    def distance(a, t):
        return len(sp[a][t]) - 1

    agent_path = nx.astar_path(g, agent, target, heuristic=distance)

    no_step = False

    if len(agent_path) == 1:
        no_step = True
        return agent_path[0], no_step

    return agent_path[1], no_step


# runs in 40 ish exams or O(Vertices)
def agentFourModel(target, g, probabilities, target_movement_range):
    node_to_observe = getMostLikelyNode(probabilities)
    examinations = 1

    if node_to_observe == target:
        return node_to_observe, probabilities, examinations

    updateProbabilities(probabilities, node_to_observe, g, target_movement_range[0])
    propagateProbabilities(probabilities, g, target_movement_range[0])

    return node_to_observe, probabilities, examinations


# this after testing 2000 ish runs, typically runs 2-4 examinations less than agent4. (38) ish

# it picks the most likely node, if it's not the target, it checks all of its neighbors. if none of them are the target,
# it sets the probability of the old node_to_observe to 0, and discourages rechecking the neighbors of node_to_observe
# until the iteration after the next one. It promotes the neighbors of the neighbors of the old node_to_observe, and the
# highest prob node will be chosen in next turn.
# agent5 also runs much faster, which can be included in the writeup
def agentFiveModel(target, g, probabilities):
    node_to_observe = getMostLikelyNode(probabilities)

    examinations = 1
    if node_to_observe == target:
        return node_to_observe, probabilities, examinations

    neighbors = list(g.neighbors(node_to_observe))
    for node in neighbors:
        examinations += 1
        if node == target:
            return node, probabilities, examinations

    # update probs
    probabilities[node_to_observe] = 0
    for node in neighbors:
        # discourage checking it again next around
        probabilities[node] = probabilities[node] / 2

        j = random.choice(list(g.neighbors(node)))

        while j not in g.neighbors(node_to_observe) and j != node_to_observe:
            j = random.choice(list(g.neighbors(node)))

            for x in g.neighbors(node):
                if probabilities[x] > j:
                    j = x

        # promote nodes 2 away to expand to
        if j not in g.neighbors(node_to_observe) and j != node_to_observe:
            probabilities[j] = probabilities[j] * 2

    for x in probabilities:
        if x != .050 and x != 0:
            probabilities[x] = .025

    return node_to_observe, probabilities, examinations


def agentSixModel(agent, target, g, probabilities, target_movement_range):
    '''if agent == target:
        no_step = True
        return agent, probabilities, 0, no_step
    probabilities[agent] = 0'''

    node_to_observe = getMostLikelyNode(probabilities)

    examinations = 1
    if node_to_observe == target:

        # calculate new probability
        prob = 1 / len(list(g.neighbors(node_to_observe)))
        # spread new probability
        for node in g:
            if node in list(g.neighbors(node_to_observe)) and node != agent:
                probabilities[node] = prob
            probabilities[node] = 0

        node_to_observe = getMostLikelyNode(probabilities)
        agent_path = nx.dijkstra_path(g, agent, node_to_observe)

        no_step = False
        if len(agent_path) == 1:
            no_step = True
            return agent_path[0], probabilities, examinations, no_step
        # simulate dijkstra visiting nodes of all node_to_observe neighbors
        examinations += len(list(g.neighbors(agent)))
        return agent_path[1], probabilities, examinations, no_step

    # when node_to_observe does not equal target
    updateProbabilities(probabilities, node_to_observe, g, target_movement_range[0])
    propagateProbabilities(probabilities, g, target_movement_range[0])
    probabilities[agent] = 0

    node_to_observe = getMostLikelyNode(probabilities)
    agent_path = nx.dijkstra_path(g, agent, node_to_observe)

    no_step = False
    if len(agent_path) == 1:
        no_step = True
        return agent_path[0], probabilities, examinations, no_step

    # simulate dijkstra visiting nodes of all node_to_observe neighbors
    examinations += len(list(g.neighbors(agent)))
    return agent_path[1], probabilities, examinations, no_step


# comb of 5 and 2 (astar and optimized probability)

def agentSevenModel(agent, target, g, probabilities, sp):
    def distance(a, t):
        return len(sp[a][t]) - 1

    if agent == target:
        no_step = True
        return agent, probabilities, 0, no_step

    probabilities[agent] = 0
    node_to_observe = getMostLikelyNode(probabilities)

    examinations = 1
    if node_to_observe == target:

        # calculate new probability
        prob = 1 / len(list(g.neighbors(node_to_observe)))
        # spread new probability
        for node in g:
            if node in list(g.neighbors(node_to_observe)) and node != agent:
                probabilities[node] = prob
            probabilities[node] = 0

        node_to_observe = getMostLikelyNode(probabilities)
        agent_path = nx.astar_path(g, agent, node_to_observe, heuristic=distance)

        no_step = False
        if len(agent_path) == 1:
            no_step = True
            return agent_path[0], probabilities, examinations, no_step
        # simulate astar visiting nodes of lowest h value
        examinations += 1
        return agent_path[1], probabilities, examinations, no_step

    # when node_to_observe does not equal target
    '''neighbors = list(g.neighbors(node_to_observe))
    for node in neighbors:
        examinations += 1
        if node == target:
            return node, probabilities, examinations'''

    # update probs
    probabilities[node_to_observe] = 0
    for node in list(g.neighbors(node_to_observe)):
        # discourage checking it again next around
        probabilities[node] = probabilities[node] / 2

        j = random.choice(list(g.neighbors(node)))

        while j not in g.neighbors(node_to_observe) and j != node_to_observe:
            j = random.choice(list(g.neighbors(node)))

            for x in g.neighbors(node):
                if probabilities[x] > j:
                    j = x

        # promote nodes 2 away to expand to
        if j not in g.neighbors(node_to_observe) and j != node_to_observe:
            probabilities[j] = probabilities[j] * 2

    for x in probabilities:
        if x != .050 and x != 0:
            probabilities[x] = .025

    node_to_observe = getMostLikelyNode(probabilities)
    agent_path = nx.astar_path(g, agent, node_to_observe, heuristic=distance)

    no_step = False
    if len(agent_path) == 1:
        no_step = True
        return agent_path[0], probabilities, examinations, no_step

    # simulate astar visiting nodes of lowest h value
    examinations += 1
    return agent_path[1], probabilities, examinations, no_step


# This sets the checked nodes probability to 0 and spreads out its old probability to the rest of the nodes in range
def updateProbabilities(probabilities, checked_node, g, target_movement_range):
    probabilities[checked_node] = 0

    nodes_in_range = getNodesInRange(g, checked_node, target_movement_range)

    for n in nodes_in_range:
        probabilities[n] += probabilities[checked_node] / len(nodes_in_range)


# Spreads out the probabilities to nearby nodes based on the target_movement_range
def propagateProbabilities(probabilities, g, target_movement_range):
    new_probs = {}

    for node, prob in probabilities.items():

        # Get nodes in movement range
        nodes_in_range = getNodesInRange(g, node, target_movement_range)

        for n in nodes_in_range:
            new_probs[n] = new_probs.get(n, 0) + prob / len(nodes_in_range)

    return new_probs


# Makes a list of the most likely nodes with the given probabilities
def getMostLikelyNode(probabilities):
    max_probability = max(probabilities.values())
    likely_nodes = [n for n, p in probabilities.items() if p == max_probability]

    return random.choice(likely_nodes)


# BFS Implementation to find nodes in range of the target_movement_range
def getNodesInRange(g, start_node, range_limit):
    visited = set()
    queue = deque([start_node])

    for x in range(range_limit):

        current = queue.popleft()
        neighbors = g.neighbors(current)

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

    return list(visited)


steps = []
exams = []
times = []
target_movement_range = [1]
c = ""

for i in range(0, 8):
    if i == 3 or i == 4 or i == 5:  # for testing certain agents
        c = "examinations"
    elif i == 0 or i == 1 or i == 2:
        c = "steps"

    if i != 6 and i != 7:
        for _ in range(1, 501):
            startTime = time.time()
            steps.append(main(i))
            endTime = time.time()
            times.append(endTime - startTime)
        print(f"Average {c} of Agent {i}: " + str(round(sum(steps) / 500)))
        avgTime = sum(times) / 500
        print(f"Average time of Agent  {i}: " + str(round(avgTime, 5)) + "\n")
        steps = []
        times = []

    if i == 6 or i == 7:
        for _ in range(1, 501):
            startTime = time.time()
            ex, s = main(i)
            exams.append(ex)
            steps.append(s)
            endTime = time.time()
            times.append(endTime - startTime)
        print(f"Average steps of Agent {i}: " + str(round(sum(steps) / 500)))
        print(f"Average examinations of Agent {i}: " + str(round(sum(exams) / 500)))
        avgTime = sum(times) / 500
        print(f"Average time of Agent  {i}: " + str(round(avgTime, 5)) + "\n")
        steps = []
        exams = []
        times = []
