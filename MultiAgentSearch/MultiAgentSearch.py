import random
import networkx as nx

# Create graph with 40 nodes
G = nx.Graph()
for i in range(1, 41):
    G.add_node(i)

# Connect nodes in a loop
for i in range(1, 40):
    G.add_edge(i, i+1)
G.add_edge(40, 1) 

num_edges_added = 0
while num_edges_added < 10:
  u = random.randint(1, 40) 
  v = random.randint(1, 40)

  if u != v and not G.has_edge(u, v):
    G.add_edge(u, v)
    print(f"Added edge: ({u}, {v})")
    num_edges_added += 1

# Initialize target at random node            
target = random.randint(1, 40) 

# Target takes a random walk
for i in range(10):
    neighbors = list(G.neighbors(target))
    next_node = random.choice(neighbors)
    target = next_node

    print(f"Step {i+1}: Target is at node {target}")