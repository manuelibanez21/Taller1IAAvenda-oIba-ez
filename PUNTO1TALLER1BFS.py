from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

# BFS para obtener la ruta
def bfs(graph, start, goal):
    visited = set()
    queue = deque([[start]])

    if start == goal:
        return [start]

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node not in visited:
            neighbors = graph[node]
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

                if neighbor == goal:
                    return new_path
            visited.add(node)
    return None

# --- Layout jerárquico ---
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError('El grafo no es un árbol')

    if root is None:
        root = list(G.nodes)[0]

    def _hierarchy_pos(G, root, left, right, vert_loc, xcenter, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = (right - left) / len(children)
            nextx = left + dx / 2
            for child in children:
                pos = _hierarchy_pos(G, child, left=nextx - dx/2, right=nextx + dx/2,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
                nextx += dx
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_loc, xcenter)


def bfs_order(graph, start):
    visited, queue, seen = [], deque([start]), {start}
    while queue:
        node = queue.popleft()
        visited.append(node)
        for nbr in graph[node]:
            if nbr not in seen:
                seen.add(nbr); queue.append(nbr)
    return visited

# --- Definición del grafo ---
graph = {
    'S' : ['A','B', 'D','E'],
    'A' : ['F', 'G'],
    'F' : ['M'],
    'M' : ['N'],
    'N' : [],
    'G' : [],
    'B' : ['H', 'R'],
    'H' : ['O','Q'],
    'O' : ['P'],
    'P' : [],
    'Q' : ['U'],
    'U' : ['V','W'],
    'V' : [],
    'W' : [],
    'R' : ['X', 'T'],
    'X' : [],
    'T' : ['GG'],
    'GG' : [],
    'D' : ['J'],
    'J' : ['Y'],
    'Y' : ['Z'],
    'Z' : ['AA', 'BB'],
    'AA' : [],
    'BB' : [],
    'E': ['K','L'],
    'K' : ['I'],
    'L' : ['CC'],
    'I' : [],
    'CC' : ['DD','EE'],
    'DD' : [],
    'EE' : ['FF'],
    'FF' : []
}

start_node = 'S'
end_node = 'W'
best_path = bfs(graph, start_node, end_node)
print("BEST PATH:", best_path)

# --- Grafo con layout jerárquico ---
G = nx.DiGraph()
for node, neighbors in graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

pos = hierarchy_pos(G, root=start_node)

nx.draw(G, pos, with_labels=False, node_size= 800, node_color="lightblue", font_weight="bold", arrows=True)

# Resaltar la ruta
if best_path:
    path_edges = list(zip(best_path, best_path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=best_path, node_color="red")
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

order = bfs_order(graph, start_node)
labels = {node: f"{node}\n({i+1})" for i, node in enumerate(order)}
nx.draw_networkx_labels(G, pos, labels=labels)
plt.title(f"Camino más corto de {start_node} a {end_node}")
plt.show()
