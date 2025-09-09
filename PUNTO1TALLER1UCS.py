import heapq
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def reconstruir_camino(padre, meta):
    camino = []
    nodo = meta
    while nodo is not None:
        camino.append(nodo)
        nodo = padre[nodo]
    return camino[::-1]

def ucs(grafo, inicio, meta):
    cola = [(0, inicio)]
    visitados = set()
    costo_acumulado = {inicio: 0}
    padre = {inicio: None}

    while cola:
        costo, nodo = heapq.heappop(cola)
        if nodo in visitados:
            continue
        visitados.add(nodo)

        if nodo == meta:
            break

        for vecino, costo_arista in grafo[nodo]:
            nuevo_costo = costo + costo_arista
            if vecino not in costo_acumulado or nuevo_costo < costo_acumulado[vecino]:
                costo_acumulado[vecino] = nuevo_costo
                heapq.heappush(cola, (nuevo_costo, vecino))
                padre[vecino] = nodo

    return reconstruir_camino(padre, meta), costo_acumulado.get(meta, float('inf'))

# Grafo con costos
grafo_costo = {
    'S' : [ ('A', 4),('B', 6), ('D', 3), ('E',8) ],
    'A' : [('F', 6), ('G', 5)],
    'F' : [('M', 8)],
    'M' : [('N', 5)],
    'N' : [],
    'G' : [],
    'B' : [('H', 6),( 'R', 0)],
    'H' : [('O',6),('Q',7)],
    'O' : [('P', 5)],
    'P' : [],
    'Q' : [('U', 7)],
    'U' : [('V', 4),('W',9)],
    'V' : [],
    'W' : [],
    'R' : [('X',5),('T', 3)],
    'X' : [],
    'T' : [('GG', 3)],
    'GG' : [],
    'D' : [('J', 2)],
    'J' : [('Y', 8)],
    'Y' : [('Z', 3)],
    'Z' : [('AA', 4),('BB', 2)],
    'AA' : [],
    'BB' : [],
    'E': [('K', 4), ('L', 1)],
    'K' : [('I', 4)],
    'L' : [('CC', 3)],
    'I' : [],
    'CC' : [('DD', 7),('EE', 3)],
    'DD' : [],
    'EE' : [('FF', 2)],
    'FF' : []

}

# Ejecutar UCS
camino, costo = ucs(grafo_costo, 'S', 'W')
print("BEST PATH:", camino, "COST:", costo)

# Crear grafo con networkx
G = nx.DiGraph()
for nodo, vecinos in grafo_costo.items():
    for vecino, peso in vecinos:
        G.add_edge(nodo, vecino, weight=peso)

# Posiciones con graphviz
pos = {
    "S":  (0, 0), 
    "A" : (-5, -1), "B" : (-2,-1), "D" : (2, -1), "E" : (5, -1),
    "F" : (-6, -2), "G" : (-4,-2), "H" : (-2,-2), "R" : (0, -2), "J" : (3,-2), "K" : (4,-2), "L" : (6, -2),
    "M" : (-6, -3), "O" : (-4,-3), "Q" : (-3, -3), "X" : (-1, -3), "T" : (1,-3), "Y" : (2,-3), "I" : (4,-3), "CC" : (5,-3),
    "N" : (-6,-4), "P" : (-4,-4), "U" : (-2,-4), "GG" : (1,-4), "Z" : (3,-4), "DD" : (4,-4), "EE" : (6,-4),
    "V" : (-2,-5), "W" : (1,-5), "AA" : (2,-5), "BB" : (4,-5), "FF" : (6, -5)

}


# Dibujar nodos y aristas
nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", arrows=True)

# Dibujar pesos
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Resaltar el mejor camino
edge_list = list(zip(camino, camino[1:]))
nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color="red", width=2)

plt.title(f"Uniform Cost Search (Best Path Cost={costo})")
plt.show()
