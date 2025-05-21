import networkx as nx
import pickle

GRAPH_PATH = "doc_graph.gpickle"


def load_doc_graph(graph_path="doc_graph.gpickle"):
    with open(graph_path, "rb") as f:
        return pickle.load(f)

def graph_hop_search(G, entry_ids, hops=1, max_nodes=50):
    visited = set(entry_ids)
    frontier = set(entry_ids)

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbors = G.neighbors(node)
            for nbr in neighbors:
                if nbr not in visited:
                    next_frontier.add(nbr)
        visited.update(next_frontier)
        frontier = next_frontier

    return list(visited)[:max_nodes]
