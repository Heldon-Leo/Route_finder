import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq


# --------------------------
# Tamil Nadu District Graph
# --------------------------
graph_data = {
    'Chennai Central': ['Arakkonam', 'Tambaram', 'Chengalpattu'],
    'Tambaram': ['Chennai Central', 'Chengalpattu', 'Villupuram'],
    'Chengalpattu': ['Tambaram', 'Chennai Central', 'Villupuram', 'Tindivanam'],
    'Villupuram': ['Chengalpattu', 'Tindivanam', 'Tiruchirappalli', 'Cuddalore', 'Ariyalur'],
    'Tindivanam': ['Chengalpattu', 'Villupuram'],
    'Cuddalore': ['Villupuram', 'Chidambaram', 'Mayiladuthurai'],
    'Chidambaram': ['Cuddalore', 'Mayiladuthurai', 'Sirkazhi'],
    'Mayiladuthurai': ['Chidambaram', 'Sirkazhi', 'Kumbakonam'],
    'Sirkazhi': ['Chidambaram', 'Mayiladuthurai'],
    'Kumbakonam': ['Mayiladuthurai', 'Thanjavur'],
    'Thanjavur': ['Kumbakonam', 'Tiruchirappalli', 'Thiruvarur', 'Pudukkottai'],
    'Thiruvarur': ['Thanjavur', 'Nagapattinam'],
    'Nagapattinam': ['Thiruvarur', 'Velankanni'],
    'Velankanni': ['Nagapattinam'],
    'Tiruchirappalli': ['Thanjavur', 'Villupuram', 'Karur', 'Dindigul', 'Ariyalur', 'Perambalur'],
    'Ariyalur': ['Tiruchirappalli', 'Villupuram', 'Perambalur'],
    'Perambalur': ['Tiruchirappalli', 'Ariyalur'],
    'Pudukkottai': ['Thanjavur', 'Karaikudi'],
    'Karaikudi': ['Pudukkottai', 'Manamadurai'],
    'Manamadurai': ['Karaikudi', 'Sivaganga'],
    'Sivaganga': ['Manamadurai', 'Madurai'],
    'Karur': ['Tiruchirappalli', 'Erode'],
    'Erode': ['Karur', 'Salem', 'Coimbatore'],
    'Salem': ['Erode', 'Jolarpettai', 'Namakkal', 'Tiruppur'],
    'Namakkal': ['Salem'],
    'Jolarpettai': ['Salem', 'Katpadi', 'Vellore'],
    'Katpadi': ['Jolarpettai', 'Vellore', 'Arakkonam'],
    'Vellore': ['Katpadi', 'Jolarpettai'],
    'Arakkonam': ['Katpadi', 'Chennai Central'],
    'Dindigul': ['Tiruchirappalli', 'Madurai'],
    'Madurai': ['Dindigul', 'Virudhunagar', 'Sivaganga'],
    'Virudhunagar': ['Madurai', 'Tirunelveli', 'Sivakasi'],
    'Sivakasi': ['Virudhunagar', 'Rajapalayam'],
    'Rajapalayam': ['Sivakasi', 'Tenkasi'],
    'Tenkasi': ['Rajapalayam', 'Tirunelveli'],
    'Tirunelveli': ['Virudhunagar', 'Nagercoil', 'Thoothukudi', 'Tenkasi'],
    'Nagercoil': ['Tirunelveli', 'Kanyakumari'],
    'Kanyakumari': ['Nagercoil'],
    'Thoothukudi': ['Tirunelveli'],
    'Coimbatore': ['Erode', 'Tiruppur'],
    'Tiruppur': ['Coimbatore', 'Salem']
}

# Weighted version (all edges cost 1)
graph_weighted = {}
for node, neighbors in graph_data.items():
    graph_weighted[node] = {n: 1 for n in neighbors}

# --------------------------
# Algorithms
# --------------------------

def bfs(graph, start, end):
    from collections import deque
    visited = set()
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == end:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(path + [neighbor])
    return []

def dfs(graph, start, end):
    visited = set()
    stack = [[start]]
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == end:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(path + [neighbor])
    return []

def ucs(graph, start, end):
    pq = [(0, [start])]
    visited = set()
    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == end:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + weight, path + [neighbor]))
    return []

def dls(graph, start, end, limit):
    def recursive_dls(node, end, limit, path, visited):
        if node == end:
            return path
        if limit <= 0:
            return None
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result = recursive_dls(neighbor, end, limit - 1, path + [neighbor], visited)
                if result:
                    return result
        return None
    return recursive_dls(start, end, limit, [start], set())

def ids(graph, start, end, max_depth=10):
    for depth in range(max_depth):
        result = dls(graph, start, end, depth)
        if result:
            return result
    return []

def heuristic(a, b):
    # Dummy heuristic: difference in string lengths (replace with coordinates if available)
    return abs(len(a) - len(b))

def greedy(graph, start, end):
    pq = [(heuristic(start, end), [start])]
    visited = set()
    while pq:
        _, path = heapq.heappop(pq)
        node = path[-1]
        if node == end:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                heapq.heappush(pq, (heuristic(neighbor, end), path + [neighbor]))
    return []

def astar(graph, start, end):
    pq = [(heuristic(start, end), 0, [start])]
    visited = set()
    while pq:
        est_total, cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == end:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph.get(node, {}).items():
                g = cost + weight
                f = g + heuristic(neighbor, end)
                heapq.heappush(pq, (f, g, path + [neighbor]))
    return []

# --------------------------
# Visualization
# --------------------------
def draw_graph(graph, path=None):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(18, 12))

    # Base graph
    nx.draw(G, pos, with_labels=True, node_color='lightgray',
            edge_color='lightgray', node_size=800, font_size=8)

    if path:
        edge_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edge_path, edge_color='red', width=2)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=900)

    st.pyplot(plt.gcf())

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="TN District Route Finder", layout="wide")
st.title("🧭 Tamil Nadu District Route Finder")
st.markdown("Find routes using different AI search algorithms.")

districts = sorted(graph_data.keys())
col1, col2, col3 = st.columns(3)
with col1:
    source = st.selectbox("Select Source District", districts)
with col2:
    destination = st.selectbox("Select Destination District", districts)
with col3:
    algo = st.selectbox("Select Algorithm",
                        ["BFS", "DFS", "UCS", "DLS", "IDS", "Greedy", "A*"])

depth_limit = None
if algo == "DLS":
    depth_limit = st.slider("Depth Limit", 1, 10, 3)

if st.button("Find Route"):
    if source == destination:
        st.warning("Source and destination are the same.")
    else:
        if algo == "BFS":
            path = bfs(graph_data, source, destination)
        elif algo == "DFS":
            path = dfs(graph_data, source, destination)
        elif algo == "UCS":
            path = ucs(graph_weighted, source, destination)
        elif algo == "DLS":
            path = dls(graph_data, source, destination, depth_limit)
        elif algo == "IDS":
            path = ids(graph_data, source, destination, max_depth=10)
        elif algo == "Greedy":
            path = greedy(graph_data, source, destination)
        elif algo == "A*":
            path = astar(graph_weighted, source, destination)
        else:
            path = []

        if path:
            st.success(f"✅ Path found using {algo}: {' ➝ '.join(path)}")
            draw_graph(graph_data, path)
        else:
            st.error("❌ No route found between the selected districts.")
