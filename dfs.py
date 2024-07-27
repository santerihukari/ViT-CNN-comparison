# Using a Python dictionary to act as an adjacency list
graph1 = {
  '1': ['2'],
  '2': ['3'],
  '3': ['4'],
  '4': []
}
graph2 = {
  '1': ['2','4'],
  '2': ['3'],
  '3': ['4'],
  '4': ['5'],
  '5': ['6'],
  '6': []

}

visited1 = set() # Set to keep track of visited nodes of graph.
visited2 = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
print("Following is the Depth-First Search")
dfs(visited1, graph1, '1')
print("Following is the Depth-First Search")
dfs(visited2, graph2, '1')
