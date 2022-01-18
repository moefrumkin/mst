# MST

An exploration of [Minimum Spanning Tree](https://en.wikipedia.org/wiki/Minimum_spanning_tree) algorithms benchmarked on real world data.

## Algorithms

Two different algorithms are used to find minimum spanning trees of graphs, [Kruskal's Algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm) and [Prim's Algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm). My implementation of Prim's algorithm uses a heap based priority queue, which I implemented myself as practice, instead of using Rust's default implementation.

## Graphs

I also implemented two different graphs representations. The first, ```EdgeListGraph``` represents a graph as a list of vertices and edges. The second, ```AdjacencyListGraph``` represents a graph as a map from vertices to sets of edges.

## Data
The testing data can be found [here](http://www.diag.uniroma1.it//challenge9/download.shtml). I used the graphs for Rome, San Fransisco, and New York.

## Running
To run, download the previously mentioned graph data into a top level directory called ```graphs```. Running ```cargo test``` will run the benchmarks, though they will take a long time.

## Performance

With Kruskal's algorithm, the edge list implementation is somewhat faster than the adjanency list. My guess as to why this is the case would be that the adjacency list requires a hash map lookup.

Since finding all edges from a given vertex is linear on the edge list implementation, Prim's algorithm is very slow with that implementation. However prims algorithm is the fastest algorithm when using the adjacency list graph implementation.