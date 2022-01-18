#![feature(test)]

use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use std::fs::File;
use std::io::{self, BufRead};

use priority_queue::PriorityQueue;

mod priority_queue;

#[derive(Eq, PartialEq, Hash, Debug)]
pub struct Edge<T>(Rc<T>, Rc<T>, usize);

/// A simple graph data structure
#[derive(PartialEq, Debug)]
pub struct EdgeListGraph<T: Eq + Hash> {
    vertices: HashSet<Rc<T>>,
    edges: HashSet<Edge<T>>,
}

#[derive(PartialEq, Debug)]
pub struct AdjacencyListGraph<T: Eq + Hash> {
    vertices: HashMap<Rc<T>, HashSet<Edge<T>>>,
}

/// A union find data structure where each item is a map entry with a refernce to its "parent"
pub struct UnionFind<T: Eq + Hash> {
    items: HashMap<Rc<T>, Rc<T>>,
}

trait Graph<T>: Default
where
    T: Eq + Hash,
{
    fn vertices<'a>(&'a self) -> Vec<&'a Rc<T>>;
    fn edges<'a>(&'a self) -> Vec<&'a Edge<T>>;

    fn add_vertex(&mut self, vertex: &Rc<T>);
    fn add_edge(&mut self, start: &Rc<T>, end: &Rc<T>, weight: usize);

    fn has_vertex(&self, vertex: &Rc<T>) -> bool;

    fn edges_from<'a>(&'a self, edge: &Rc<T>) -> Vec<&'a Edge<T>>;

    fn add_edge_safe(&mut self, start: &Rc<T>, end: &Rc<T>, weight: usize) {
        if !self.has_vertex(start) {
            self.add_vertex(start);
        }
        if !self.has_vertex(end) {
            self.add_vertex(end);
        }
        self.add_edge(start, end, weight);
    }

    /// Returns a minimuim spanning tree of a graph use Kruskal Algorithm.
    /// The graph is not modified or consumed by the operation, rather, all references are copied
    /// which is much cheaper than creating a deep copy.
    fn mst_kruskal(&self) -> Self {
        // The minimun spanning tree to return
        let mut mst = Self::default();
        let mut union_finder: UnionFind<T> = UnionFind::new();

        for vertex in self.vertices().into_iter() {
            union_finder.make_set(vertex);
        }

        let mut sorted_edges: Vec<&Edge<T>> = self.edges().into_iter().collect();

        // Sort the edges by weight in ascending order
        sorted_edges.sort_by(|Edge(_, _, a), Edge(_, _, b)| a.cmp(&b));

        for Edge(a, b, weight) in sorted_edges {
            // if a and b are not in the same set (ie. connected) the current edge is a light edge
            if union_finder.find_set(a) != union_finder.find_set(b) {
                mst.add_vertex(a);
                mst.add_vertex(b);
                mst.add_edge(a, b, *weight);
                // If something goes wrong here, just throw an error.
                // It might be better to return a Result<Self, Err>
                union_finder.union(a, b).unwrap();
            }
        }

        // Return the mst
        mst
    }

    fn mst_prim(&self) -> Self {
        if self.vertices().is_empty() {
            return Self::default();
        }

        let mut mst = Self::default();

        let mut pq: PriorityQueue<&Edge<T>> = PriorityQueue::new();

        let mut nodes: Vec<&Rc<T>> = self.vertices();

        let start = nodes.pop().unwrap();
        mst.add_vertex(start);

        self.edges_from(start)
            .into_iter()
            .for_each(|edge| pq.insert(edge));

        while nodes.len() > 0 {
            if let Some(Edge(start, end, weight)) = pq.pop() {
                if mst.has_vertex(start) && !mst.has_vertex(end) {
                    mst.add_vertex(end);
                    mst.add_edge(start, end, *weight);
                    self.edges_from(end)
                        .into_iter()
                        .for_each(|edge| pq.insert(edge));
                } else if mst.has_vertex(end) && !mst.has_vertex(start) {
                    mst.add_vertex(start);
                    mst.add_edge(start, end, *weight);
                    self.edges_from(start)
                        .into_iter()
                        .for_each(|edge| pq.insert(edge));
                }
            } else {
                return mst;
            }
        }

        mst
    }

    /// Calculates the number of connnected components in a graph by constructing a [UnionFind] and
    /// calling the [sets](UnionFind::sets) method on the [UnionFind] object.
    fn connected_components(&self) -> usize {
        let mut union_finder = UnionFind::new();

        // add each vertex to the union find object
        for vertex in self.vertices().into_iter() {
            union_finder.make_set(vertex);
        }

        // connect connected edges in the union find object
        for Edge(a, b, _) in self.edges().into_iter() {
            union_finder.union(a, b).unwrap();
        }

        union_finder.sets()
    }
}

fn load_from<'a, T: Graph<String> + Default>(path: &str) -> T {
    let mut graph = T::default();

    let file = File::open(path).expect("Unable to open graph file");

    for line in io::BufReader::new(file)
        .lines()
        .map(|line| line.expect("Unable to read line"))
    {
        let words: Vec<_> = line.split(" ").collect();
        if words[0] == "a" {
            graph.add_edge_safe(
                &Rc::new(words[1].into()),
                &Rc::new(words[2].into()),
                words[3]
                    .parse()
                    .expect("Cannot convert edge weight to integer"),
            );
        }
    }

    graph
}

impl<T: Eq + Hash> EdgeListGraph<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Eq + Hash> Default for EdgeListGraph<T> {
    fn default() -> Self {
        Self {
            vertices: HashSet::new(),
            edges: HashSet::new(),
        }
    }
}

impl<T: Eq + Hash> Graph<T> for EdgeListGraph<T> {
    fn vertices<'a>(&'a self) -> Vec<&'a Rc<T>> {
        self.vertices.iter().collect()
    }

    fn edges<'a>(&'a self) -> Vec<&'a Edge<T>> {
        self.edges.iter().collect()
    }

    fn add_vertex(&mut self, t: &Rc<T>) {
        self.vertices.insert(Rc::clone(t));
    }

    fn add_edge(&mut self, a: &Rc<T>, b: &Rc<T>, weight: usize) {
        self.edges.insert(Edge(Rc::clone(a), Rc::clone(b), weight));
    }

    fn has_vertex(&self, vertex: &Rc<T>) -> bool {
        self.vertices.contains(vertex)
    }

    fn edges_from<'a>(&'a self, vertex: &Rc<T>) -> Vec<&'a Edge<T>> {
        self.edges
            .iter()
            .filter(|&Edge(a, b, _)| a == vertex || b == vertex)
            .collect()
    }
}

impl<T: Eq + Hash> AdjacencyListGraph<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Eq + Hash> Default for AdjacencyListGraph<T> {
    fn default() -> Self {
        Self {
            vertices: HashMap::new(),
        }
    }
}

impl<T: Eq + Hash> Graph<T> for AdjacencyListGraph<T> {
    fn vertices<'a>(&'a self) -> Vec<&'a Rc<T>> {
        self.vertices.keys().collect()
    }

    fn edges<'a>(&'a self) -> Vec<&'a Edge<T>> {
        self.vertices
            .values()
            .flat_map(|edge_list| edge_list.iter())
            .collect()
    }

    fn add_vertex(&mut self, vertex: &Rc<T>) {
        self.vertices
            .entry(Rc::clone(vertex))
            .or_insert_with(|| HashSet::new());
    }

    fn add_edge(&mut self, start: &Rc<T>, end: &Rc<T>, weight: usize) {
        self.vertices
            .get_mut(start)
            .expect("Cannot add edge to vertex that does not exist")
            .insert(Edge(Rc::clone(start), Rc::clone(end), weight));
        self.vertices
            .get_mut(end)
            .expect("Cannot add edge to vertex that does not exist")
            .insert(Edge(Rc::clone(end), Rc::clone(start), weight));
    }

    fn has_vertex(&self, vertex: &Rc<T>) -> bool {
        self.vertices.contains_key(vertex)
    }

    fn edges_from(&self, vertex: &Rc<T>) -> Vec<&Edge<T>> {
        self.vertices
            .get(vertex)
            .expect("Cannot list edges from vertex that does not exist")
            .iter()
            .collect()
    }
}

impl<T: PartialEq> PartialOrd for Edge<T> {
    fn partial_cmp(&self, &Edge(_, _, other): &Self) -> Option<Ordering> {
        Some(self.2.cmp(&other))
    }
}

impl<T: Eq> Ord for Edge<T> {
    fn cmp(&self, &Edge(_, _, other): &Self) -> Ordering {
        self.2.cmp(&other)
    }
}

impl<T> UnionFind<T>
where
    T: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
        }
    }

    pub fn make_set(&mut self, t: &Rc<T>) {
        self.items.insert(Rc::clone(&t), Rc::clone(&t));
    }

    /// Mutates the underlying [HashMap] so that `a` and `b` have the same parent.
    /// There is not guarantee as to which parents `a` and `b` will end up with, only that they are
    /// the same.
    pub fn union(&mut self, a: &Rc<T>, b: &Rc<T>) -> Result<(), &str> {
        if let (Ok(a), Ok(b)) = (self.find_set(a), self.find_set(b)) {
            self.items.insert(a, b);
            Ok(())
        } else {
            Err("Cannot perform union on objects without set")
        }
    }

    /// Returns a [Rc] pointer to the "parent" of a given item.
    /// Returns `Err` if `t` is not the the set
    pub fn find_set(&self, t: &Rc<T>) -> Result<Rc<T>, &str> {
        match self.items.get(t) {
            Some(p) if p == t => Ok(Rc::clone(p)),
            Some(p) => self.find_set(p),
            None => Err("Cannot find set of object without set"),
        }
    }

    /// Returns the number of unique sets in the struct
    pub fn sets(&self) -> usize {
        // count the number of items who are their own parent
        self.items
            .iter()
            .filter(|(item, parent)| item == parent)
            .count()
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_kruskal_edgelist_rome(b: &mut Bencher) {
        let rome = load_from::<EdgeListGraph<String>>("graphs/Rome.gr");
        b.iter(|| rome.mst_kruskal());
    }

    #[bench]
    fn bench_prim_edgelist_rome(b: &mut Bencher) {
        let rome = load_from::<EdgeListGraph<String>>("graphs/Rome.gr");
        b.iter(|| rome.mst_prim());
    }

    #[bench]
    fn bench_kruskal_adjacencylist_rome(b: &mut Bencher) {
        let rome = load_from::<AdjacencyListGraph<String>>("graphs/Rome.gr");
        b.iter(|| rome.mst_kruskal());
    }

    #[bench]
    fn bench_prim_adjacencylist_rome(b: &mut Bencher) {
        let rome = load_from::<AdjacencyListGraph<String>>("graphs/Rome.gr");
        b.iter(|| rome.mst_prim());
    }

    #[bench]
    fn bench_kruskal_edgelist_ny(b: &mut Bencher) {
        let ny = load_from::<EdgeListGraph<String>>("graphs/NY.gr");
        b.iter(|| ny.mst_kruskal());
    }

    #[bench]
    fn bench_prim_edgelist_ny(b: &mut Bencher) {
        let ny = load_from::<EdgeListGraph<String>>("graphs/NY.gr");
        b.iter(|| ny.mst_prim());
    }

    #[bench]
    fn bench_kruskal_adjacencylist_ny(b: &mut Bencher) {
        let ny = load_from::<AdjacencyListGraph<String>>("graphs/NY.gr");
        b.iter(|| ny.mst_kruskal());
    }

    #[bench]
    fn bench_prim_adjacencylist_ny(b: &mut Bencher) {
        let ny = load_from::<AdjacencyListGraph<String>>("graphs/NY.gr");
        b.iter(|| ny.mst_prim());
    }

    #[bench]
    fn bench_kruskal_edgelist_sf(b: &mut Bencher) {
        let sf = load_from::<EdgeListGraph<String>>("graphs/SF.gr");
        b.iter(|| sf.mst_kruskal());
    }

    #[bench]
    fn bench_prim_edgelist_sf(b: &mut Bencher) {
        let sf = load_from::<EdgeListGraph<String>>("graphs/SF.gr");
        b.iter(|| sf.mst_prim());
    }

    #[bench]
    fn bench_kruskal_adjacencylist_sf(b: &mut Bencher) {
        let sf = load_from::<AdjacencyListGraph<String>>("graphs/SF.gr");
        b.iter(|| sf.mst_kruskal());
    }

    #[bench]
    fn bench_prim_adjacencylist_sf(b: &mut Bencher) {
        let sf = load_from::<AdjacencyListGraph<String>>("graphs/SF.gr");
        b.iter(|| sf.mst_prim());
    }

    #[test]
    fn test_adjacencylist() {
        let mut graph = AdjacencyListGraph::new();

        let p1 = Rc::new((1, 1));
        let p2 = Rc::new((4, 6));

        let p3 = Rc::new((-3, 8));

        assert!(!graph.has_vertex(&p1));

        graph.add_vertex(&p1);

        assert!(graph.has_vertex(&p1));

        graph.add_vertex(&p2);
        graph.add_vertex(&p3);

        assert!(graph.has_vertex(&p2));
        assert!(graph.has_vertex(&p3));
    }

    #[test]
    fn test_connected_components() {
        let mut graph = EdgeListGraph::new();
        let p1 = Rc::new((1, 1));
        let p2 = Rc::new((4, 6));

        let p3 = Rc::new((-3, 8));
        let p4 = Rc::new((-4, 4));
        let p5 = Rc::new((-9, -20));

        let p6 = Rc::new((-2, -19));
        let p7 = Rc::new((-3, -3));

        let p8 = Rc::new((17, -13));

        graph.add_vertex(&p1);
        graph.add_vertex(&p2);
        graph.add_vertex(&p3);
        graph.add_vertex(&p4);
        graph.add_vertex(&p5);
        graph.add_vertex(&p6);
        graph.add_vertex(&p7);
        graph.add_vertex(&p8);

        graph.add_edge(&p1, &p2, 0);
        graph.add_edge(&p3, &p4, 0);
        graph.add_edge(&p3, &p5, 0);
        graph.add_edge(&p6, &p7, 0);

        assert_eq!(graph.connected_components(), 4);
    }

    #[test]
    fn test_edgelist_kruskal() {
        test_kruskal::<EdgeListGraph<&str>>();
    }

    #[test]
    fn test_edgelist_prim() {
        test_prim::<EdgeListGraph<&str>>();
    }

    #[test]
    fn test_adjacencylist_kruskal() {
        test_kruskal::<AdjacencyListGraph<&str>>();
    }

    #[test]
    fn test_adjacencylist_prim() {
        test_prim::<AdjacencyListGraph<&str>>();
    }

    #[test]
    fn test_union_find() {
        let mut union_find = UnionFind::<usize>::new();
        let one = Rc::new(1);
        let two = Rc::new(2);
        let five = Rc::new(5);
        let seven_twenty = Rc::new(720);

        assert!(union_find.find_set(&one).is_err());
        assert!(union_find.find_set(&five).is_err());

        union_find.make_set(&one);
        union_find.make_set(&five);
        union_find.make_set(&two);
        union_find.make_set(&seven_twenty);

        assert_eq!(union_find.find_set(&two), Ok(Rc::clone(&two)));
        assert_eq!(union_find.find_set(&five), Ok(Rc::clone(&five)));

        assert!(union_find.union(&one, &five).is_ok());
        assert!(union_find.union(&two, &seven_twenty).is_ok());

        assert_eq!(union_find.find_set(&one), union_find.find_set(&five));
        assert_ne!(
            union_find.find_set(&seven_twenty),
            union_find.find_set(&one)
        );

        assert!(union_find.union(&five, &two).is_ok());

        assert_eq!(
            union_find.find_set(&one),
            union_find.find_set(&seven_twenty)
        );
    }

    fn test_kruskal<'a, T: Graph<&'a str> + Default + PartialEq + std::fmt::Debug>() {
        let mut graph = T::default();

        let cow = Rc::new("cow");
        let how = Rc::new("how");
        let hat = Rc::new("hat");
        let mow = Rc::new("mow");
        let cowering = Rc::new("cowering");
        let hatred = Rc::new("hatred");

        graph.add_vertex(&cow);
        graph.add_vertex(&how);
        graph.add_vertex(&hat);
        graph.add_vertex(&mow);
        graph.add_vertex(&cowering);
        graph.add_vertex(&hatred);

        graph.add_edge(&cow, &how, 1);
        graph.add_edge(&how, &hat, 2);
        graph.add_edge(&hat, &mow, 3);
        graph.add_edge(&how, &cowering, 6);
        graph.add_edge(&hat, &cowering, 7);
        graph.add_edge(&cow, &hatred, 6);

        let mut mst = T::default();

        mst.add_vertex(&cow);
        mst.add_vertex(&how);
        mst.add_vertex(&hat);
        mst.add_vertex(&mow);
        mst.add_vertex(&cowering);
        mst.add_vertex(&hatred);

        mst.add_edge(&cow, &how, 1);
        mst.add_edge(&how, &hat, 2);
        mst.add_edge(&hat, &mow, 3);
        mst.add_edge(&how, &cowering, 6);
        mst.add_edge(&cow, &hatred, 6);

        assert_eq!(graph.mst_kruskal(), mst);
    }

    fn test_prim<'a, T: Graph<&'a str> + Default + PartialEq + std::fmt::Debug>() {
        let mut graph = T::default();

        let cow = Rc::new("cow");
        let how = Rc::new("how");
        let hat = Rc::new("hat");
        let mow = Rc::new("mow");
        let cowering = Rc::new("cowering");
        let hatred = Rc::new("hatred");

        graph.add_vertex(&cow);
        graph.add_vertex(&how);
        graph.add_vertex(&hat);
        graph.add_vertex(&mow);
        graph.add_vertex(&cowering);
        graph.add_vertex(&hatred);

        graph.add_edge(&cow, &how, 1);
        graph.add_edge(&how, &hat, 2);
        graph.add_edge(&hat, &mow, 3);
        graph.add_edge(&how, &cowering, 6);
        graph.add_edge(&hat, &cowering, 7);
        graph.add_edge(&cow, &hatred, 6);

        let mut mst = T::default();

        mst.add_vertex(&cow);
        mst.add_vertex(&how);
        mst.add_vertex(&hat);
        mst.add_vertex(&mow);
        mst.add_vertex(&cowering);
        mst.add_vertex(&hatred);

        mst.add_edge(&cow, &how, 1);
        mst.add_edge(&how, &hat, 2);
        mst.add_edge(&hat, &mow, 3);
        mst.add_edge(&how, &cowering, 6);
        mst.add_edge(&cow, &hatred, 6);

        assert_eq!(graph.mst_prim(), mst);
    }
}
