/// A minimum priority queue
pub struct PriorityQueue<T: Ord> {
    items: Vec<T>,
}

impl<T: Ord> PriorityQueue<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn insert(&mut self, t: T) {
        self.items.push(t);
        self.shift_up(self.items.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.items.is_empty() {
            None
        } else {
            // Swap remove does exactly what we want it to do :)
            let item = self.items.swap_remove(0);
            self.shift_down(0);
            Some(item)
        }
    }

    /// Upshifts the `i`th element in the items [Vec] so that the heap property is preserved
    fn shift_up(&mut self, i: usize) {
        if i == 0 {
            return;
        }
        let mut current = i;
        let mut parent = Self::parent(i).unwrap();

        while current > 0 && self.items[current] < self.items[parent] {
            self.items.swap(current, parent);
            current = parent;
            match Self::parent(current) {
                Some(item) => parent = item,
                None => return,
            }
        }
    }

    /// Downshifts the `i`th element in the items [Vec] so that the heap property in preserved
    fn shift_down(&mut self, i: usize) {
        let len = self.items.len();
        let mut current_index = i;

        loop {
            let left_index = Self::left_child(current_index);
            let right_index = Self::right_child(current_index);

            let swap_index = if left_index >= len && right_index >= len {
                return;
            }
            // no children, nothing to do
            else if left_index >= len {
                right_index
            } else if right_index >= len {
                left_index
            } else if self.items[left_index] < self.items[right_index] {
                left_index
            } else {
                right_index
            };

            self.items.swap(current_index, swap_index);
            current_index = swap_index;
        }
    }

    fn parent(i: usize) -> Option<usize> {
        match i {
            0 => None,
            i => Some((i - 1) / 2),
        }
    }

    fn left_child(i: usize) -> usize {
        2 * i + 1
    }

    fn right_child(i: usize) -> usize {
        2 * i + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_priority_queue() {
        let mut pq = PriorityQueue::new();

        pq.insert(4);
        pq.insert(8);
        pq.insert(5);
        pq.insert(9);
        pq.insert(0);
        pq.insert(12);
        pq.insert(10);

        assert_eq!(pq.pop(), Some(0));
        assert_eq!(pq.pop(), Some(4));
        assert_eq!(pq.pop(), Some(5));
        assert_eq!(pq.pop(), Some(8));
        assert_eq!(pq.pop(), Some(9));
        assert_eq!(pq.pop(), Some(10));
        assert_eq!(pq.pop(), Some(12));
    }
}
