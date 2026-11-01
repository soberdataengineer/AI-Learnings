"""An LRU cache built with a dictionary and a doubly linked list."""


class Node:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """Most recently used node is head.next; least recently used is tail.prev.

    get() raises KeyError on a miss, so values such as None remain cacheable.
    Intended for sequential use, such as processing one Spark partition.
    """

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")

        self.capacity = capacity
        self.nodes = {}  # key -> the existing node object
        self.head = Node()  # Dummy nodes hold no cached entries.
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _detach(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _add_to_front(self, node):
        first = self.head.next
        node.prev = self.head
        node.next = first
        self.head.next = node
        first.prev = node

    def get(self, key):
        node = self.nodes[key]  # Raises KeyError if not cached.
        self._detach(node)
        self._add_to_front(node)
        return node.value

    def put(self, key, value):
        if key in self.nodes:
            node = self.nodes[key]
            node.value = value
            self._detach(node)
            self._add_to_front(node)
            return

        node = Node(key, value)
        self.nodes[key] = node
        self._add_to_front(node)

        if len(self.nodes) > self.capacity:
            oldest = self.tail.prev
            self._detach(oldest)
            del self.nodes[oldest.key]

    def keys_mru_to_lru(self):
        """Return a snapshot for demonstration; this traversal takes O(n)."""
        keys = []
        node = self.head.next
        while node is not self.tail:
            keys.append(node.key)
            node = node.next
        return keys

    def __len__(self):
        return len(self.nodes)
