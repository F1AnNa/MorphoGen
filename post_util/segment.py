import post_util.iterator

class segment(object):

    def __init__(self, start_node=None, end_node=None, iter_reverse=True):
        self.start_node = start_node
        self.end_node = end_node
        self.iter_reverse = iter_reverse
        self.branch_segment = []
        self.child_segment = []

    @property
    def lenth(self):
        return self.len_method()

    def __len__(self):
        self.len = self.len_method()
        return self.len

    def __iter__(self):
        return segIterator(self)

    def __reversed__(self):
        self.iter_reverse = 1 - self.iter_reverse
        iterator = segIterator(self)
        self.iter_reverse = 1 - self.iter_reverse
        return iterator

    def len_method(self):
        length = 0
        start = self.end_node
        while start != self.start_node:
            length += 1
            start = start.parent
        return length

    def check(self):
        self.child_segment = self.end_node.child
        current = self.end_node
        temp_child = current
        while current!=self.start_node:
            current = current.parent
            if len(current.child)>1:
                for n in current.child:
                    if n != temp_child:
                        self.branch_segment.append(n)
            temp_child = current
