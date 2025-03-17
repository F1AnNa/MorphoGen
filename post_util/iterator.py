class walk(object):
    def __init__(self, root):
        self.root = [root] if type(root)!=list else root
        self.queue = self.root.copy()
        self.node = self.queue.pop(0)
        self.Stop_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.Stop_flag == True:
            raise StopIteration
        temp_node = self.node
        if self.node.child == []:
            if len(self.queue) == 0:
                self.Stop_flag = True
            else:
                self.node = self.queue.pop()
        elif len(self.node.child) > 1:
            for rest in self.node.child[1:]:
                self.queue.append(rest)
            self.node = self.node.child[0]
        else:
            self.node = self.node.child[0]
        return temp_node

class condition_walk(object):
    def __init__(self, root, condition, *condition_inf):
        self.root = [root] if type(root)!=list else root
        self.queue = self.root.copy()
        self.node = self.queue.pop(0)
        self.Stop_flag = False
        self.condition = condition
        self.condition_inf = condition_inf

    def __iter__(self):
        return self

    def __next__(self):
        if self.Stop_flag == True:
            raise StopIteration
        temp_node = self.node
        if self.node.child == [] or self.condition(temp_node, self.condition_inf):
            if len(self.queue) == 0:
                self.Stop_flag = True
            else:
                self.node = self.queue.pop()
        elif len(self.node.child) > 1:
            for rest in self.node.child[1:]:
                self.queue.append(rest)
            self.node = self.node.child[0]
        else:
            self.node = self.node.child[0]
        return temp_node

class back(object):
    def __init__(self, node):
        self.start = node
        self.node = node
        self.Stop_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.Stop_flag == True:
            raise StopIteration
        temp_node = self.node
        if self.node.parent == None:
            self.Stop_flag = True
        else:
            self.node = self.node.parent
        return temp_node

class segIterator(object):
    def __init__(self, segobj):
        self.seg = segobj
        self.reverse = self.seg.iter_reverse
        self.node = segobj.end_node if self.reverse==True else segobj.start_node
        self.generate_reverse_nodelist()
        if self.reverse==False:
            self.reverse_nodelist.reverse()
        self.i = 0

    def generate_reverse_nodelist(self):
        self.reverse_nodelist = []
        current = self.seg.end_node
        self.reverse_nodelist.append(current)
        while(current != self.seg.start_node):
            current = current.parent
            self.reverse_nodelist.append(current)
        self.length = len(self.reverse_nodelist)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.length:
            self.i += 1
            return self.reverse_nodelist[self.i - 1]
        else:
            raise StopIteration