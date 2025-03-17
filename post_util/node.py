class node(object):

    __slot__="x", "y", "z", "radius"

    def __init__(self, x, y, z, ptype=None, radius=1.0, parent=None, child=None):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.parent = parent
        if ptype != None:
            self.ptype = ptype
        elif parent == None:
            self.ptype = 1
        else:
            self.ptype = 3
        self.child = [] if child == None else child if type(child) == list else [child]#?

    def connect(self, parent):
        self.parent = parent
        parent.child.append(self)

    def insert(self, parent, child):
        if type(child) == list:
            for c in child:
                if c.parent != parent:
                    print("mismatch condition, do nothing!")
                    return 0
            for c in child:
                if c not in self.child:
                    self.child.append(c)
                parent.child.remove(c)
                c.parent = self
        else:
            if child.parent != parent:
                print("mismatch condition, do nothing!")
                return 0
            if child not in self.child:
                self.child.append(child)
            parent.child.remove(child)
            child.parent = self
        self.parent = parent
        parent.child.append(self)

    def remove(self):
        if self.parent != None:
            self.parent.child.remove(self)
            self.parent.child += self.child
        for c in self.child:
            c.parent = self.parent

    def cut(self):
        if self.parent != None:
            self.parent.child.remove(self)
        self.parent = None