from post_util.node import node
from post_util.segment import segment
from post_util.iterator import *
import math
import numpy as np
import copy

class ntree(object):

    def __init__(self, load_path=None):
        self.root = []
        self.nodedict = dict()
        if load_path != None:
            self.load(load_path)



    def __len__(self):
        return len(self.nodedict)

    def __iter__(self):
        self.root = self.get_somanodes()
        return walk(self.root)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return list(self.nodedict.values())[key]
        elif isinstance(key, int):
            return list(self.nodedict.values())[key]
        try:
            return self.nodedict[key]
        except KeyError:
            mindist = 1e6
            the_key = None
            for coord in list(self.nodedict.keys()):
                temp_dist = ((key[0] - coord[0]) ** 2 + (key[1] - coord[1]) ** 2 + (key[2] - coord[2]) ** 2) ** 0.5
                if temp_dist < mindist:
                    mindist = temp_dist
                    the_key = coord
            print('KeyError, return nodeobj at:', the_key)
            return self.nodedict[the_key]

    # def __copy__(self):
    #     return copy.copy(self)
    import numpy as np
    def rebuild_nodedict(self):
        """重新构建nodedict以同步节点的位置"""
        new_nodedict = {}
        for node in self.nodedict.values():
            # 使用round防止浮点数精度问题
            new_key = (round(node.x, 4), round(node.y, 4), round(node.z, 4))
            new_nodedict[new_key] = node
        self.nodedict = new_nodedict

    def smooth(self, angle_threshold=30.0, smoothing_factor=0.5):
        """
        Perform smoothing on the tree structure to reduce sharp angles between adjacent branches.

        :param angle_threshold: Threshold angle in degrees above which smoothing is applied.
        :param smoothing_factor: Factor to determine how much to move the node towards the average position.
        """
        nodes = list(self.nodedict.values())

        for i in range(1, len(nodes) - 1):
            prev_node = nodes[i - 1]
            current_node = nodes[i]
            next_node = nodes[i + 1]

            # Calculate direction vectors
            prev_vec = np.array(prev_node['coords']) - np.array(current_node['coords'])
            next_vec = np.array(next_node['coords']) - np.array(current_node['coords'])

            # Normalize direction vectors
            prev_vec /= np.linalg.norm(prev_vec)
            next_vec /= np.linalg.norm(next_vec)

            # Calculate the angle between the two direction vectors
            dot_product = np.dot(prev_vec, next_vec)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180.0 / np.pi)

            # If the angle is greater than the threshold, perform smoothing
            if angle > angle_threshold:
                # Calculate the average direction
                avg_vec = (prev_vec + next_vec) / 2.0
                avg_vec /= np.linalg.norm(avg_vec)

                # Adjust the current node position towards the average direction
                current_node['coords'] = np.array(current_node['coords']) + smoothing_factor * np.dot(avg_vec, np.array(
                    current_node['coords']) - np.array(prev_node['coords']))

    def smooth_bends(self, angle_threshold=45, smoothing_radius=1):
        def calculate_angle(p1, p2, p3):
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
            dot_product = np.dot(v1, v2)
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)
            cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 避免浮动误差
            return np.degrees(angle)

        nodes = list(self.nodedict.values())
        for i in range(1, len(nodes) - 1):
            prev_node = nodes[i - 1]
            curr_node = nodes[i]
            next_node = nodes[i + 1]

            angle = calculate_angle((prev_node.x, prev_node.y, prev_node.z),
                                    (curr_node.x, curr_node.y, curr_node.z),
                                    (next_node.x, next_node.y, next_node.z))

            if angle > angle_threshold:
                smoothed_position = np.mean([
                    [prev_node.x, prev_node.y, prev_node.z],
                    [curr_node.x, curr_node.y, curr_node.z],
                    [next_node.x, next_node.y, next_node.z]
                ], axis=0)
                curr_node.x, curr_node.y, curr_node.z = smoothed_position
        self.rebuild_nodedict()

    def load(self, load_path):
        # Load function implementation
        pass

    def get_somanodes(self):
        # Function to return root nodes
        pass

    def copy(self):
        return copy.copy(self)

    def add(self, x, y, z, ptype=None, radius=None, parent=None, child=None):
        new_node = node(x, y, z, ptype=ptype, radius=radius, parent=parent, child=child)
        self.nodedict[new_node.x, new_node.y, new_node.z] = new_node
        if parent!=None:
            new_node.connect(parent)
        else:
            self.root.append(new_node)
        return new_node

    def insert(self, x, y, z, ptype=None, radius=None, parent=None, child=None):
        new_node = node(x, y, z, ptype=ptype, radius=radius, parent=parent, child=child)
        self.nodedict[new_node.x, new_node.y, new_node.z] = new_node
        new_node.insert(parent=parent, child=child)
        return new_node

    def remove(self, nodeobj):
        self.nodedict.pop((nodeobj.x, nodeobj.y, nodeobj.z))
        nodeobj.remove()

    def load(self, swc_filename):
        with open(swc_filename) as f:
            nodeid_dict = {}
            lines = f.read().split("\n")
            for line in lines:
                if not line.startswith('#'):
                    cells = line.split(' ')
                    while '' in cells:
                        cells.remove('')
                    if len(cells) < 7:
                        continue
                    cells = [(float(c) if '.' in c else int(c)) for c in cells]
                    if (cells[2], cells[3], cells[4]) in self.nodedict:
                        nodeid_dict[cells[0]] = self.nodedict[cells[2], cells[3], cells[4]]
                    else:
                        nodeid_dict[cells[0]] = self.add(x=cells[2], y=cells[3], z=cells[4], ptype=cells[1],
                                                         radius=cells[5],
                                                         parent=nodeid_dict[cells[6]] if cells[6] != -1 else None)

    def safe_load(self, swc_filename):
        print(swc_filename)
        with open(swc_filename) as f:
            nodeid_dict = {}
            lines = f.read().split("\n")
            for line in lines:
                if not line.startswith('#'):
                    cells = line.split(' ')
                    while '' in cells:
                        cells.remove('')
                    if len(cells) < 7:
                        continue
                    cells = [(float(c) if '.' in c else int(c)) for c in cells]
                    if (cells[2], cells[3], cells[4]) in self.nodedict:
                        nodeid_dict[cells[0]] = self.nodedict[cells[2], cells[3], cells[4]]
                    else:
                        nodeid_dict[cells[0]] = self.add(x=cells[2], y=cells[3], z=cells[4], ptype=cells[1],
                                                         radius=cells[5], parent=None)
        with open(swc_filename) as f:
            lines = f.read().split("\n")
            for line in lines:
                if not line.startswith('#'):
                    cells = line.split(' ')
                    while '' in cells:
                        cells.remove('')
                    if len(cells) < 7:
                        continue
                    cells = [(float(c) if '.' in c else int(c)) for c in cells]
                    if (cells[2], cells[3], cells[4]) in self.nodedict:
                        now = self.nodedict[cells[2], cells[3], cells[4]]
                        parent = nodeid_dict[cells[6]] if cells[6] != -1 else None
                        if parent != None:
                            now.connect(parent)

    def save(self, swc_filename, seed=None):#mark

        if seed == None:
            soma = self.get_somanodes()
        else:
            soma = []
            for s in seed:
                while s.parent != None:
                    s = s.parent
                if s in soma:
                    print("false connection!")
                soma.append(s)
        # print(soma)
        seedlist = [[s, -1] for s in soma]
        i = 1
        inf = seedlist.pop(0)
        node = inf[0]
        parent_id = inf[1]

        with open(swc_filename, 'w') as f:
            while 1:
                if node==None:
                    inf = seedlist.pop()
                    node = inf[0]
                    parent_id = inf[1]
                    continue
                if node.radius == None:
                    node.radius = 1.0
                print('%d %d %.3f %.3f %.3f %.3f %d' %
                      tuple([i, 3, node.x, node.y, node.z, node.radius, parent_id]), file=f)
                if node.child == []:
                    if len(seedlist) == 0:
                        break
                    inf = seedlist.pop()
                    node = inf[0]
                    parent_id = inf[1]
                elif len(node.child) > 1:
                    for rest in node.child[1:]:
                        seedlist.append([rest, i])
                    node = node.child[0]
                    parent_id = i

                else:
                    node = node.child[0]
                    parent_id = i
                i += 1


    def get_somanodes(self):
        somanodes = []
        for node in self.nodedict.values():
            if node.parent == None:
                somanodes.append(node)
        return somanodes

    def bounding(self):
        x_min = list(self.nodedict.values())[0].x
        x_max = list(self.nodedict.values())[0].x
        y_min = list(self.nodedict.values())[0].y
        y_max = list(self.nodedict.values())[0].y
        z_min = list(self.nodedict.values())[0].z
        z_max = list(self.nodedict.values())[0].z
        for node in list(self.nodedict.values())[1:]:
            if node.x < x_min:
                x_min = node.x
            if node.x > x_max:
                x_max = node.x
            if node.y < y_min:
                y_min = node.y
            if node.y > y_max:
                y_max = node.y
            if node.z < z_min:
                z_min = node.z
            if node.z > z_max:
                z_max = node.z
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def crop(self, x_min, x_max, y_min, y_max, z_min, z_max, return_new_ntree=True):
        if return_new_ntree == True:
            new_ntree = ntree()
            for node in self:
                if node.x > x_min and node.x < x_max and node.y > y_min and node.y < y_max and node.z > z_min and node.z < z_max:
                    if node.parent != None and (node.parent.x, node.parent.y, node.parent.z) in new_ntree.nodedict:
                        new_ntree.add(node.x, node.y, node.z, ptype=node.ptype, radius=node.radius,
                                   parent=new_ntree.nodedict[node.parent.x, node.parent.y, node.parent.z])
                    else:
                        new_ntree.add(node.x, node.y, node.z, ptype=node.ptype, radius=node.radius)
            return new_ntree
        else:
            i = 0
            while i < len(self.nodelist):
                node = self.nodelist[i]
                if node.x < x_min or node.x > x_max or node.y < y_min or node.y > y_max or node.z < z_min or node.z > z_max:
                    self.remove(node)
                else:
                    i += 1

    def get_branchnodes(self):
        branch_nodes = []
        for node in self.nodedict.values():
            if len(node.child) > 1:
                branch_nodes.append(node)
        return branch_nodes

    def get_endnodes(self):
        endnodes = []
        for node in self.nodedict.values():
            if node.child == []:
                endnodes.append(node)
        return endnodes

    def get_endnode_with_orient(self):
        endnodes_with_orient = []
        for node in self.nodedict.values():
            if node.child == []:
                endnodes_with_orient.append(
                    [node, [node.x - node.parent.x, node.y - node.parent.y, node.z - node.parent.z]])
        return endnodes_with_orient

    def show(self):
        import matplotlib.pyplot as plt
        import cv2
        bound = self.bounding()
        print(bound)
        x_min = bound[0]
        y_min = bound[2]
        z_min = bound[4]
        swc_img_z = np.zeros([math.ceil(bound[3] - bound[2]), math.ceil(bound[1] - bound[0])])
        swc_img_y = np.zeros([math.ceil(bound[5] - bound[4]), math.ceil(bound[1] - bound[0])])
        swc_img_x = np.zeros([math.ceil(bound[5] - bound[4]), math.ceil(bound[3] - bound[2])])
        for node in self.nodedict.values():
            for c in node.child:
                cv2.line(swc_img_z, (int(round(node.x - x_min)), int(round(node.y - y_min))),
                         (int(round(c.x - x_min)), int(round(c.y - y_min))), 255, 2)
                cv2.line(swc_img_y, (int(round(node.x - x_min)), int(round(node.z - z_min))),
                         (int(round(c.x - x_min)), int(round(c.z - z_min))), 255, 2)
                cv2.line(swc_img_x, (int(round(node.y - y_min)), int(round(node.z - z_min))),
                         (int(round(c.y - y_min)), int(round(c.z - z_min))), 255, 2)
        plt.subplot(1, 2, 1)
        # plt.xticks(np.linspace(0, bound[1] - bound[0],5), np.linspace(bound[0],bound[1],5))
        # plt.yticks(np.linspace(0, bound[3] - bound[2],5), np.linspace(bound[2],bound[3],5))
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.imshow(swc_img_z)
        plt.subplot(2, 2, 2)
        plt.xlabel('X Axis')
        plt.ylabel('Z Axis')
        plt.imshow(swc_img_y)
        plt.subplot(2, 2, 4)
        plt.xlabel('Y Axis')
        plt.ylabel('Z Axis')
        plt.imshow(swc_img_x)
        plt.show()

    def tomask(self, r):
        import math
        r_ceil = int(math.ceil(r))
        biaslist = []
        for x in range(0, 2 * r_ceil + 1):
            for y in range(0, 2 * r_ceil + 1):
                for z in range(0, 2 * r_ceil + 1):
                    if (x - r_ceil)**2 + (y - r_ceil)**2 + (z - r_ceil)**2 <= r**2:
                        biaslist.append([x, y, z])
        bound = self.bounding()
        x_min = bound[0]
        y_min = bound[2]
        z_min = bound[4]
        swc_mask = np.zeros([math.ceil(bound[1] - bound[0] + 2 * r_ceil), math.ceil(bound[3] - bound[2] + 2 * r_ceil), math.ceil(bound[5] - bound[4] + 2 * r_ceil)], dtype=np.uint8)
        new_swc = self.simple_upsample(1)
        for node in new_swc.nodedict.values():
            swc_mask[int(round(node.x - x_min + r_ceil)), int(round(node.y - y_min + r_ceil)), int(round(node.z - z_min + r_ceil))] = 255
        return swc_mask

    def simple_downsample(self, thre_dist, return_newnt=False):
        if return_newnt == True:
            new_ntree = ntree()
            soma = self.get_somanodes()
            for s in soma:
                new_ntree.add(s.x, s.y, s.z, ptype=s.ptype, radius=s.radius, parent=None)
            seedlistwithdist = [[s, None, thre_dist] for s in soma]
            inf = seedlistwithdist.pop(0)
            node = inf[0]
            current = inf[1]
            dist = inf[2]
            while node != None:
                if dist >= thre_dist:
                    current = new_ntree.add(node.x, node.y, node.z, ptype=node.ptype, radius=node.radius, parent=current)
                    dist -= thre_dist
                if node.child == []:
                    if len(seedlistwithdist) == 0:
                        break
                    inf = seedlistwithdist.pop()
                    node = inf[0]
                    current = inf[1]
                    dist = inf[2]
                elif len(node.child) > 1:
                    for rest in node.child[1:]:
                        seedlistwithdist.append([rest, current, dist + e_dist(node, rest)])
                    node = node.child[0]
                    dist = dist + e_dist(node, node.parent)
                else:
                    node = node.child[0]
                    dist = dist + e_dist(node, node.parent)
            return new_ntree
        else:
            soma = self.get_somanodes()
            seedlistwithdist = [[s, thre_dist] for s in soma]
            inf = seedlistwithdist.pop(0)
            node = inf[0]
            dist = inf[1]
            while node != None:
                if dist >= thre_dist:
                    dist -= thre_dist
                else:
                    self.remove(node)
                if node.child == []:
                    if len(seedlistwithdist) == 0:
                        break
                    inf = seedlistwithdist.pop()
                    node = inf[0]
                    dist = inf[1]
                elif len(node.child) > 1:
                    for rest in node.child[1:]:
                        seedlistwithdist.append([rest, dist + e_dist(node, rest)])
                    dist = dist + e_dist(node, node.child[0])
                    node = node.child[0]
                else:
                    dist = dist + e_dist(node, node.child[0])
                    node = node.child[0]

    def simple_upsample(self, thre_dist):
        soma = self.get_somanodes()
        temp_inf = []
        for seed in soma:
            node = seed
            while node != None:  # node.child!=None
                if node.parent != None:
                    dist = e_dist(node.parent, node)
                    if dist > thre_dist:
                        num = math.ceil(dist / (thre_dist if thre_dist > 0 else 0.001))
                        dx = node.x - node.parent.x
                        dy = node.y - node.parent.y
                        dz = node.z - node.parent.z
                        temp = node.parent
                        for i in range(1, num):
                            temp = self.insert(temp.x + dx / num, temp.y + dy / num, temp.z + dz / num, parent=temp,
                                               child=node)

                if node.child == []:
                    if len(temp_inf) == 0:
                        break
                    node = temp_inf[-1]
                    del temp_inf[-1]
                    continue

                elif len(node.child) > 1:
                    for rest in node.child[1:]:
                        temp_inf.append(rest)
                node = node.child[0]

    def segment_remove(self, segment):
        temp = segment.end_node
        while temp != segment.start_node:
            tp = temp.parent
            self.remove(temp)
            temp = tp
        #self.remove(temp)

    def small_tree_delete(self,thres):
        somanodes = self.get_somanodes()
        # print(somanodes)
        for s in somanodes:
            l = 0
            for _ in walk(s):
                l += 1
            if l <= thres:
                dlist = []
                for n in walk(s):
                    dlist.append(n)
                if s not in dlist:
                    dlist.append(s)
                for n in dlist:
                    self.remove(n)

    def short_branch_prune(self, thres, treedeletethres=None):
        endnodes = self.get_endnodes()
        node_segment_dict = dict()
        for e in endnodes:

            temps = e

            continueflag = 0
            if temps.parent == None:
                continue

            if temps.parent.parent == None:
                continue

            temps = temps.parent
            lenth = 1

            while len(temps.child) == 1:
                if temps.parent.parent != None:
                    temps = temps.parent
                    lenth += 1
                else:
                    continueflag = 1
                    break

            if continueflag:
                continue

            tempseg = segment(start_node=temps, end_node=e)
            tempseg.len = lenth
            if tempseg.len < thres:
                if tempseg.start_node in node_segment_dict:
                    node_segment_dict[tempseg.start_node].append(tempseg)
                else:
                    node_segment_dict[tempseg.start_node] = [tempseg]

        for segs in node_segment_dict.values():
            if len(segs) == 1:
                self.segment_remove(segs[0])
            else:
                while segs != [] and segs[0].start_node.child != 1:
                    min_l = segs[0].len
                    min_s = segs[0]
                    for s in segs[1:]:
                        if s.len < min_l:
                            min_l = s.len
                            min_s = s
                    segs.remove(min_s)
                    self.segment_remove(min_s)
        if treedeletethres!=None:
            #treedeletethres = thres
            self.small_tree_delete(treedeletethres)

    def transform(self, x=None, y=None, z=None):
        x_transform = x.replace('x', 'node.x') if x != None else None
        y_transform = y.replace('y', 'node.y') if y != None else None
        z_transform = z.replace('z', 'node.z') if z != None else None
        key_list = []
        node_list = []
        for key, node in self.nodedict.items():
            key_list.append(key)
            node_list.append(node)
        for key, node in zip(key_list, node_list):
            if x_transform != None:
                exec('node.x=' + x_transform)
            if y_transform != None:
                exec('node.y=' + y_transform)
            if z_transform != None:
                exec('node.z=' + z_transform)
            self.nodedict[node.x, node.y, node.z] = self.nodedict.pop(key)

    def change_somanode(self, old_soma, new_soma):
        core_segment = segment(start_node=old_soma, end_node=new_soma)
        flag = 0
        dolist = []
        for n in core_segment:
            if flag == 0:
                n.child.append(n.parent)
                temp = n
                m = n
                flag = 1
            else:
                n.child.remove(temp)
                if n.parent!=None:
                    n.child.append(n.parent)
                dolist.append([n, temp])
                temp = n
        temp.parent = n
        m.parent = None
        for k in dolist:
            k[0].parent = k[1]
        if n.child == [None]:
            n.child = []

    def soma_fix(self, soma, r):
        for n in self:
            if (n.x - soma.x) ** 2 + (n.y - soma.y) ** 2 + (n.z - soma.z) ** 2 < r ** 2:
                if n!=soma:
                    self.remove(n)


    def numpy(self):
        pass

    def ana(self):
        somas = self.get_somanodes()


def e_dist(node1, node2):
    return ((node1.x-node2.x)**2+(node1.y-node2.y)**2+(node1.z-node2.z)**2)**0.5
