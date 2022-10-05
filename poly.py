import geometry
import numpy as np
import collections
from typing import *
from typing import OrderedDict


class vertex():

    def __init__(self, id: int, p: geometry.point):
        # self.p = geometry.point(0, 0)
        self.id = self.side = 0
        self.id = id
        self.sp: [shortest_path_tree,shortest_path_tree] = [None, None]
        self.head = None
        self.p = p
        # self.sp.append(shortest_path_tree(vertex(0,geometry.point(0,0)),vertex(0,geometry.point(0,0))))
        # self.sp.append(shortest_path_tree(vertex(0, geometry.point(0, 0)), vertex(0, geometry.point(0, 0))))
        self.side = -1

    def display_shortest_path(self, i: int):
        self.sp[i].sp_display(i)

    def vertex_display(self):
        print(self.id, end=" ")
        self.p.point_display()

    # def set_id(self, id: int):
    #     self.id = id

    def vertex_set_head(self, head):
        self.head = head

    def vertex_set_shortest_path_tree(self, i: int, sp1):

        self.sp[i] = sp1

        # self.sp.remove(i+1)
        # self.sp[i] = sp1

    def vertex_set_side(self, side):
        self.side = side

    def vertex_set_id(self, i):
        self.id = i

    def vertex_get_id(self):
        return self.id

    def vertex_get_point(self):
        return self.p

    def vertex_get_shortest_path_tree(self, i: int):
        return self.sp[i]

    def vertex_get_side(self):
        return self.side

    def vertex_get_head(self):
        return self.head

    def connect(self, that, f0):
        e0 = edge(self, that)
        e1 = edge(that, self)
        twin_edge(e0, e1)
        ei = self.vertex_get_head()
        if ei.edge_get_facet() == f0:
            pass
        else:
            ei = ei.edge_get_vertex_next()
            while ei != self.vertex_get_head():
                if ei.edge_get_facet() == f0:
                    break
                ei = ei.edge_get_vertex_next()
        ej = that.vertex_get_head()
        if ej.edge_get_facet() == f0:
            pass
        else:
            ej = ej.edge_get_vertex_next()
            while ej != that.vertex_get_head():
                if ej.edge_get_facet() == f0:
                    break
                ej = ej.edge_get_vertex_next()
        link_edge(ej.edge_get_prev(), e1)
        link_edge(ei.edge_get_prev(), e0)
        link_edge(e0, ej)
        link_edge(e1, ei)
        f1 = facet()
        if ei.edge_get_next().edge_get_to() == that:
            e0.set_facet(f0)
            e1.set_facet(f1)
            f0.facet_set_loop(e0)
            f1.facet_set_loop(e1)
            ek = ei
            while ek.edge_get_from() != that:
                ek.set_facet(f1)
                ek = ek.edge_get_next()
        else:
            e0.set_facet(f1)
            e1.set_facet(f0)
            f0.facet_set_loop(e1)
            f1.facet_set_loop(e0)
            ek = ej
            while ek.edge_get_from() != self:
                ek.set_facet(f1)
                ek = ek.edge_get_next()
        return e0

    def shortest_link(self, i: int, parent):
        if not parent:
            sp = shortest_path_tree(self, None)
        else:
            sp = shortest_path_tree(self, parent.vertex_get_shortest_path_tree(i))
        self.vertex_set_shortest_path_tree(i, sp)

    def vertex_left_convex(self, i: int):
        return self.sp[i].left_convex()

    def vertex_right_convex(self, i: int):
        return self.sp[i].right_convex()

    def vertex_get_edges(self):
        e = self.head
        result = [e]
        e = e.edge_get_vertex_next()
        while e != self.head:
            result.append(e)
            e = e.edge_get_vertex_next()
        return result

    def vertex_match_vertex(self, that):
        edges = self.vertex_get_edges()
        for i in range(len(edges)):
            e = edges[i]
            if e.edge_get_to() == that:
                return e
        return None

    def vertex_match_facet(self, f):
        edges = self.vertex_get_edges()
        for i in range(len(edges)):
            e = edges[i]
            if e.edge_get_facet() == f:
                return e
        return None


def get_vect(v0: vertex, v1: vertex):
    return geometry.vect(v0.vertex_get_point(), v1.vertex_get_point())


class edge(object):
    def __init__(self, from_vertex: vertex, to_vertex: vertex):
        # self.f: vertex
        # self.t: vertex
        self.twin: edge=None
        # self.prev = edge(vertex(0, geometry.point(0, 0)), vertex(0, geometry.point(0, 0)))
        # self.next = edge(vertex(0, geometry.point(0, 0)), vertex(0, geometry.point(0, 0)))
        self.prev: edge=None
        self.next: edge=None
        self.left: facet=None
        self.f:vertex = from_vertex
        self.t:vertex = to_vertex

    def edge_display(self):
        print("<", end="")
        self.f.vertex_display()
        self.t.vertex_display()
        print(">")

    def set_twin(self, e):
        self.twin = e

    def set_next(self, e):
        self.next = e

    def set_prev(self, e):
        self.prev = e

    def set_facet(self, f):
        self.left = f

    def edge_get_from(self) -> vertex:
        return self.f

    def edge_get_to(self) -> vertex:
        return self.t

    def edge_get_next(self):
        return self.next

    def edge_get_prev(self):
        return self.prev

    def edge_get_twin(self):
        return self.twin

    def edge_get_facet(self):
        return self.left

    def edge_get_vertex_next(self):
        return self.edge_get_prev().edge_get_twin()

    def edge_get_vertex_prev(self):
        return self.edge_get_twin().edge_get_next()

    def edge_get_side(self):
        return self.f.vertex_get_side() & self.t.vertex_get_side()

    def edge_disconnect(self):
        e1 = self.edge_get_twin()
        f0 = self.edge_get_facet()
        f1 = e1.edge_get_facet()
        if f0.facet_get_loop() == self:
            f0.facet_set_loop(self.edge_get_next())
        edges = f1.facet_get_edges()
        for i in range(len(edges)):
            edges[i].set_facet(f0)
        e1.edge_get_prev().set_next(self.edge_get_next())
        self.edge_get_next().set_prev(e1.edge_get_prev())
        e1.edge_get_next().set_prev(self.edge_get_prev())
        self.edge_get_prev().set_next(e1.edge_get_next())

    def edge_split_at(self, v: vertex):
        if self.f.vertex_get_point().on_single_point(v.vertex_get_point()):
            return self.f
        if self.t.vertex_get_point().on_single_point(v.vertex_get_point()):
            return self.t
        e1 = self
        e0 = self.twin
        e1_ = edge(v, e1.t)
        e0_ = edge(v, e0.t)
        e1.t = v
        e0.t = v
        v.vertex_set_head(e1_)
        twin_edge(e1, e0_)
        twin_edge(e0, e1_)
        link_edge(e1_, e1.next)
        link_edge(e1, e1_)
        link_edge(e0_, e0.next)
        link_edge(e0, e0_)
        e1_.left = e1.left
        e0_.left = e0.left
        return None

    def edge_interpolation_x(self, x):
        return geometry.interpolation_x(self.f.vertex_get_point(), self.t.vertex_get_point(), x)

    def edge_interpolation_y(self, y):
        return geometry.interpolation_y(self.f.vertex_get_point(), self.t.vertex_get_point(), y)

    def edge_get_vect(self):
        return geometry.vect(self.edge_get_from().vertex_get_point(), self.edge_get_to().vertex_get_point())


def twin_edge(e0: edge, e1: edge):
    e0.set_twin(e1)
    e1.set_twin(e0)


def link_edge(e0: edge, e1: edge):
    e0.set_next(e1)
    e1.set_prev(e0)


class chord():
    def __init__(self, *args):
        self.f: OrderedDict[vertex, edge]
        self.t: OrderedDict[vertex, edge]
        # print(type(self.f))
        self.f = collections.OrderedDict()
        # print(type(self.f))
        self.t = collections.OrderedDict()
        if len(args) == 1 and isinstance(args[0], edge):
            self.f[args[0].edge_get_from()] = args[0].edge_get_prev()
            self.t[args[0].edge_get_to()] = args[0].edge_get_next()
        if len(args) == 2 and isinstance(args[0], vertex) and isinstance(args[1], vertex):
            self.f[args[0]] = args[0].vertex_get_head()
            self.t[args[1]] = args[1].vertex_get_head()
        if len(args) == 4 and isinstance(args[0], vertex) and isinstance(args[1], vertex):
            self.f[args[0]] = args[2]
            self.t[args[1]] = args[3]
        if self.f and self.t:
            v0 = list(self.f.keys())[0]
            v1 = list(self.t.keys())[0]
            e0 = list(self.f.values())[0]
            e1 = list(self.t.values())[0]
            if v0.vertex_get_side() == -1:
                v0.vertex_set_side(e0.edge_get_side())
            if v1.vertex_get_side() == -1:
                v1.vertex_set_side(e1.edge_get_side())

    def get_from_dict(self):
        return self.f

    def get_to_dict(self):
        return self.t

    def get_from_vertex(self) -> vertex:
        if not self.f:
            return None
        else:
            return list(self.f.keys())[0]

    def get_to_vertex(self) -> vertex:
        if not self.t:
            return None
        else:
            return list(self.t.keys())[0]

    def get_from_edge(self) -> edge:
        if not self.f:
            return None
        else:
            return list(self.f.values())[0]

    def get_to_edge(self) -> edge:
        if not self.t:
            return None
        else:
            return list(self.t.values())[0]

    def chord_display(self):
        if self.get_from_edge():
            self.get_from_edge().edge_display()
        if self.get_to_edge():
            self.get_to_edge().edge_display()
        print("From: {}, To:{}".format(self.get_from_vertex().vertex_get_side(), self.get_to_vertex().vertex_get_side()))

    def intermediate(self) -> bool:
        return self.get_from_vertex().vertex_get_side() ^ self.get_to_vertex().vertex_get_side() == 3

    def get_first_facet(self):
        if self.get_from_edge():
            return self.get_from_edge().edge_get_twin().edge_get_facet()
        else:
            return self.get_to_edge().edge_get_twin().edge_get_facet()

    def reach(self):
        fx = self.get_from_vertex().vertex_get_point().get_point_x()
        tx = self.get_to_vertex().vertex_get_point().get_point_x()
        return fx if fx > tx else tx

    def chord_range(self):
        fx = self.get_from_vertex().vertex_get_point().get_point_x()
        tx = self.get_to_vertex().vertex_get_point().get_point_x()
        return abs(fx - tx)

    def chord_get_vect(self):
        return geometry.vect(self.get_to_vertex().vertex_get_point(), self.get_to_vertex().vertex_get_point())


class facet():
    def __init__(self):
        self.loop = edge(vertex(0, geometry.point(0, 0)), vertex(0, geometry.point(0, 0)))
        self.visited = False

    def facet_display(self):
        print("[", end="")
        edges = self.facet_get_edges()
        for i in range(np.size(edges)):
            e = edges[i]
            print(e.edge_get_from().vertex_get_id(), end=" ")
        print("]")

    def facet_set_loop(self, e: edge):
        self.loop = e

    def facet_get_loop(self):
        return self.loop

    def facet_set_visited(self):
        self.visited = True

    def facet_clean_visited(self):
        self.visited = False

    def facet_is_visited(self):
        return self.visited

    def facet_get_start_edge(self) -> edge:
        result = None
        e = self.loop
        p = e.edge_get_from().vertex_get_point()
        if not result or result.edge_get_from().vertex_get_point().cartesian_compare_to(p) == -1:
            result = e
        e = e.edge_get_next()
        while e != self.loop:
            p = e.edge_get_from().vertex_get_point()
            if not result or result.edge_get_from().vertex_get_point().cartesian_compare_to(p) == -1:
                result = e
            e = e.edge_get_next()
        return result

    def facet_get_end_edge(self) -> edge:
        result = None
        e = self.loop
        p = e.edge_get_from().vertex_get_point()
        if not result or result.edge_get_from().vertex_get_point().cartesian_compare_to(p) == 1:
            result = e
        e = e.edge_get_next()
        while e != self.loop:
            p = e.edge_get_from().vertex_get_point()
            if not result or result.edge_get_from().vertex_get_point().cartesian_compare_to(p) == 1:
                result = e
            e = e.edge_get_next()
        return result

    def facet_get_edges(self):
        e = self.loop
        result = [e]
        e = e.edge_get_next()
        while (e != self.loop):
            result.append(e)
            e = e.edge_get_next()
        return result

    def facet_area(self):
        result = 0.0
        origin = self.loop.edge_get_from()
        edges = self.facet_get_edges()
        for i in range(np.size(edges)):
            e = edges[i]
            v0 = get_vect(origin, e.edge_get_from())
            v1 = get_vect(origin, e.edge_get_to())
            c = v0.cross(v1) / 2
            result += c
        return result

    def facet_triangulate(self) -> None:
        start = self.facet_get_start_edge()
        end = self.facet_get_end_edge()
        queue = [[]]
        queue.append([3, start.edge_get_from()])
        upper = start
        lower = start.edge_get_prev()
        count = 0
        while upper.edge_get_to() != end.edge_get_from() and lower.edge_get_from() != end.edge_get_from():

            if upper.edge_get_to().vertex_get_point().cartesian_compare_to(
                    lower.edge_get_from().vertex_get_point()) == 1:
                queue.append([0, upper.edge_get_to()])
                upper = upper.edge_get_next()
            else:
                queue.append([1, lower.edge_get_from()])
                lower = lower.edge_get_prev()
        while upper.edge_get_to() != end.edge_get_from():
            queue.append([0, upper.edge_get_to()])
            upper = upper.edge_get_next()
        while lower.edge_get_from() != end.edge_get_from():
            queue.append([1, lower.edge_get_from()])
            lower = lower.edge_get_prev()
        queue.append([3, end.edge_get_from()])

        stack = [queue[0], queue[1]]

        # A = B = C = []
        for i in range(2, len(queue)):

            A = queue[i]

            if not A[0] == stack[-1][0]:
                for j in range(1, len(stack)):
                    B = stack[j]

                    A[1].connect(B[1], self)
                while stack:
                    B = stack.pop()
                stack.append(queue[i - 1])
                stack.append(A)
            else:
                B = stack.pop()
                while stack:
                    v0 = geometry.vect(stack[-1][1].vertex_get_point(), B[1].vertex_get_point())
                    v1 = geometry.vect(B[1].vertex_get_point(), A[1].vertex_get_point())
                    sc = v0.cross_sgn(v1)
                    if (A[0] == 0 and sc > 0) or (A[0] == 1 and sc < 0):
                        C = stack.pop()
                        A[1].connect(C[1], self)
                        B = C
                    else:
                        break
                stack.append(B)
                stack.append(A)
        A = queue[len(queue) - 1]
        stack.pop()
        for j in range(len(stack)):
            B = stack[j]
            A[1].connect(B[1], self)


class funnel():
    def __init__(self, apex, left, right):
        self.apex = None
        self.left = []
        self.right = []
        if apex:
            self.apex = apex
        if left:
            self.left.append(left)
        if right:
            self.right.append(right)

    def funnel_copy(self, that):
        self.apex = that.apex
        self.left = that.left
        self.right = that.right

    def funnel_display(self):
        print("[", end="")
        for i in range(np.size(self.left)):
            print(self.left[i].vertex_get_id(), end=" ")
        print("[ {} ]", format(self.apex.vertex_get_id()))
        for i in range(np.size(self.right)):
            print(self.right[i].vertex_get_id(), end=" ")
        print("]")

    def funnel_get_apex(self) -> vertex:
        return self.apex

    def funnel_get_left(self, i: int) -> vertex:
        if i == -1:
            return self.apex
        else:
            return self.left[i]

    def funnel_get_right(self, i: int) -> vertex:
        if i == -1:
            return self.apex
        else:
            return self.right[i]

    def funnel_get_left_head(self) -> vertex:
        if not self.left:
            return self.apex
        else:
            return self.left[0]

    def funnel_get_right_head(self) -> vertex:
        if not self.right:
            return self.apex
        else:
            return self.right[0]

    def funnel_size(self) -> int:
        return np.size(self.left) + np.size(self.right)

    def funnel_search(self, v: vertex) -> vertex:
        if not self.left:
            return self.apex
        if not self.right:
            return self.apex
        i = 1
        cni = -1
        for i in range(np.size(self.left) - 1, -1, -1):
            v0 = get_vect(self.funnel_get_left(i - 1), self.funnel_get_left(i))
            v1 = get_vect(self.funnel_get_left(i), v)
            sc = v0.cross_sgn(v1)
            if cni == -1 and sc >= 0:
                cni = i
            if sc > 0:
                break
        if cni >= 0:
            return self.funnel_get_left(cni)
        j = 0
        cnj = -1
        for j in range(np.size(self.right) - 1, -1, -1):
            v0 = get_vect(self.funnel_get_right(j - 1), self.funnel_get_right(j))
            v1 = get_vect(self.funnel_get_right(j), v)
            sc = v0.cross_sgn(v1)
            if cnj == -1 and sc <= 0:
                cnj = j
            if sc < 0:
                break
        if cnj >= 0:
            return self.funnel_get_right(cnj)
        else:
            return self.apex

    def funnel_split(self, index: int, v: vertex, that) -> str:
        if not self.left:
            v.shortest_link(index, self.apex)
            that.funnel_copy(funnel(self.apex, None, v))
            self.left.append(v)
            return "LEFT"
        if not self.right:
            v.shortest_link(index, self.apex)
            that.funnel_copy(funnel(self.apex, v, None))
            self.right.append(v)
            return "RIGHT"
        i = 0
        cni = -1
        for i in range(len(self.left) - 1, -1, -1):
            v0 = get_vect(self.funnel_get_left(i - 1), self.funnel_get_left(i))
            v1 = get_vect(self.funnel_get_left(i), v)
            sc = v0.cross_sgn(v1)
            if cni == -1 and sc >= 0:
                cni = i
            if sc > 0:
                break
        if cni >= 0:
            v.shortest_link(index, self.funnel_get_left(cni))
            that.funnel_copy(funnel(self.funnel_get_left(i), None, v))
            for k in range(i + 1, len(self.left)):
                that.left.append(self.funnel_get_left(k))
            for k in range(len(self.left) - 1, cni, -1):
                self.left.remove(k)
            self.left.append(v)
            return "LEFT"
        j = 0
        cnj = -1
        for j in range(len(self.right) - 1, -1, -1):
            v0 = get_vect(self.funnel_get_right(j - 1), self.funnel_get_right(j))
            v1 = get_vect(self.funnel_get_right(j), v)
            sc = v0.cross_sgn(v1)
            if cnj == -1 and sc <= 0:
                cnj = j
            if sc < 0:
                break
        if cnj >= 0:
            v.shortest_link(index, self.funnel_get_right(cnj))
            that.funnel_copy(funnel(self.funnel_get_right(j), v, None))
            for k in range(j + 1, len(self.right)):
                that.right.append(self.funnel_get_right(k))
            for k in range(len(self.right) - 1, cnj, -1):
                self.right.remove(k)
            self.right.append(v)
            return "RIGHT"
        else:
            v.shortest_link(index, self.apex)
            that.funnel_copy(funnel(self.funnel_get_right(j), v, None))
            for k in range(len(self.right)):
                that.right.append(self.funnel_get_right(k))
            for k in range(len(self.right) - 1, -1, -1):
                self.right.remove(k)
            self.right.append(v)
            return "RIGHT"


def structuralize(vertices: [vertex], external: facet):
    pass


def points_to_vertices(points: [geometry.point]) -> [vertex]:
    vertices = []
    for i in range(len(points)):
        p = points[i]
        v = vertex(i, p)
        vertices.append(v)

    return vertices


def series_to_vertices(s: geometry.series, eps):
    points = []
    for i in range(s.size):
        p = s.series_get(i)
        q = geometry.point(p.x, p.y)
        v = geometry.vect(0, -eps)
        points.append(q.point_add(v))
    for j in range(s.size - 1, -1, -1):
        p = s.series_get(j)
        q = geometry.point(p.x, p.y)
        v = geometry.vect(0, -eps)
        points.append(q.point_add(v))
    return points_to_vertices(points)


class polytope():
    def __init__(self, vertices: [vertex]):
        self.externel = facet()
        self.vertices = vertices
        for i in range(len(vertices)):
            v = self.vertices[i]
            if i == 0 or v.vertex_get_point().get_point_x() > self.vertices[i - 1].vertex_get_point().get_point_x():
                v.vertex_set_side(1)
            else:
                v.vertex_set_side(2)
        internal = facet()
        external = facet()
        last0 = None
        last1 = None
        for i in range(len(vertices)):
            f = self.vertices[i]
            if i == len(vertices) - 1:
                t = self.vertices[0]
            else:
                t = vertices[i + 1]
            e0 = edge(f, t)
            e1 = edge(t, f)
            e0.set_facet(internal)
            e1.set_facet(external)
            twin_edge(e0, e1)
            if i == 0:
                internal.facet_set_loop(e0)
                external.facet_set_loop(e1)
            else:
                link_edge(last0, e0)
                link_edge(e1, last1)
            f.vertex_set_head(e0)
            last0 = e0
            last1 = e1
        link_edge(last0, internal.facet_get_loop())
        link_edge(external.facet_get_loop(), last1)
        self.externel = external

    def polytope_display(self):
        print(len(self.vertices))
        self.externel.facet_display()
        for i in range(len(self.vertices)):
            self.vertices[i].vertex_display()

    def polytope_size(self):
        return len(self.vertices)

    def polytope_add_vertex(self, v: vertex) -> vertex:
        self.vertices.append(v)
        v.vertex_set_id(len(self.vertices) - 1)
        return v

    def polytope_get_vertex(self, i: int):
        if i < 0:
            return self.vertices[i + self.polytope_size()]
        else:
            return self.vertices[i]

    def polytope_is_boundary(self, e: edge) -> bool:
        return e.edge_get_facet() == self.externel or e.edge_get_twin().edge_get_facet() == self.externel

    def polytope_is_outside(self, e: edge) -> bool:
        return e.edge_get_facet() == self.externel

    def polytope_get_facets(self) -> list:
        queue = []
        self.externel.facet_set_visited()
        queue.append(self.externel)
        for i in range(len(queue)):
            f = queue[i]
            e = f.facet_get_loop()
            f0 = e.edge_get_twin().edge_get_facet()
            if not f0.facet_is_visited():
                f0.facet_set_visited()
                queue.append(f0)
            e = e.edge_get_next()
            while e != f.facet_get_loop():
                f0 = e.edge_get_twin().edge_get_facet()
                if not f0.facet_is_visited():
                    f0.facet_set_visited()
                    queue.append(f0)
                e = e.edge_get_next()
            for i in range(len(queue)):
                queue[i].facet_clean_visited()
            return queue

    def facet_check_triangulation(self):
        queue = self.polytope_get_facets()
        result = 0
        min = None
        max = None
        for i in range(len(queue)):
            f = queue[i]
            area = f.facet_area()
            result += area
            if f != self.externel:
                if min == None or area < min:
                    min = area
                if max == None or area > max:
                    max = area
        print("difference:{}, min:{}, max:{}, facets:{}", format(result, min, max, len(queue) - 1))

    def polytope_triangulate(self) -> None:
        facets = self.polytope_get_facets()
        # print(len(facets))
        for i in range(len(facets)):
            internal = facets[i]
            if internal != self.externel:
                internal.facet_triangulate()

    def polytope_intersect_edges(self, c: chord):
        result = []
        queue = []
        margin = []
        v0 = get_vect(c.get_from_vertex(), c.get_to_vertex())
        start = c.get_first_facet()
        start.facet_set_visited()
        queue.append(start)
        for i in range(len(queue)):
            f = queue[i]
            e = f.facet_get_loop()
            if not self.polytope_is_outside(e.edge_get_twin()):
                f0 = e.edge_get_twin().edge_get_facet()
                if not f0.facet_is_visited():
                    f0.facet_set_visited()
                    v1 = e.edge_get_vect()
                    if not v0.is_segment_intersect(v1):
                        margin.append(f0)
                    else:
                        result.append(e)
                        queue.append(f0)
            e = e.edge_get_next()
            while e != f.facet_get_loop():
                if not self.polytope_is_outside(e.edge_get_twin()):
                    f0 = e.edge_get_twin().edge_get_facet()
                    if not f0.facet_is_visited():
                        f0.facet_set_visited()
                        v1 = e.edge_get_vect()
                        if not v0.is_segment_intersect(v1):
                            margin.append(f0)
                        else:
                            result.append(e)
                            queue.append(f0)
                e = e.edge_get_next()
            for i in range(len(queue)):
                queue[i].facet_clean_visited()
            for j in range(len(margin)):
                margin[j].facet_clean_visited()
        return result


class polytube(polytope):
    def __init__(self, s, eps):
        super(polytube, self).__init__(series_to_vertices(s, eps))
        self.start = self.polytope_get_vertex(-1).vertex_get_head()
        self.end = self.polytope_get_vertex(int(self.polytope_size() / 2) - 1).vertex_get_head()
        self.end_chord: geometry.vect = None
        self.end_chord_: geometry.vect = None
        self.end_range: geometry.vect = None
        self.ratio = 0
        self.relative = False

    def polytube_get_start(self):
        return self.start

    def polytube_get_end(self):
        return self.end

    def polytube_add_chord(self, windows: [chord], ch: chord):
        if ch.get_from_vertex().vertex_get_point() != None and ch.get_to_vertex().vertex_get_point() != None:
            if ch.intermediate():
                windows.append(ch)

    def polytube_end_chords(self, type: str, p0: geometry.point, p1: geometry.point, b: vertex, t: geometry.vect,
                            v: geometry.vect, f: funnel, t0: vertex):
        self.end_range = geometry.vect(p0 if p0 else b.vertex_get_point(), p1 if p1 else b.vertex_get_point())
        if not self.ratio:
            self.end_chord = t
        else:
            er = self.end_range if self.relative else v
            y = er.vect_interpolation_y_ratio(self.ratio).get_point_y()
            if type == "LEFT":
                end_vertex = vertex(-1, geometry.point(p0.get_point_x(), y))
                self.end_chord = get_vect(f.funnel_search(end_vertex), end_vertex)
                if not end_vertex.vertex_get_point().on_vect(self.end_range):
                    self.end_chord_ = get_vect(t0, vertex(-1, p0))
            else:
                end_vertex = vertex(-1, geometry.point(p1.get_point_x(), y))
                self.end_chord = get_vect(f.funnel_search(end_vertex), end_vertex)
                if not end_vertex.vertex_get_point().on_vect(self.end_range):
                    self.end_chord_ = get_vect(t0, vertex(-1, p0))

    def bi_depth_first_search(self, e: edge, fn0: funnel, fn1: funnel) -> list:
        windows = []
        a = e.edge_get_from()
        b = e.edge_get_next().edge_get_to()
        c = e.edge_get_to()
        t00 = fn0.funnel_get_apex()
        t11 = fn1.funnel_get_apex()
        t10 = fn0.funnel_get_right_head()
        t01 = fn1.funnel_get_left_head()
        t0 = get_vect(t11, t01)
        t1 = get_vect(t00, t10)
        fn0_ = funnel(None, None, None)
        fn1_ = funnel(None, None, None)
        if fn0.funnel_split(0, b, fn0_) == "LEFT":
            temp = fn0
            fn0 = fn0_
            fn0_ = temp
        if fn1.funnel_split(1, b, fn1_) == "LEFT":
            temp = fn1
            fn1 = fn1_
            fn1_ = temp
        # print(a.vertex_left_convex(0))
        # print("b is")
        # b.vertex_display()
        # print(b.vertex_right_convex(1))
        count = 0
        if a.vertex_left_convex(0) and b.vertex_right_convex(1):
            ab = e.edge_get_prev().edge_get_twin()
            print("ab is")
            ab.edge_display()
            if not self.polytope_is_outside(ab):
                for item in self.bi_depth_first_search(ab, fn0, fn1):
                    print(count)
                    windows.append(item)
                # print("out side")
                # if self.bi_depth_first_search(ab,fn0,fn1)!=[]:
                #     print("get into")
                #     windows.append(self.bi_depth_first_search(ab,fn0,fn1))
            else:

                v0 = ab.edge_get_vect()
                p0 = v0.segment_line_intersect(t0)
                p1 = v0.segment_line_intersect(t1)
                self.polytube_add_chord(windows, chord(t01, vertex(-1, p0), None, ab))
                self.polytube_add_chord(windows, chord(vertex(-1, p1), t10, ab, None))
                if ab == self.polytube_get_end().edge_get_twin():
                    self.polytube_end_chords("LEFT", p0, p1, b, t0, v0, fn0, t01)
        if b.vertex_left_convex(0) and c.vertex_right_convex(1):
            bc = e.edge_get_next().edge_get_twin()
            if not self.polytope_is_outside(bc):
                for item in self.bi_depth_first_search(bc, fn0_, fn1_):
                    print("item is", format(item))
                    windows.append(item)
                # windows.append(self.bi_depth_first_search(bc,fn0_,fn1_))
            else:
                v1 = bc.edge_get_vect()
                p0 = v1.segment_line_intersect(t0)
                p1 = v1.segment_line_intersect(t1)
                self.polytube_add_chord(windows, chord(vertex(-1, p1), t10, bc, None))
                self.polytube_add_chord(windows, chord(t01, vertex(-1, p0), None, bc))
                if bc == self.polytube_get_end().edge_get_twin():
                    self.polytube_end_chords("RIGHT", p0, p1, b, t1, v1, fn0_, t10)
        return windows

    def polytube_get_windows_from_edge(self, e: edge):
        fn0 = funnel(e.edge_get_from(), None, e.edge_get_to())
        fn1 = funnel(e.edge_get_to(), e.edge_get_from(), None)
        e.edge_get_from().shortest_link(0, None)
        e.edge_get_to().shortest_link(1, None)
        e.edge_get_from().shortest_link(1, e.edge_get_to())
        e.edge_get_to().shortest_link(0, e.edge_get_from())
        return self.bi_depth_first_search(e, fn0, fn1)

    def polytope_sim_depth_first_search(self, e: edge, fn: funnel):
        windows = []
        a = e.edge_get_from()
        b = e.edge_get_next().edge_get_to()
        c = e.edge_get_to()
        t00 = fn.funnel_get_apex()
        # t11 = fn1.funnel_get_apex()
        t10 = fn.funnel_get_right_head()
        t01 = fn.funnel_get_left_head()
        t0 = get_vect(t00, t01)
        t1 = get_vect(t00, t10)
        fn_: funnel = None
        if fn.funnel_split(0, b, fn_) == "LEFT":
            temp = fn
            fn = fn_
            fn_ = temp
        if a.vertex_left_convex(0) and b.vertex_right_convex(0):
            ab = e.edge_get_prev().edge_get_twin()
            if not self.polytope_is_outside(ab):
                for item in self.polytope_sim_depth_first_search(ab, fn):
                    windows.append(item)
            else:
                v0 = ab.edge_get_vect()
                p0 = v0.segment_line_intersect(t0)
                p1 = v0.segment_line_intersect(t1)
                self.polytube_add_chord(windows, chord(t01, vertex(-1, p0), None, ab))
                self.polytube_add_chord(windows, chord(vertex(-1, p1), t10, ab, None))
                if ab == self.polytube_get_end().edge_get_twin():
                    self.polytube_end_chords("LEFT", p0, p1, b, t0, v0, fn, t01)
        if b.vertex_left_convex(0) and c.vertex_right_convex(0):
            bc = e.edge_get_next().edge_get_twin()
            if not self.polytope_is_outside(bc):
                for item in self.polytope_sim_depth_first_search(bc, fn_):
                    windows.append(item)
            else:
                v1 = bc.edge_get_vect()
                p0 = v1.segment_line_intersect(t0)
                p1 = v1.segment_line_intersect(t1)
                self.polytube_add_chord(windows, chord(vertex(-1, p1), t10, bc, None))
                self.polytube_add_chord(windows, chord(t01, vertex(-1, p0), None, bc))
                if bc == self.polytube_get_end().edge_get_twin():
                    self.polytube_end_chords("RIGHT", p0, p1, b, t1, v1, fn_, t10)
        return windows

    def polytube_get_windows_from_vertex(self, v: vertex) -> list:
        e = v.get_head()
        fn = funnel(v, None, e.edge_get_to())
        v.shortest_link(0, None)
        e.edge_get_to().shortest_link(0, v)
        return self.polytope_sim_depth_first_search(e, fn)

    def polytube_get_trace(self, windows: [chord]) -> [chord]:
        trace = []
        # print(windows)
        # iterator = None
        while len(windows) > 0:
            c: chord = None
            for j in range(len(windows)):
                if c == None or windows[j].reach() > c.reach():
                    # if c == None or (isinstance(c, chord) and windows[j].reach() > c.reach()):
                    c = windows[j]
            # print(c)
            trace.append(c)
            edges = self.polytope_intersect_edges(c)
            for j in range(len(edges)):
                edges[j].edge_disconnect()
            v0 = c.get_from_vertex()
            v1 = c.get_to_vertex()
            if c.get_from_edge():
                v0_ = c.get_from_edge().edge_split_at(v0)
                v0 = v0_ if v0_ else self.polytope_add_vertex(v0)
            else:
                v1_ = c.get_to_edge().edge_split_at(v1)
                v1 = v1_ if v1_ else self.polytope_add_vertex(v1)
            iterator = v0.connect(v1, c.get_first_facet())
            iterator.edge_get_facet().facet_triangulate()
            windows = self.polytube_get_windows_from_edge(iterator)
        return trace

    def polytube_get_points(self, trace: [chord]) -> [geometry.point]:
        points = []
        v0 = self.polytube_get_start().edge_get_vect()
        v1 = None
        for i in range(len(trace)):
            v1 = trace[i].chord_get_vect()
            points.append(v0.line_intersect(v1))
            v0 = v1
        if not self.end_chord_:
            p = v0.line_intersect(self.end_chord)
            if p:
                points.append(p)
            p = self.end_chord_.line_intersect(self.end_chord)
        else:
            p = v0.line_intersect(self.end_chord_)
            if p:
                points.append(p)
            p = self.end_chord_.line_intersect(self.end_chord)
            if p:
                points.append(p)
            points.append(self.end_chord.line_intersect(self.polytube_get_end().edge_get_vect()))
        return points

    def polytube_link_path(self):
        self.ratio = None
        start = self.polytube_get_start()
        self.polytope_triangulate()
        # print(self.polytube_get_windows_from_edge(start))
        return self.polytube_get_points(self.polytube_get_trace(self.polytube_get_windows_from_edge(start)))

    def polytube_link_path_with_parameters(self, y: geometry.class_range, rs, re, relative: bool) -> [geometry.point]:
        self.ratio = re
        self.relative = relative
        y0 = y.range_interpolation(rs)
        v = vertex(-1, self.polytube_get_start().edge_interpolation_y(y0))
        start = self.polytube_get_start().edge_split_at(v)
        if not start:
            start = self.polytope_add_vertex(v)
            start.vertex_set_side(3)
        self.polytope_triangulate()
        self.end_chord = self.end_chord_ = None
        points = self.polytube_get_points(self.polytube_get_trace(self.polytube_get_windows_from_edge(start)))
        y.range_set_x(self.end_range.get_from().get_point_y())
        y.range_set_y(self.end_range.get_to().get_point_y())
        return points


class shortest_path_tree():
    def __init__(self, v: vertex, parent1):
        self.vertex = v
        self.parent = parent1
        self.convex = 0
        if not parent1:
            self.depth = 0
            self.convex = 3
            self.cross = 0
            self.parent = None
        # elif isinstance(parent, shortest_path_tree):
        else:
            self.depth = parent1.get_depth() + 1
            grandparent = parent1.get_parent()
            if not grandparent:
                self.convex = 3
                self.cross = 0
                parent1.parent = None
            else:
                print("test here")
                v0 = get_vect(grandparent.sp_get_vertex(), parent1.sp_get_vertex())
                v1 = get_vect(parent1.sp_get_vertex(), v)
                # a = 3
                self.cross = v0.cross(v1)
                if v0.cross_sgn(v1) < 0:
                    self.convex = parent1.get_convex() & 1
                elif v0.cross_sgn(v1) == 0:
                    self.convex = parent1.get_convex() & 3
                elif v0.cross_sgn(v1) > 0:
                    self.convex = parent1.get_convex() & 2
                # self.convex = a

    def sp_display(self, index: int):
        print("(", end="")
        t = self
        while t:
            print(t.vertex.vertex_get_id(), end=" ")
            t = t.vertex.get_shortest_path_tree(index).get_parent()
        print(") [", end="")
        t = self
        while t:
            print(t.cross, end=" ")
            t = t.vertex.get_shortest_path_tree(index).get_parent()
        print("] [", end="")
        t = self
        while t:
            t.vertex.vertex_display()
            t = t.vertex.get_shortest_path_tree(index).get_parent()
        print("]", end="")

    def sp_get_vertex(self):
        return self.vertex

    def get_parent(self):
        return self.parent

    def get_convex(self):
        return self.convex

    def get_depth(self):
        return self.depth

    def left_convex(self) -> bool:
        return (self.convex & 2) > 0

    def right_convex(self) -> bool:
        return (self.convex & 1) > 0
