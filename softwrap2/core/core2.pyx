# distutils: language=c++
# cython: cdivision=True
# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False

# Copyright (C) 2021 Jean Da Costa machado.
# Jean3dimensional@gmail.com
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

import atexit
from libc.math cimport INFINITY
from libc.limits cimport UINT_MAX
from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport uintptr_t
from libc.string cimport memset
from cython.parallel cimport prange, parallel
cimport cython


cdef extern from *:
    '''
    template<typename T1, typename T2>
    CYTHON_INLINE T1 nmax(T1 a, T2 b) {return a > b ? a : (T1)b;}

    template<typename T1, typename T2>
    CYTHON_INLINE T1 nmin(T1 a, T2 b) {return a < b ? a : (T1)b;}

    template<typename T1>
    CYTHON_INLINE T1 nabs(T1 a) {return a < 0 ? -a : a;}

    template<typename T1, typename T2, typename T3>
    CYTHON_INLINE T1 nlerp(T1 a, T2 b, T3 t) {return a + (b - a) * t;}

    template<typename T1, typename T2, typename T3>
    CYTHON_INLINE T1 nclamp(T1 a, T2 min, T3 max) {return nmax(nmin(a, max), min);}


    '''

    T1 nmin[T1, T2](T1, T2) nogil
    T1 nmax[T1, T2](T1, T2) nogil
    T1 nabs[T1](T1) nogil
    T1 nlerp[T1, T2, T3](T1, T2, T3) nogil
    T1 nclamp[T1, T2, T3](T1, T2, T3) nogil


cdef extern from *:
    '''
    #define EPS 0.00001f
    struct vec3 {
        float v[3];

        vec3 (float a[3]) {
            v[0] = a[0];
            v[1] = a[1];
            v[2] = a[2];
        }

        vec3 (float x, float y, float z) {
            v[0] = x;
            v[1] = y;
            v[2] = z;
        }

        vec3 (float f) {
            v[0] = f;
            v[1] = f;
            v[2] = f;
        }

        vec3() = default;

        #define VEC_OP(op)\\
            CYTHON_INLINE vec3 operator op (vec3 other) {\\
                return vec3(v[0] op other.v[0], v[1] op other.v[1], v[2] op other.v[2]);\\
            }

        VEC_OP(+)
        VEC_OP(-)
        VEC_OP(*)
        VEC_OP(/)
        VEC_OP(+=)
        VEC_OP(-=)
        VEC_OP(*=)
        VEC_OP(/=)

        #undef VEC_OP
        #define VEC_OP(op)\\
            CYTHON_INLINE vec3 operator op (float other) {\\
                return vec3(v[0] op other, v[1] op other, v[2] op other);\\
            }

        VEC_OP(+)
        VEC_OP(-)
        VEC_OP(*)
        VEC_OP(/)
        VEC_OP(+=)
        VEC_OP(-=)
        VEC_OP(*=)
        VEC_OP(/=)
        VEC_OP(=)

        #undef VEC_OP
        CYTHON_INLINE float& x() { return v[0];}
        CYTHON_INLINE float& y() { return v[1];}
        CYTHON_INLINE float& z() { return v[2];}
        CYTHON_INLINE float sum() {return v[0] + v[1] + v[2];}
        CYTHON_INLINE float dot(vec3 other) {return v[0] * other.v[0] + v[1] * other.v[1] + v[2] * other.v[2];}
        CYTHON_INLINE float len() {return sqrtf(this->dot(*this));}
        CYTHON_INLINE float len_sqr() {return this->dot(*this);}
        CYTHON_INLINE void normalize() {
            float f = this->len();
            if (f == 0) {
                return;
            }
            f = 1 / f;
            *this *= f;
        }
        CYTHON_INLINE vec3 normalized() {
            float f = this->len();
            if (f == 0) {
                return *this;
            }
            f = 1 / f;
            return *this * f;
        }
        CYTHON_INLINE vec3 project(vec3 other) {return other * (this->dot(other) / other.dot(other));}
        CYTHON_INLINE vec3 project_unit(vec3 other) {return other * this->dot(other);}
        CYTHON_INLINE vec3 cross(vec3 other) {
            return vec3(v[1] * other.v[2] - v[2] * other.v[1],
                        v[2] * other.v[0] - v[0] * other.v[2],
                        v[0] * other.v[1] - v[1] * other.v[0]);
        }

        CYTHON_INLINE vec3 max_aa(vec3 other) {
            return vec3(nmax(v[0], other.v[0]), nmax(v[1], other.v[1]), nmax(v[2], other.v[2]));
        }
        CYTHON_INLINE vec3 min_aa(vec3 other) {
            return vec3(nmin(v[0], other.v[0]), nmin(v[1], other.v[1]), nmin(v[2], other.v[2]));
        }
        CYTHON_INLINE vec3 lerp(vec3 other, float fac) {
            return *this + (other - *this) * fac;
        }
        CYTHON_INLINE vec3 clamp(float min, float max) {
            float d = this->dot(*this);
            if (d == 0) {
                return *this;
            }
            else if (d < min * min) {
                return *this * (1 / sqrtf(d) * min);
            }
            else if (d > max * max) {
                return *this * (1 / sqrtf(d) * max);
            }
            return *this;
        }
    };
    '''
    float EPS
    float sqrtf(float) nogil
    cdef cppclass vec3 nogil:
        float v[3]
        vec3(float a[3]) nogil
        vec3(float, float, float) nogil
        vec3(float) nogil
        vec3() nogil
        vec3 operator + (vec3) nogil
        vec3 operator - (vec3) nogil
        vec3 operator * (vec3) nogil
        vec3 operator / (vec3) nogil
        vec3 operator + (float) nogil
        vec3 operator - (float) nogil
        vec3 operator * (float) nogil
        vec3 operator / (float) nogil
        vec3 operator = (float) nogil
        float & x() nogil
        float & y() nogil
        float & z() nogil
        float sum() nogil
        float dot(vec3) nogil
        float len() nogil
        float len_sqr() nogil
        void normalize() nogil
        vec3 normalized() nogil
        vec3 project(vec3) nogil
        vec3 project_unit(vec3) nogil
        vec3 cross(vec3) nogil
        vec3 max_aa(vec3) nogil
        vec3 min_aa(vec3) nogil
        vec3 lerp(vec3, float) nogil
        vec3 clamp(float, float) nogil

cdef extern from *:
    '''
    CYTHON_INLINE vec3 project_point_triangle(vec3& a, vec3& b, vec3& c, vec3& n, vec3& p){
        vec3 edge_v = b - a;
        vec3 pv = p - a;
        vec3 n1 = edge_v.cross(pv);

        if (n1.dot(n) < 0){
            return edge_v * nclamp(edge_v.dot(pv) / (edge_v.dot(edge_v) + EPS), 0, 1) + a;
        }

        edge_v = c - b;
        pv = p - b;
        n1 = edge_v.cross(pv);

        if (n1.dot(n) < 0){
            return  edge_v * nclamp(edge_v.dot(pv) / (edge_v.dot(edge_v) + EPS), 0, 1) + b;
        }

        edge_v = a - c;
        pv = p - c;
        n1 = edge_v.cross(pv);

        if (n1.dot(n) < 0){
            return edge_v * nclamp(edge_v.dot(pv) / (edge_v.dot(edge_v) + EPS), 0, 1) + c;
        }

        return p - pv.project_unit(n);
    }

    CYTHON_INLINE vec3 project_point_plane(vec3& p, vec3& pp, vec3& pn){
        vec3 v = (p - pp).project_unit(pn);
        return p - v;
    }
    '''
    vec3 project_point_triangle(vec3 & a, vec3 & b, vec3 & c, vec3 & n, vec3 & p) nogil
    vec3 project_point_plane(vec3& p, vec3& pp, vec3& pn) nogil

# ctypedef fused number:
#     char
#     signed char
#     unsigned char
#     int
#     signed int
#     unsigned int
#     short int
#     signed short int
#     unsigned short int
#     long int
#     signed long int
#     unsigned long int
#     float
#     double
#     long double


cdef dict not_freed_pointers = {}

cdef extern from *:
    ""
    int __LINE__

cdef void* maelloc(size_t size, int id) except NULL:
    cdef void* ptr = malloc(size)
    if ptr == NULL:
        raise MemoryError
    not_freed_pointers[<uintptr_t>ptr] = id
    return ptr

cdef void* caelloc(size_t n, size_t size, int id) except NULL:
    cdef void* ptr = calloc(n, size)
    if ptr == NULL:
        raise MemoryError
    not_freed_pointers[<uintptr_t>ptr] = id
    return ptr

cdef void fraeee(void* ptr):
    if ptr is not NULL:
        del not_freed_pointers[<uintptr_t> ptr]
    free(ptr)


def leak_check():
    import gc
    gc.collect()

    if not_freed_pointers:
        print('\n ------------ Oops Softwrap2 memory leak detected ------------ \n pointers not freed:')
        for ptr, line in not_freed_pointers.items():
            print(f'    ptr: {hex(ptr)}, core2.pyx:{line}')
        print('\n -------------------------------------------------------------')


atexit.register(leak_check)


cdef list vec3arr_to_list(vec3* arr, int size):
    cdef list l = [None] * size
    cdef int i
    for i in range(size):
        l[i] = (arr[i].v[0], arr[i].v[1], arr[i].v[2])

    return l


cdef inline unsigned int xorshift(unsigned int* seed) nogil:
    seed[0] ^= seed[0] << 11
    seed[0] ^= seed[0] >> 17
    seed[0] ^= seed[0] << 9
    return seed[0]

cdef inline float uint2neg1pos1f(unsigned int v) nogil:
    return <float> v / <float> ( <unsigned int> 0xffffffff >> 1) - <float>1.0

cdef float xorshiftf(unsigned int* seed) nogil:
    return uint2neg1pos1f(xorshift(seed))


cpdef index_check(int idx, int size):
    if idx < 0 or idx >= size:
        raise IndexError(idx)

@cython.final
cdef class Mesh:
    cdef:
        vec3* verts
        int[3]* triangles
        vec3* centroids
        vec3* vert_normals
        vec3* face_normals
        int n_verts
        int n_triangles

    def get_vert_normals(self):
        return vec3arr_to_list(self.vert_normals, self.n_verts)

    def get_face_normals(self):
        return vec3arr_to_list(self.face_normals, self.n_triangles)

    def __init__(self, verts, triangles):
        self.n_verts = len(verts)
        self.n_triangles = len(triangles)

        if len(verts) < 3 or len(triangles) < 1:
            raise ValueError('Invalid or empty mesh')

        self.verts = <vec3*>maelloc(sizeof(vec3) * self.n_verts, __LINE__)
        self.triangles = <int[3]*>maelloc(sizeof(int[3]) * self.n_triangles, __LINE__)

        cdef int i
        for i in range(self.n_verts):
            self.verts[i].v = verts[i]

        for i in range(self.n_triangles):
            self.triangles[i] = triangles[i]

    cdef inline vec3*tri_vert_ptr(self, int tri, int tri_i) nogil:
        return & self.verts[self.triangles[tri][tri_i]]

    cdef inline vec3 tri_vert(self, int tri, int tri_i) nogil:
        return self.verts[self.triangles[tri][tri_i]]

    cpdef void update_face_normals(self):
        if not self.face_normals:
            self.face_normals = <vec3*>caelloc(self.n_triangles, sizeof(vec3), __LINE__)

        cdef int i

        cdef vec3* a
        cdef vec3* b
        cdef vec3* c
        with nogil:
            for i in prange(self.n_triangles):
                a = self.tri_vert_ptr(i, 0)
                b = self.tri_vert_ptr(i, 1)
                c = self.tri_vert_ptr(i, 2)
                self.face_normals[i] = (b[0] - a[0]).cross(c[0] - a[0]).normalized()

    cpdef void update_vert_normals(self):
        if not self.vert_normals:
            self.vert_normals = <vec3*>caelloc(self.n_verts, sizeof(vec3), __LINE__)

        cdef int i, j

        cdef int * tri

        with nogil:
            for i in prange(self.n_verts):
                self.vert_normals[i] = vec3(0)

            for i in range(self.n_triangles):
                tri = <int*>self.triangles[i]
                for j in range(3):
                    self.vert_normals[tri[j]] += self.face_normals[i]

            for i in prange(self.n_verts):
                self.vert_normals[i].normalize()

    cpdef void update_centroids(self):
        if not self.centroids:
            self.centroids = <vec3*>maelloc(sizeof(vec3) * self.n_triangles, __LINE__)

        cdef int i
        for i in prange(self.n_triangles, nogil=True):
            self.centroids[i] = (self.tri_vert(i, 0) + self.tri_vert(i, 1) + self.tri_vert(i, 2)) / 3


    def closest_vert(self, pos):
        cdef vec3 p
        cdef float tmp_dist, dist = INFINITY
        cdef int i=0, idx = -1

        p.v = pos

        for i in range(self.n_verts):
            tmp_dist = (self.verts[i] - p).len_sqr()
            if tmp_dist < dist:
                dist = tmp_dist
                idx = i

        return idx, self.verts[idx].v


    def __dealloc__(self):
        fraeee(self.verts)
        fraeee(self.triangles)
        fraeee(self.centroids)
        fraeee(self.face_normals)
        fraeee(self.vert_normals)


cdef extern from *:
    '''
    #include <math.h>

    union BvhNode;
    struct BvhNodeBox;
    struct BvhNodeLeaf;

    struct BvhNodeBox {
        int split_axis;
        float split_pos;
        vec3 min, max;
        BvhNode* nodes[2];

        // BvhNodeBox() {
        //     split_axis = -1;
        //     min = vec3(INFINITY);
        //     max = vec3(-INFINITY);
        // }
        // CYTHON_INLINE void expand(BvhNodeBox other) {
        //     min = min.min_aa(other.min);
        //     max = max.max_aa(other.max);
        // }
        CYTHON_INLINE void expand(vec3 other) {
            min = min.min_aa(other);
            max = max.max_aa(other);
        }
        CYTHON_INLINE int major_axis(){
            int axis = 0;
            float curr_le, le = 0;

            for (int i=0; i<3; i++){
                curr_le = max.v[i] - min.v[i];
                if (curr_le > le){
                    axis = i;
                    le = curr_le;
                }
            }
            return axis;
        }
        CYTHON_INLINE vec3 box_center(){
            return (max + min) * 0.5;
        }
        CYTHON_INLINE float box_center_axis(int axis){
            return (max.v[axis] + min.v[axis]) * 0.5f;
        }
        CYTHON_INLINE float box_distance_sqr(vec3 p){
            return (p - max.min_aa(min.max_aa(p))).len_sqr();
        }
        CYTHON_INLINE void set_split_axis(int axis){
            this->split_axis = ~axis;
        }
        CYTHON_INLINE int get_split_axis(){
            return ~this->split_axis;
        }
        CYTHON_INLINE int is_leaf() {
            return split_axis >= 0;
        }
    };

    struct BvhNodeLeaf {
        int index;
        // BvhNodeLeaf (int _index) {
        //     index = _index;
        // }
    };

    union BvhNode{
        BvhNodeBox box;
        BvhNodeLeaf leaf;
        CYTHON_INLINE int is_leaf() {
            return leaf.index >= 0;
        }
    };

    struct BvhNearestResult {
        vec3 point;
        float distance;
        int tri_index;
    };
    '''

    cdef cppclass BvhNodeBox:
        int split_axis
        float split_pos
        vec3 min, max
        BvhNode * nodes[2]
        # BvhNodeBox() nogil
        void expand(BvhNodeBox) nogil
        void expand(vec3) nogil
        int major_axis() nogil
        vec3 box_center() nogil
        float box_center_axis(int) nogil
        int box_distance_sqr(vec3) nogil
        void set_split_axis(int) nogil
        int get_split_axis() nogil
        bint is_leaf() nogil

    cdef cppclass BvhNodeLeaf:
        int type, index
        # BvhNodeLeaf(int) nogil

    union BvhNode:
        BvhNodeBox box
        BvhNodeLeaf leaf
        bint is_leaf() nogil

    struct BvhNearestResult:
        vec3 point
        float distance
        int tri_index


# cdef struct BvhNearestResult:
#     vec3 point
#     int tri_index
#     float distance


@cython.final
cdef class BVH:
    cdef:
        BvhNodeLeaf* leaves
        BvhNodeBox* boxes
        BvhNode* root

        int free_box_idx
        int size
        Mesh mesh

    def __cinit__(self, Mesh mesh):
        self.mesh = mesh
        if mesh == None:
            raise ValueError

        self.size = mesh.n_triangles
        self.leaves = <BvhNodeLeaf*>maelloc(sizeof(BvhNodeLeaf) * self.size, __LINE__)
        self.boxes = <BvhNodeBox*>maelloc(sizeof(BvhNodeBox) * self.size, __LINE__)
        self.free_box_idx = self.size

        cdef int i
        with nogil:
            for i in prange(self.size):
                self.leaves[i].index = i
                self.boxes[i].split_axis = -1;
                self.boxes[i].min = vec3(INFINITY);
                self.boxes[i].max = vec3(-INFINITY);

        self.mesh.update_centroids()
        self.mesh.update_face_normals()
        self.root = self.build_tree(self.leaves, self.size)

    def __dealloc__(self):
        fraeee(self.leaves)
        fraeee(self.boxes)

    cdef inline BvhNodeBox* pop_box(self) nogil:
        self.free_box_idx -= 1
        # if self.free_box_idx < 0:
        #     raise RuntimeError('Ran out of boxes')

        return & self.boxes[self.free_box_idx]

    cdef BvhNode* build_tree(self, BvhNodeLeaf* leaves, int size) nogil:
        cdef int i, j

        if size == 1:
            return <BvhNode*>leaves

        cdef BvhNodeBox * box = self.pop_box()

        for i in range(size):
            for j in range(3):
                box.expand(self.mesh.tri_vert(leaves[i].index, j))

        cdef vec3 median = (box.max + box.min) * 0.5

        cdef int axis = box.major_axis()
        box.set_split_axis(axis)
        box.split_pos = median.v[axis]

        j = 0
        for i in range(size):
            if self.mesh.centroids[leaves[i].index].v[axis] < median.v[axis]:
                leaves[i].index, leaves[j].index = leaves[j].index, leaves[i].index
                j += 1

        if j == 0 or j == size:
            j = size // 2

        box.nodes[0] = self.build_tree(leaves, j)
        box.nodes[1] = self.build_tree(&leaves[j], size - j)
        return <BvhNode*>box

    def find_nearest(self, _p):
        cdef vec3 p
        p.v = _p
        cdef BvhNearestResult result = self._find_nearest(p)
        return (result.point.v, result.tri_index, result.distance)

    cdef inline BvhNearestResult _find_nearest(self, vec3 p) nogil:
        cdef BvhNearestResult result
        result.distance = INFINITY
        self.find_nearest_recursive( & p, & result, self.root)
        return result

    cdef void find_nearest_recursive(self, vec3* p, BvhNearestResult* out, BvhNode* node) nogil:
        cdef vec3 closest
        cdef int i, axis
        cdef float dist
        if node.is_leaf():
            i = (< BvhNodeLeaf*>node).index
            closest = project_point_triangle(self.mesh.tri_vert_ptr(i, 0)[0],
                                             self.mesh.tri_vert_ptr(i, 1)[0],
                                             self.mesh.tri_vert_ptr(i, 2)[0],
                                             self.mesh.face_normals[i],
                                             p[0])

            dist = (closest - p[0]).len_sqr()
            if dist < out.distance:
                out.point = closest
                out.tri_index = i
                out.distance = dist

        else:
            dist = node.box.box_distance_sqr(p[0])
            if dist < out.distance:
                # if not node.box.nodes[0].is_leaf():
                #     self.find_nearest_recursive(p, out, node.box.nodes[0])
                #     self.find_nearest_recursive(p, out, node.box.nodes[1])
                #     return
                #
                # if not node.box.nodes[1].is_leaf():
                #     self.find_nearest_recursive(p, out, node.box.nodes[1])
                #     self.find_nearest_recursive(p, out, node.box.nodes[0])
                #     return
                #
                # if (node.box.nodes[0].box.box_center() - p[0]).len_sqr() < (node.box.nodes[1].box.box_center() - p[0]).len_sqr():
                #     self.find_nearest_recursive(p, out, node.box.nodes[0])
                #     self.find_nearest_recursive(p, out, node.box.nodes[1])
                #     return
                axis = node.box.get_split_axis()
                if node.box.split_pos > p.v[axis]:
                    self.find_nearest_recursive(p, out, node.box.nodes[0])
                    self.find_nearest_recursive(p, out, node.box.nodes[1])
                    return
                self.find_nearest_recursive(p, out, node.box.nodes[1])
                self.find_nearest_recursive(p, out, node.box.nodes[0])




cdef extern from *:
    '''
    struct HalfLink {
        int a;
        float original_length;
        float scale;
    };

    struct HalfLinkArr {
        int n;
        HalfLink half_links[];
    };
    '''

    struct HalfLink:
        int a
        float original_length
        float scale


    struct HalfLinkArr:
        int n
        HalfLink half_links[0]


@cython.final
cdef class LinkProbe:
    cdef  readonly SpringLinks springs
    cdef HalfLinkArr* halflinks
    cdef readonly int index

    def __cinit__(self, SpringLinks springs, int index):
        self.springs = springs
        if index  < 0 or index >= springs.engine.n_verts:
            raise IndexError

        self.index = index
        self.halflinks = springs.links[index]

    def __getitem__(self, int link_index):
        if link_index < 0 or link_index >= self.halflinks.n:
            raise IndexError
        return self.halflinks.half_links[link_index].a

    def __len__(self):
        return self.halflinks.n

    def __iter__(self):
        cdef int i
        for i in range(self.halflinks.n):
            yield self.halflinks.half_links[i].a

    @property
    def avg_radius(self):
        cdef int i
        cdef float r = 0
        for i in range(self.halflinks.n):
            r += self.halflinks.half_links[i].original_length

        return r / (self.halflinks.n + EPS)

cdef class SpringLinks:

    cdef readonly SpringEngine engine
    cdef int max_links
    cdef HalfLinkArr** links

    @cython.boundscheck(True)
    def __cinit__(self, SpringEngine engine, list links):
        self.engine = engine
        self.links = <HalfLinkArr**>maelloc(sizeof(HalfLink**) * self.engine.n_verts, __LINE__)
        self.links[0] = <HalfLinkArr*>maelloc(sizeof(HalfLinkArr) * self.engine.n_verts + sizeof(HalfLink) * 2 * len(links), __LINE__)

        cdef int a, b, i, j, n

        cdef list arranged_links = [[] for i in range(engine.n_verts)]

        for a, b in links:
            arranged_links[a].append(b)
            arranged_links[b].append(a)

        cdef list half_lst

        for i in range(engine.n_verts):
            half_lst = arranged_links[i]
            n = len(half_lst)

            if i > 0:
                self.links[i] = <HalfLinkArr*>(<char*>(self.links[i - 1]) + sizeof(HalfLinkArr) + sizeof(HalfLink) * self.links[i - 1].n)

            self.links[i].n = n

            for j in range(n):
                a = half_lst[j]
                self.links[i].half_links[j].a = a

        self.lengths_update()

    def __dealloc__(self):
        fraeee(self.links[0])
        fraeee(self.links)

    def __getitem__(self, int key):
        if key < 0 or key >= self.engine.n_verts:
            raise IndexError

        return LinkProbe(self, key)

    cpdef void lengths_update(self):
        cdef int i, j, a

        for i in range(self.engine.n_verts):
            for j in range(self.links[i].n):
                a = self.links[i].half_links[j].a
                self.links[i].half_links[j].original_length = (self.engine.m.verts[i] - self.engine.m.verts[a]).len()
                self.links[i].half_links[j].scale = 1

    cpdef void smooth(self, float factor):
        cdef int i, j, n
        cdef HalfLink* half_links
        cdef vec3 avg

        with nogil:
            for i in prange(self.engine.n_verts):
                n = self.links[i].n

                half_links = self.links[i].half_links

                if n == 0:
                    avg = self.engine.m.verts[i]
                    continue

                avg = vec3(0)

                for j in range(n):
                    avg = avg + self.engine.m.verts[half_links[j].a]

                avg = avg * <float>1.0 / n

                self.engine.tmp_verts[i] = self.engine.m.verts[i].lerp(avg, factor)

        self.engine.m.verts, self.engine.tmp_verts = self.engine.tmp_verts, self.engine.m.verts


    cpdef void soft_spring_force(self, float factor, float deform_update=0.3, float deform_restore=0.03, float min_deform=0.3, float max_deform=3.0):
        cdef int i, j, n, a
        cdef HalfLink* half_links
        cdef vec3 delta
        cdef float curr_length, diff

        with nogil:
            for i in prange(self.engine.n_verts):
                n = self.links[i].n

                half_links = self.links[i].half_links

                self.engine.tmp_verts[i] = vec3(0)

                for j in range(n):
                    a = half_links[j].a

                    delta = self.engine.m.verts[i] - self.engine.m.verts[a]
                    curr_length = nmax(delta.len(), EPS)
                    diff = ((half_links[j].original_length * half_links[j].scale - curr_length) / curr_length)
                    self.engine.tmp_verts[i] += delta * diff
                    half_links[j].scale = nlerp(half_links[j].scale, nclamp(curr_length / half_links[j].original_length, min_deform, max_deform), deform_update)
                    half_links[j].scale = nlerp(half_links[j].scale, 1, deform_restore)

                self.engine.tmp_verts[i] *= <float>1.0 / (n + EPS) * factor
                self.engine.tmp_verts[i] += self.engine.m.verts[i]

        self.engine.m.verts, self.engine.tmp_verts = self.engine.tmp_verts, self.engine.m.verts

    cpdef void stiff_spring_force(self, float factor):
        cdef int i, j, n, a
        cdef HalfLink* half_links
        cdef vec3 delta
        cdef float curr_length, diff
        with nogil:
            for i in prange(self.engine.n_verts):
                n = self.links[i].n

                half_links = self.links[i].half_links

                self.engine.tmp_verts[i] = vec3(0)

                for j in range(n):
                    a = half_links[j].a

                    delta = self.engine.m.verts[i] - self.engine.m.verts[a]
                    curr_length = delta.len()

                    diff = (half_links[j].original_length * half_links[j].scale) / curr_length

                    diff = nlerp(<float>1.0, diff, factor)

                    delta = delta * diff
                    self.engine.tmp_verts[i] += delta + self.engine.m.verts[a]


                self.engine.tmp_verts[i] *= <float>1.0 / nmax(n, 1)

        self.engine.m.verts, self.engine.tmp_verts = self.engine.tmp_verts, self.engine.m.verts


cdef extern from *:
    '''
    struct TernaryLink {
        int a, b;
        int side;
        float avg_dist;
    };

    struct TernaryLinkArr {
        int n;
        TernaryLink arr[];
    };
    '''

    struct TernaryLink:
        int a, b
        int side
        float avg_dist

    struct TernaryLinkArr:
        int n
        TernaryLink arr[0]

@cython.final
cdef class TernarySmoothingLinks:
    cdef readonly SpringEngine engine
    cdef TernaryLinkArr** links

    @cython.boundscheck(True)
    def __cinit__(self, SpringEngine engine, list links):
        self.engine = engine

        self.links = <TernaryLinkArr**>maelloc(sizeof(TernaryLinkArr**) * engine.n_verts, __LINE__)
        self.links[0] = <TernaryLinkArr*>maelloc(sizeof(TernaryLinkArr) * engine.n_verts + sizeof(TernaryLink) * len(links), __LINE__)

        cdef list arranged_links = [[] for i in range(engine.n_verts)]

        cdef int c, a, b, i, j, n

        for c, a, b in links:
            arranged_links[c].append((a, b))

        cdef list links_lst
        cdef vec3 avg, d

        for i in range(engine.n_verts):
            links_lst = arranged_links[i]
            if i > 0:
                self.links[i] = <TernaryLinkArr*>((<char*>self.links[i - 1]) + sizeof(TernaryLinkArr) + sizeof(TernaryLink) * self.links[i - 1].n)

            n = len(links_lst)
            self.links[i].n = n

            for j in range(n):
                a, b = links_lst[j]
                self.links[i].arr[j].a = a
                self.links[i].arr[j].b = b


    def __dealloc__(self):
        fraeee(self.links[0])
        fraeee(self.links)

    cpdef void displacements_update(self):
        cdef int i, j, a, b
        cdef vec3 avg, d
        cdef TernaryLinkArr* arr

        self.engine.m.update_face_normals()
        self.engine.m.update_vert_normals()

        for i in range(self.engine.n_verts):
            arr = self.links[i]

            for j in range(arr.n):
                a = arr.arr[j].a
                b = arr.arr[j].b
                avg = (self.engine.m.verts[a] + self.engine.m.verts[b]) * 0.5
                d = (self.engine.m.verts[i] - avg)
                arr.arr[j].side = d.dot(self.engine.m.vert_normals[i]) > 0
                arr.arr[j].avg_dist = d.len() / ((self.engine.m.verts[a] - self.engine.m.verts[b]).len() + EPS)

    cpdef void displacement_force(self, float factor):
        cdef int i, j, a, b
        cdef vec3 avg, d
        cdef TernaryLinkArr* arr

        cdef float l

        with nogil:
            for i in prange(self.engine.n_verts):
                arr = self.links[i]
                if arr.n == 0:
                    continue

                self.engine.tmp_verts[i] = vec3(0)

                for j in range(arr.n):
                    a = arr.arr[j].a
                    b = arr.arr[j].b

                    avg = (self.engine.m.verts[a] + self.engine.m.verts[b]) * 0.5
                    d = (self.engine.m.verts[i] - avg)
                    if not (d.dot(self.engine.m.vert_normals[i]) > 0) == arr.arr[j].side:
                        d = d - d.project_unit(self.engine.m.vert_normals[i]) * 2
                    l = d.len()
                    if l > EPS:
                        l = 1 / l
                    d = d * l * arr.arr[j].avg_dist * (self.engine.m.verts[a] - self.engine.m.verts[b]).len()

                    self.engine.tmp_verts[i] += avg + d

                self.engine.tmp_verts[i] *= <float>1.0 / arr.n

            for i in prange(self.engine.n_verts):
                if self.links[i].n == 0:
                    continue

                self.engine.m.verts[i] = self.engine.m.verts[i].lerp(self.engine.tmp_verts[i], factor)


cdef struct QuaternaryLink:
    int a, b, c, d
    float ratio
    int side

cdef class QuaternarySmoothingLinks:
    cdef readonly SpringEngine engine
    cdef readonly int n_links
    cdef QuaternaryLink* links
    cdef int* accum_n

    @cython.boundscheck(True)
    def __cinit__(self, SpringEngine engine, list links):
        self.engine = engine
        self.n_links = len(links)
        self.links = <QuaternaryLink*>maelloc(sizeof(QuaternaryLink) * self.n_links, __LINE__)
        self.accum_n = <int*>caelloc(engine.n_verts, sizeof(int), __LINE__)

        cdef QuaternaryLink* lnk
        cdef int i

        for i in range(self.n_links):
            lnk = self.links + i
            lnk.a, lnk.b, lnk.c, lnk.d = links[i]
            index_check(lnk.a, engine.n_verts)
            index_check(lnk.b, engine.n_verts)
            index_check(lnk.c, engine.n_verts)
            index_check(lnk.d, engine.n_verts)
            self.accum_n[lnk.a] += 1
            self.accum_n[lnk.b] += 1
            self.accum_n[lnk.c] += 1
            self.accum_n[lnk.d] += 1

        for i in range(engine.n_verts):
            if self.accum_n[i] == 0:
                self.accum_n[i] = 1

        self.lengths_update()

    def __dealloc__(self):
        fraeee(self.links)
        fraeee(self.accum_n)

    cpdef void lengths_update(self):

        cdef int i
        cdef QuaternaryLink* lnk
        cdef vec3 ab, cd

        for i in range(self.n_links):
            lnk = self.links + i
            ab = (self.engine.m.verts[lnk.a] - self.engine.m.verts[lnk.b])
            cd = (self.engine.m.verts[lnk.c] - self.engine.m.verts[lnk.d])

            lnk.ratio = ab.len() / cd.len()
            lnk.side = ab.dot(cd) > 1

    cpdef void smooth(self, float factor, float max_ratio=3):
        cdef int i
        cdef QuaternaryLink* lnk
        cdef vec3 ab, cd
        cdef float lab, lcd, rlab, rlcd
        cdef float min_ratio = <float>1.0 / max_ratio

        for i in range(self.engine.n_verts):
            self.engine.tmp_verts[i] = vec3(0)

        for i in range(self.n_links):
            lnk = self.links + i
            if lnk.ratio < min_ratio or lnk.ratio > max_ratio:
                continue

            ab = (self.engine.m.verts[lnk.a] - self.engine.m.verts[lnk.b])
            cd = (self.engine.m.verts[lnk.c] - self.engine.m.verts[lnk.d])


            rlab = ab.len()
            rlcd = cd.len()

            lab = rlcd * lnk.ratio
            lcd = rlab / lnk.ratio

            ab -= ab / rlab * lab
            cd -= cd / rlcd * lcd

            self.engine.tmp_verts[lnk.a] -= ab
            self.engine.tmp_verts[lnk.b] += ab

            self.engine.tmp_verts[lnk.c] -= cd
            self.engine.tmp_verts[lnk.d] += cd

        for i in range(self.engine.n_verts):
            self.engine.m.verts[i] += self.engine.tmp_verts[i] * factor / self.accum_n[i]


@cython.final
cdef class SpringEnginePin:
    cdef readonly SpringEngine engine
    cdef readonly int n_rings
    cdef readonly int start_index
    cdef int** rings


    def __cinit__(self, SpringLinks links, int start_index, int n_rings):
        n_rings = nmax(n_rings, 1)
        self.engine = links.engine

        if start_index < 0 or start_index >= self.engine.n_verts:
            raise ValueError('invalid start_index')

        self.start_index = start_index

        cdef list rings = [[start_index]]
        cdef set seen = set((start_index,))
        cdef list new_front
        cdef LinkProbe probe

        cdef int i
        cdef int vert_idx
        for i in range(n_rings - 1):
            new_front = []
            for vert_idx in rings[-1]:
                probe = links[vert_idx]
                for new_idx in probe:
                    if new_idx not in seen:
                        seen.add(new_idx)
                        new_front.append(new_idx)
            rings.append(new_front)

        self.n_rings = n_rings
        self.rings = <int**>maelloc(sizeof(int**) * self.n_rings, __LINE__)

        for i in range(self.n_rings):
            self.rings[i] = NULL

        cdef list front
        cdef int j

        for i, front in enumerate(rings):
            self.rings[i] = <int*>maelloc(sizeof(int*) * (len(front) + 1), __LINE__)
            self.rings[i][0] = len(front)

            for j in range(self.rings[i][0]):
                self.rings[i][j + 1] = front[j]

    def __getitem__(self, int index):
        if index < 0 or index >= self.n_rings:
            raise IndexError

        def iter_ring():
            cdef int i
            for i in range(self.rings[index][0]):
                yield self.rings[index][i + 1]

        return iter_ring()

    def __iter__(self):
        cdef int i
        for i in range(self.n_rings):
            yield self[i]

    def move(self, float x, float y, float z, float scale):
        cdef vec3 vec = vec3(x, y, z)
        cdef vec3 ring_vec

        cdef int i, j
        for i in range(self.n_rings):
            ring_vec = vec * nclamp((scale - i - <float>1) / (scale - <float>1), <float>0, <float>1)
            for j in range(self.rings[i][0]):
                self.engine.m.verts[self.rings[i][j + 1]] += ring_vec
                # self.engine.prev_verts[self.rings[i][j + 1]] += ring_vec


    def __dealloc__(self):
        cdef int i
        for i in range(self.n_rings):
            fraeee(self.rings[i])
        fraeee(self.rings)


cdef class _MaskContextManager:
    cdef SpringEngineMask mask
    cdef float factor
    cdef bint invert

    def __cinit__(self, SpringEngineMask mask, float factor, bint invert):
        self.mask = mask
        self.factor = factor
        self.invert = invert

    def __enter__(self):
        self.mask.load()

    def __exit__(self, type, value, traceback):
        self.mask.masked_store(self.factor, self.invert)



cdef class SpringEngineMask:
    cdef float* mask
    cdef vec3* vec_data

    cdef SpringEngine engine

    def __cinit__(self, SpringEngine engine, list mask):
        self.engine = engine

        if not mask:
            self.mask = NULL
            self.vec_data = NULL
            return

        elif not len(mask) == engine.n_verts:
            raise ValueError('mask must have the same number of elements as engine verts')

        self.mask = <float*>maelloc(sizeof(float) * self.engine.n_verts, __LINE__)
        self.vec_data = <vec3*>maelloc(sizeof(vec3) * self.engine.n_verts, __LINE__)

        cdef int i
        for i in range(self.engine.n_verts):
            self.mask[i] = mask[i]

    def __getitem__(self, int index):
        if self.mask == NULL:
            return 0.0

        index_check(index, self.engine.n_verts)
        return self.mask[index]

    def __setitem__(self, int index, value):
        if self.mask == NULL:
            return

        index_check(index, self.engine.n_verts)
        self.mask[index] = value

    cpdef set_force(self, float x, float y, float z):
        if self.mask == NULL:
            return

        cdef int i
        cdef vec3 d, v
        v = vec3(x, y, z)
        with nogil:
            for i in prange(self.engine.n_verts):
                d = self.engine.m.verts[i] - self.engine.prev_verts[i]
                d = d.lerp(v, self.mask[i])
                self.engine.m.verts[i] = self.engine.prev_verts[i] + d

    cpdef void load(self):
        if self.mask == NULL:
            return

        cdef int i
        with nogil:
            for i in prange(self.engine.n_verts):
                self.vec_data[i] = self.engine.m.verts[i]

    cpdef void store(self):
        if self.mask == NULL:
            return

        cdef int i
        with nogil:
            for i in prange(self.engine.n_verts):
                self.engine.m.verts[i] = self.vec_data[i]

    cpdef void masked_store(self, float factor=1, bint invert=False):
        if self.mask == NULL:
            return

        cdef int i
        with nogil:
            if invert:
                for i in prange(self.engine.n_verts):
                    self.engine.m.verts[i] = self.engine.m.verts[i].lerp(self.vec_data[i], (<float>1.0 - self.mask[i]) * factor)
            else:
                for i in prange(self.engine.n_verts):
                    self.engine.m.verts[i] = self.engine.m.verts[i].lerp(self.vec_data[i], self.mask[i] * factor)

    cpdef set(self, list vec_data):
        if self.mask == NULL:
            return

        cdef int i
        if not len(vec_data) == self.engine.n_verts * 3:
            raise ValueError('vec_data must have the same number of elements as engine verts times 3')

        for i in range(self.engine.n_verts * 3):
            self.vec_data[i // 3].v[i % 3] = vec_data[i]

    def masked_context(self, float factor=1, bint invert=False):
        return _MaskContextManager(self, factor, invert)

    def __dealloc__(self):
        fraeee(self.vec_data)
        fraeee(self.mask)



@cython.final
cdef class SymmetryMap:
    cdef int* symm_map
    cdef SpringEngine engine
    cdef readonly float[3] error

    def __cinit__(self, SpringEngine engine):
        if engine == None:
            raise ValueError
        cdef BVH bvh = BVH(engine.m)
        self.engine = engine
        self.symm_map = <int*>maelloc(sizeof(int) * engine.n_verts * 3, __LINE__)

        cdef int i, j, axis, v_index
        cdef BvhNearestResult result
        cdef vec3 v
        cdef double dist, tmp_dist

        self.error = (0, 0, 0)

        with nogil:
            for i in prange(self.engine.n_verts):
                for axis in range(3):
                    dist = INFINITY
                    v = self.engine.m.verts[i]
                    v.v[axis] = -v.v[axis]

                    result = bvh._find_nearest(v)
                    for j in range(3):
                        v_index = self.engine.m.triangles[result.tri_index][j]
                        tmp_dist = (self.engine.m.verts[v_index] - v).len()
                        if tmp_dist < dist:
                            dist = tmp_dist
                            self.symm_map[i + self.engine.m.n_verts * axis] = v_index
                    self.error[axis] = nmax(dist, self.error[axis])

    def __getitem__(self, int index):
        if index < 0 or index >= self.engine.n_verts:
            raise KeyError

        return tuple(self.symm_map[index + axis * self.engine.n_verts] for axis in range(3))

    cpdef void mirror(self, bint x, bint y, bint z):
        cdef int i, j, axis, v_index
        cdef int[3] mirror = (x, y, z)
        cdef vec3 v

        with nogil:
            for axis in range(3):
                if not mirror[axis]:
                    continue

                for i in prange(self.engine.n_verts):
                    v_index = self.symm_map[i + axis * self.engine.m.n_verts]
                    v = self.engine.m.verts[v_index]
                    v.v[axis] = -v.v[axis]
                    self.engine.tmp_verts[i] = (self.engine.m.verts[i] + v) * 0.5

                self.engine.tmp_verts, self.engine.m.verts = self.engine.m.verts, self.engine.tmp_verts

    def __dealloc__(self):
        fraeee(self.symm_map)


cdef class SpringEngine:
    cdef Mesh m
    cdef vec3* prev_verts
    cdef vec3* tmp_verts
    cdef BVH bvh
    cdef int* bvh_closest_indexes
    cdef unsigned int snap_count

    cdef int n_verts

    def __init__(self, Mesh m, BVH bvh=None):
        self.m = m
        self.n_verts = m.n_verts
        self.snap_count = 0
        self.prev_verts = <vec3*>maelloc(sizeof(vec3) * self.n_verts, __LINE__)
        self.tmp_verts = <vec3*>maelloc(sizeof(vec3) * self.n_verts, __LINE__)
        self.bvh_closest_indexes = <int*>maelloc(sizeof(vec3) * self.n_verts, __LINE__)
        cdef int i
        for i in range(self.n_verts):
            self.prev_verts[i] = self.m.verts[i]

        self.set_bvh(bvh)

    cpdef void set_bvh(self, BVH bvh):
        self.bvh = bvh

        cdef int i
        if bvh:
            with nogil:
                for i in prange(self.n_verts):
                    self.bvh_closest_indexes[i] = -1

    def __dealloc__(self):
        fraeee(self.prev_verts)
        fraeee(self.tmp_verts)
        fraeee(self.bvh_closest_indexes)

    def __len__(self):
        return self.n_verts * 3

    def __getitem__(self, int key):
        return self.m.verts[key // 3].v[key % 3]

    def __setitem__(self, int key, float v):
        self.m.verts[key // 3].v[key % 3] = v
        self.prev_verts[key // 3].v[key % 3] = v

    def from_list(self, list items):
        if not len(items) >= self.n_verts * 3:
            raise ValueError('list too small')

        cdef int i
        for i in range(self.n_verts * 3):
            self.m.verts[i // 3].v[i % 3] = items[i]
            self.prev_verts[i // 3].v[i % 3] = items[i]

    def set_verts(self, list indexes, list locations):
        cdef int i, index
        cdef float x, y, z
        if not len(indexes) == len(locations):
            raise ValueError('indexes array is not the same size as locations array')

        for i in range(len(indexes)):
            index = indexes[i]
            x, y, z = locations[i]
            self.prev_verts[index] = self.m.verts[index] = vec3(x, y, z)

    def get_verts(self, list indexes):
        cdef int i, index
        cdef list ret = []
        for i in range(len(indexes)):
            index = indexes[i]
            if index < 0 or index >= self.n_verts:
                raise IndexError
            ret.append(self.m.verts[index].v)
        return ret

    def move_verts(self, list indexes, float x, float y, float z):
        cdef vec3 delta = vec3(x, y, z)
        cdef int i, index
        for i in range(len(indexes)):
            index = indexes[i]
            if index < 0 or index >= self.n_verts:
                raise IndexError
            self.m.verts[index] += delta
            self.prev_verts[index] += delta

    def snap_to_bvh(self, float factor=1, int cycle_quality=10, str snapping_mode='SURFACE'):
        cdef int snapping_mode_ = {'SURFACE': 1,
                                   'OUTSIDE': 2,
                                   'INSIDE':  4}[snapping_mode]

        if not self.bvh:
            raise RuntimeError('No bvh avaliable')

        cycle_quality = nabs(cycle_quality)
        self.snap_count += 1

        cdef BvhNearestResult result
        cdef int i
        cdef vec3 v
        cdef float snapping
        cdef unsigned int cycle

        with nogil:
            for i in prange(self.n_verts):
                cycle = (<unsigned int>i ^ <unsigned int>0x243F6A88) * <unsigned int>0x243F6A88
                cycle = cycle ^ cycle >> 5
                cycle = cycle + self.snap_count

                if cycle % cycle_quality > 0 and self.bvh_closest_indexes[i] >= 0:
                    result.tri_index = self.bvh_closest_indexes[i]
                    result.point = project_point_plane(
                        self.m.verts[i],
                        self.bvh.mesh.verts[self.bvh.mesh.triangles[result.tri_index][0]],
                        self.bvh.mesh.face_normals[result.tri_index]
                    )
                else:
                    result = self.bvh._find_nearest(self.m.verts[i])
                    self.bvh_closest_indexes[i] = result.tri_index

                v = result.point - self.m.verts[i]

                if snapping_mode_ & 1:
                    if v.dot(self.bvh.mesh.face_normals[result.tri_index]) > 0:
                        if v.dot(self.m.vert_normals[i]) < 0:
                            v = self.m.vert_normals[i] * v.len() + v * 0.5


                    snapping = self.m.vert_normals[i].dot(self.bvh.mesh.face_normals[result.tri_index])
                    snapping = snapping * snapping

                    v = v * factor * snapping
                    self.m.verts[i] += v

                elif snapping_mode_ & (2 | 4):
                    if (v.dot(self.bvh.mesh.face_normals[result.tri_index]) > 0) ^ (snapping_mode_ & 4 > 0):
                        v = self.m.verts[i] = result.point

    cpdef SpringLinks create_spring_group(self, list links):
        return SpringLinks(self, links)

    cpdef SymmetryMap create_symmetry_map(self):
        return SymmetryMap(self)

    cpdef TernarySmoothingLinks create_ternary_links(self, list links):
        return TernarySmoothingLinks(self, links)

    cpdef QuaternarySmoothingLinks create_quaternary_links(self, list links):
        return QuaternarySmoothingLinks(self, links)

    cpdef SpringEngineMask create_mask(self, mask):
        return SpringEngineMask(self, mask)

    cpdef void random_verts(self, float factor, unsigned int seed=0x452821E6):
        cdef int i
        cdef unsigned int xorstate = 0x452821E6
        cdef vec3 v
        with nogil:
            for i in prange(self.n_verts):
                v = vec3(xorshiftf(&xorstate), xorshiftf(&xorstate), xorshiftf(&xorstate)) * factor
                self.m.verts[i] += v
                self.prev_verts[i] += v

    cpdef void kinetic_step(self, float damping):
        cdef int i

        with nogil:
            for i in prange(self.n_verts):
                self.tmp_verts[i] = (self.m.verts[i] - self.prev_verts[i]) * damping + self.m.verts[i]

        self.prev_verts, self.m.verts, self.tmp_verts = self.m.verts, self.tmp_verts, self.prev_verts

    cpdef void update_mesh_normals(self, float factor):
        self.tmp_verts, self.m.vert_normals = self.m.vert_normals, self.tmp_verts

        self.m.update_face_normals()
        self.m.update_vert_normals()

        cdef int i
        if factor < 1:
            with nogil:
                for i in prange(self.n_verts):
                    self.m.vert_normals[i] = self.tmp_verts[i].lerp(self.m.vert_normals[i], factor)
