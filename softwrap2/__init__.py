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

import bpy
import bmesh
import gc
import random
import time
from math import atan2, ceil
from . import softwrap_core2 as core
from . draw_3d import DrawCallback
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d, location_3d_to_region_2d
from mathutils.geometry import intersect_line_plane
from mathutils import Vector, Matrix
from collections import namedtuple, defaultdict
import bpy.app.handlers as handlers


bl_info = {'name': 'Softwrap 2',
           'description': 'Transfer topology from one model to another using a softbody simulation',
           'author': 'Jean Da Costa Machado',
           'version': ('2.1.2-alpha-for-blender3.1 -- a928357391',), # ship:-v
           'blender': (3, 1, 0),
           'doc_url': 'https://jeacom25b.github.io/Softwrap-Manual/',
           'tracker_url': 'https://jeacom25b.github.io/Softwrap-Manual/',
           'category': 'Mesh',
           'location': '3D view > properties (N-panel) > Softwrap 2'}

all_classes = []

running_op = None

SW_SHAPE_KEY_NAME = 'SoftWrap_Shape_key'


def register_cls(cls):
    all_classes.append(cls)
    return cls


class PerfTimer:
    timer_data = defaultdict(list)
    timer_tmp = {}

    def start(self, key='time'):
        self.timer_tmp[key] = time.time()

    def stop(self, key='time',  max_samples=30):
        self.timer_data[key].append(time.time() - self.timer_tmp[key])

    def __str__(self):
        return '<Timer [' + ', '.join(f'{key}: {sum(samples) / len(samples)}'
                                      for key, samples in self.timer_data.items()) + ']>'


def register_panel_draw(label=None, parent_cls=None, poll=lambda s, c: True):
    def decorator(draw_fnc):
        nonlocal label, parent_cls, poll
        if not label or callable(label):
            label = ' '.join(x.capitalize() for x in draw_fnc.__name__.split('_'))
        namespace = {
            'bl_label': label,
            'bl_space_type': 'VIEW_3D',
            'bl_region_type': 'UI',
            'bl_category': 'Softwrap 2',
            'poll': classmethod(poll),
            'draw': draw_fnc
        }
        if parent_cls:
            namespace['bl_parent_id'] = parent_cls.__name__

        return register_cls(type(f'VIEW3D_PT_softwrap2_{draw_fnc.__name__}',
                                 (bpy.types.Panel,),
                                 namespace))
    if callable(label):  # actually draw_fnc
        return decorator(label)
    return decorator


def smoothstep(f, fmin=0, fmax=1):
    f = max(0, min(1, (f - fmin) / (fmax - fmin)))
    return f * f * (3 - 2 * f)


def lerp(a, b, f):
    return (b - a) * f + a


def intersect_point_2d_rectangle(px, py, rx, ry, width, height):
    if px <= rx:
        return False
    if py <= ry:
        return False
    if px - rx >= width:
        return False
    if py - ry >= height:
        return False

    return True


def areas_under_mouse(context, event):
    mx, my = event.mouse_x, event.mouse_y
    areas = []
    for area in context.screen.areas:
        regions = []

        if intersect_point_2d_rectangle(mx, my, area.x, area.y, area.width, area.height):
            for region in area.regions:
                if intersect_point_2d_rectangle(mx, my, region.x, region.y, region.width, region.height):
                    regions.append(region)

        areas.append((area, regions))

    return areas


def get_mouse_ray(context, event, mat=Matrix.Identity(4)):
    region = context.region
    r3d = context.space_data.region_3d
    co = event.mouse_region_x, event.mouse_region_y
    origin = mat @ region_2d_to_origin_3d(region, r3d, co)
    vec = region_2d_to_vector_3d(region, r3d, co)
    vec.rotate(mat)
    return origin, vec


def mouse_raycast(obj, context, event):
    mat = obj.matrix_world.inverted()
    origin, vec = get_mouse_ray(context, event, mat)
    return obj.ray_cast(origin, vec)


def global_to_screen(co, context):
    region = context.region
    r3d = context.space_data.region_3d
    return location_3d_to_region_2d(region, r3d, co)


def vertex_group_to_list(obj, vg_name):
    vg = obj.vertex_groups.get(vg_name, None)
    if not vg:
        return None
    data = []
    for i in range(len(obj.data.vertices)):
        try:
            data.append(vg.weight(i))
        except RuntimeError:
            data.append(0)
    return data


def core_mesh_from_bm(bm):
    bm.verts.ensure_lookup_table()
    verts = [tuple(v.co) for v in bm.verts]

    def triangles(face):
        for i in range(len(face.verts) - 2):
            yield face.verts[0].index, face.verts[i + 1].index, face.verts[i + 2].index

    faces = [tri for f in bm.faces for tri in triangles(f)]
    return core.Mesh(verts, faces)


def deduplicate_links(items):
    if not items:
        return items
    return list(tuple(x) for x in set(frozenset(a) for a in items) if len(x) == 2)


def loop_pairs(elems):
    n = len(elems)
    half_n = n // 2

    for i in range(half_n + n % 2):
        yield (elems[i], elems[(i + half_n) % n])


def sort_vert_link_edges(vert):
    u = vert.normal.orthogonal().normalized()
    v = vert.normal.cross(u)
    o = vert.co

    def angle(edge):
        vec = vert.co - edge.other_vert(vert).co
        return atan2(vec.dot(u), vec.dot(v))

    return sorted(vert.link_edges, key=angle)


def sort_vert_link_loops(vert):
    u = vert.normal.orthogonal().normalized()
    v = vert.normal.cross(u)
    o = vert.co

    loops = vert.link_loops

    def angle(loop):
        vec = loop.vert.co - loop.link_loop_next.vert.co
        return atan2(vec.dot(u), vec.dot(v))

    return sorted(loops, key=angle)


def structural_springs_indexes(bm):
    springs = list(tuple(v.index for v in edge.verts) for edge in bm.edges)
    return springs


def somoothing_springs_indexes(bm):
    return [tuple(v.index for v in edge.verts) for edge in bm.edges if edge.verts[0].is_boundary == edge.verts[1].is_boundary]


def shear_spring_indexes(bm):
    return [(v.index for v in pair) for face in bm.faces for pair in loop_pairs(face.verts)]


def bending_spring_indexes(bm, distance=1):
    distance = max(distance, 1)

    springs = {}
    length_correlations = set()
    links_by_edge = defaultdict(list)

    for vert in bm.verts:
        for loop in vert.link_loops:
            edge = loop.edge
            links_by_edge[edge].append([])
            for _ in range(distance):
                loop = loop.link_loop_next
                for _ in range(len(loop.vert.link_edges) // 2 - 1):
                    loop = loop.link_loop_radial_next.link_loop_next

                other = loop.vert
                if vert.index == other.index:
                    continue

                spr = frozenset((vert.index, other.index))
                if spr not in springs:
                    springs[spr] = 1
                    links_by_edge[edge][-1].append(len(springs) - 1)

    for edge in bm.edges:
        for spring_chain in links_by_edge[edge]:
            for i in range(len(spring_chain) - 1):
                length_correlations.add(frozenset((spring_chain[i], spring_chain[i + 1])))

    return list(springs.keys())  # , list(length_correlations)


def ternary_links_indexes(bm):
    links = []
    for vert in bm.verts:
        for ea, eb in loop_pairs(sort_vert_link_edges(vert)):
            va = ea.other_vert(vert)
            vb = eb.other_vert(vert)
            link = vert.index, va.index, vb.index
            if not len(frozenset(link)) < 3:
                links.append(link)
    return links


def quaternary_link_indexes(bm):
    links = []
    for face in bm.faces:
        if len(face.verts) == 4:
            indexes = *(v.index for v in face.verts),
            links.append((indexes[0], indexes[1], indexes[3], indexes[2]))
            links.append((indexes[0], indexes[3], indexes[1], indexes[2]))
            links.append((indexes[0], indexes[2], indexes[1], indexes[3]))

    for vert in bm.verts:
        for e1, e2 in loop_pairs(sort_vert_link_edges(vert)):
            if e1.is_manifold and e2.is_manifold:
                links.append((vert.index, e1.other_vert(vert).index, vert.index, e2.other_vert(vert).index))

    return links


def iter_float_factor(value, max_val, power_fac=1, size=1):
    value = (value / size) ** power_fac * size
    i = 0
    while value > 0:
        yield i, min(value, max_val)
        value -= max_val
        i += 1


@register_cls
class SoftwrapSettings(bpy.types.PropertyGroup):
    def stop_engine(self, context):
        if running_op:
            running_op.stop(context)

    def set_wire(self, context):
        if self.source_ob:
            self.wire = self.source_ob.show_wire

    def source_ob_update(self, context):
        if self.source_ob and not self.source_ob.type == 'MESH':
            self.source_ob = None
        self.stop_engine(context)
        self.set_wire(context)

    def target_ob_update(self, context):
        if self.target_ob and not self.target_ob.type == 'MESH':
            self.target_ob = None
        self.stop_engine(context)

    def wire_update(self, context):
        if self.source_ob:
            self.source_ob.show_wire = self.wire
            self.source_ob.show_all_edges = self.wire

    def snapping_group_update(self, context):
        if running_op:
            running_op.snapping_mask_update(context)

    def simulation_group_update(self, context):
        if running_op:
            running_op.simulation_mask_update(context)

    def mesh_poll(self, ob):
        return ob.type == 'MESH' and ob.name in bpy.context.scene.objects

    source_ob: bpy.props.PointerProperty(
        name='source mesh', type=bpy.types.Object, update=source_ob_update, poll=mesh_poll,
        description='The mesh that is going to be deformed into a new shape using the target mesh as reference')

    target_ob: bpy.props.PointerProperty(
        name='target mesh', type=bpy.types.Object, update=target_ob_update, poll=mesh_poll,
        description='The mesh that is going to be used as reference, (tipcally a sculpt or a 3d scan)')

    snapping_group: bpy.props.StringProperty(
        name='Snapping Group', update=snapping_group_update,
        description='vertex group to mask what vertices are involved on the snapping')

    snapping_group_invert: bpy.props.BoolProperty(
        name='Invert Snapping Group',
        description='Inverts the vertex group influence',)

    simulation_group: bpy.props.StringProperty(
        name='Simulation Group', update=simulation_group_update,
        description='vertex group to mask what vertices should be simulated,')

    simulation_group_invert: bpy.props.BoolProperty(
        name='Invert Simulation Group',
        description='Inverts the vertex group influence',)

    wire: bpy.props.BoolProperty(
        name='Wire', update=wire_update,
        description='Toggle wireframe display on the source mesh')

    mirror: bpy.props.BoolVectorProperty(
        name='Mirror', size=3, default=(False, False, False),
        description='Enforce symmetry acrons an axis.')

    min_scaling: bpy.props.FloatProperty(
        name='Min Scaling', default=0.3,
        description='Minimun allowed rest length for edges')

    max_scaling: bpy.props.FloatProperty(
        name='Max Scaling', default=3,
        description='Maximun allowed rest length for edges')

    scale_plasticity: bpy.props.FloatProperty(
        name='Scale Plasticity', min=0, max=1, default=0.1,
        description='Amount of semi-permanent deformation per frame')

    scale_restoration: bpy.props.FloatProperty(
        name='Scale Restoration', min=0, max=1, default=0.05,
        description='Amount of restoration to the semi-parmanent deformation per frame')

    smooth: bpy.props.FloatProperty(
        name='Smooth', min=0, soft_max=5, default=0,
        description='Amount of somothing applied to the mesh per frame (nonlinear)\n'
                    'Note: has the side effect of swrinking the mesh,')

    quad_smoothing: bpy.props.FloatProperty(
        name='Quad Smooth', min=0, soft_max=10, default=0,
        description='Amount of force applied per frame to restore the shape of quads (nonlinear)\n'
                    'Note: has the side effect of swrinking the mesh,')

    topologic_smooth: bpy.props.FloatProperty(
        name='Topologic Smooth', min=0, soft_max=5, default=2,
        description='Amount of topology-aware smoothing applied to the mesh per frame, (nonlinear)\n'
                    'Note: Less aggressive than smooth, ideal for removing kinks in edge loops caused by by pins')

    structural_stiffness: bpy.props.FloatProperty(
        name='Structural Stiffness', min=0, soft_max=10, default=2,
        description='Stiffness of the direct links between vertices')

    bending_stiffness: bpy.props.FloatProperty(
        name='Bending Stiffness', min=0, soft_max=10, default=2,
        description='Stiffness of the links across multiple edges')

    shear_stiffness: bpy.props.FloatProperty(
        name='Shear Stiffness', min=0, soft_max=10, default=2,
        description='Stiffness of the links across face diagonals')

    bending_distance: bpy.props.IntProperty(
        name='Bending Distance', min=0, default=3,
        description='Maximun distance (by edges) for bending springs to be created')

    damping: bpy.props.FloatProperty(
        name='Damping', min=0, max=1, default=0.25,
        description='Dampen the simulation to increase stability')

    simulation_steps: bpy.props.IntProperty(
        name='Simulation Steps', min=0, default=2,
        description='Number of simulation steps per frame')

    snapping_quality: bpy.props.IntProperty(
        name='Snapping Quality', min=1, max=20, default=10,
        description='How of often to update the snapping direction from the source mesh to the target mesh')

    snapping_force: bpy.props.FloatProperty(
        name='Snapping Strength', min=0, max=1, default=0,
        description='Strength of the snapping, how much it pulls the source mesh towards the target mesh')

    snapping_mode: bpy.props.EnumProperty(
        name='Snap Mode', default='SURFACE',
        items = [('SURFACE', 'Surface', 'Surface'),
                 ('OUTSIDE', 'Outside', 'Outside'),
                 ('INSIDE', 'Inside', 'Inside')],
        description='Controls which side of the target mesh affects the snapping.')

    project_pins: bpy.props.BoolProperty(
        name='Project Pins', default=True,
        description='Virtually snap pins to the surface of the target mesh based on snapping force.\n')

    pin_force: bpy.props.FloatProperty(
        name='Pin Force', min=0, max=1, default=1,
        description='How strongly pins pull on the mesh')

    mouse_grab_size: bpy.props.IntProperty(
        name='Mouse Grab Size', min=1, default=3,
        description='Size of the area grabbed by the mouse')

    pause: bpy.props.BoolProperty(
        name='Pause (Space)', default=False,
        description='Temporarily halt the simulation')

    interact_mouse: bpy.props.BoolProperty(
        name='Interaction (Shift + Space)', default=True,
        description='Enable interception of mouse events for pin creation and grabbing')

    mouse_button: bpy.props.EnumProperty(
        name='Interact With', default='LEFTMOUSE',
        items=[('LEFTMOUSE', 'Left Mouse', 'Left Mouse'), ('RIGHTMOUSE', 'Right Mouse', 'Right Mouse')],
        description='Which mouse button to use for interacting with the simulation ')


def get_settings(context):
    return bpy.context.scene.softwrap2


S = type('SettingsProbe', (), {
    '__getattr__': lambda s, k: getattr(get_settings(bpy.context), k),
    '__setattr__': lambda s, k, v: setattr(get_settings(bpy.context), k, v),
    '__call__': lambda s: get_settings(bpy.context)
})()


@register_panel_draw
def softwrap_2(self, context):
    pass


@register_panel_draw(parent_cls=softwrap_2)
def interaction(self, context):
    layout = self.layout
    if running_op:
        if S.interact_mouse:
            mouse_side = S.mouse_button.replace('_', ' ').lower()
            layout.label(text=f'[shift + {mouse_side}] to add a pin')
        else:
            layout.label(text=f'Mouse interaction disabled')

    col = layout.column(align=True)
    col.prop(S(), 'interact_mouse', toggle=True)
    col.prop(S(), 'pause', toggle=True)
    row = col.row(align=True)
    row.prop(S(), 'mouse_button', expand=True)
    col.separator()

    layout.prop(S(), 'mouse_grab_size')


@register_panel_draw(parent_cls=softwrap_2)
def initialization(self, context):
    layout = self.layout
    col = layout.column(align=True)
    row = col.row(align=True)
    row.scale_y = 2
    if running_op:
        running_anim = ['*---', '-*--',
                        '--*-', '---*',
                        '--*-', '-*--'][int(time.time() * 5) % 5]
        row.operator('object.start_softwrap', text=f'Stop {running_anim}', icon='CANCEL')
        row.prop(S(), 'pause', icon='PAUSE')
    else:
        row.operator('object.start_softwrap', text='Start', icon='PLAY')

    row = col.row(align=True)
    row.operator('object.apply_softwrap')
    row.operator('object.reset_softwrap')
    row.operator('object.pins_remove_softwrap')

    if S.source_ob:
        layout.label(text='Display')
        row = layout.row(align=True)
        row.prop(S(), 'wire', toggle=True)
        row.prop(S().source_ob, 'show_in_front', toggle=True)
        layout.separator()

    col = layout.column(align=True)
    col.label(text='Source Mesh:')
    col.prop(S(), 'source_ob', text='')
    col.separator()

    col.label(text='Target Mesh:')
    col.prop(S(), 'target_ob', text='')
    layout.separator()

    col = layout.column()
    col.active = not running_op
    col.prop(S(), 'bending_distance')


@register_panel_draw(parent_cls=softwrap_2)
def dynamics(self, context):
    pass


@register_panel_draw(parent_cls=dynamics)
def symmetry(self, context):
    layout = self.layout
    row = layout.row(align=True)
    row.label(text='Mirror')
    for i, axis in enumerate('XYZ'):
        row.prop(S(), 'mirror', text=axis, index=i, toggle=True)

    if running_op:
        for axis, enable, error, scale, dim in zip('XYZ', S.mirror, running_op.symmetry_map.error, S.source_ob.scale, S.source_ob.dimensions):
            if enable and error / (dim / scale) > 0.05:
                layout.label(text=f'Warning: source mesh may not be symmetrical at axis {axis}', icon='ERROR')
                layout.label(text=f'    Error: {round(error / (dim / scale) , 6)}')


@register_panel_draw(parent_cls=dynamics)
def Stiffness(self, context):
    layout = self.layout
    col = layout.column(align=True)
    col.prop(S(), 'structural_stiffness', slider=True)
    col.prop(S(), 'shear_stiffness', slider=True)
    col.prop(S(), 'bending_stiffness', slider=True)
    layout.prop(S(), 'damping', slider=True)


@register_panel_draw(parent_cls=dynamics)
def smoothing(self, context):
    layout = self.layout
    layout.prop(S(), 'smooth', slider=True)
    layout.prop(S(), 'quad_smoothing', slider=True)
    layout.prop(S(), 'topologic_smooth', slider=True)
    layout.separator()


@register_panel_draw(parent_cls=dynamics)
def snapping(self, context):
    layout = self.layout
    layout.prop(S(), 'snapping_force', slider=True)
    layout.prop(S(), 'snapping_mode')

    layout.separator()

    if S.source_ob:
        layout.label(text='Snapping Group')
        row = layout.row(align=True)
        row.prop_search(S(), 'snapping_group', S.source_ob, 'vertex_groups', text='')
        row.prop(S(), 'snapping_group_invert', text='', icon='ARROW_LEFTRIGHT')

        layout.label(text='Simulation Group')
        row = layout.row(align=True)
        row.prop_search(S(), 'simulation_group', S.source_ob, 'vertex_groups', text='')
        row.prop(S(), 'simulation_group_invert', text='', icon='ARROW_LEFTRIGHT')

    layout.label(text='Pins:')
    layout.prop(S(), 'project_pins')
    layout.prop(S(), 'pin_force', slider=True)


@register_panel_draw(parent_cls=dynamics)
def plasticity(self, context):
    layout = self.layout
    row = layout.row(align=True)
    row.prop(S(), 'min_scaling')
    row.prop(S(), 'max_scaling')

    row = layout.row(align=True)
    row.prop(S(), 'scale_plasticity')
    row.prop(S(), 'scale_restoration')


@register_panel_draw(parent_cls=softwrap_2)
def performance(self, context):
    layout = self.layout
    layout.prop(S(), 'simulation_steps')
    layout.prop(S(), 'snapping_quality', slider=True)


PinCacheData = namedtuple('PinCacheData', 'engine_pin type scale factor world_pos', defaults=(None,) * 5)


@register_cls
class OBJECT_OT_start_softwrap(bpy.types.Operator):
    bl_idname = 'object.start_softwrap'
    bl_label = 'Test springs'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Run the softwrap softbody engine'

    _timer = None

    source_mesh = None
    target_mesh = None

    bvh = None
    engine = None
    structural_springs = None
    shear_springs = None
    bending_springs = None

    symmetry_map = None

    mouse_pin_pos = None
    mouse_pin_delta = None
    mouse_pin = None

    pin_cache = None

    simulation_mask = None
    snapping_mask = None

    draw3d = None

    last_mode = None

    perf_timer = None

    @classmethod
    def poll(self, context):

        if S.target_ob and not S.target_ob.type == 'MESH':
            return False
        return S.source_ob and S.source_ob.type == 'MESH'

    def invoke(self, context, event):

        global running_op
        if running_op:
            running_op.stop(context)
            return {'CANCELLED'}

        S.source_ob.data.update()
        bm = bmesh.new()
        bm.from_mesh(S.source_ob.data)

        self.source_mesh = core_mesh_from_bm(bm)

        if S.target_ob:
            tbm = bmesh.new()
            tbm.from_mesh(S.target_ob.data)
            bmesh.ops.transform(tbm, matrix=S.target_ob.matrix_world, space=Matrix.Identity(4), verts=tbm.verts)
            bmesh.ops.transform(tbm, matrix=S.source_ob.matrix_world.inverted(), space=Matrix.Identity(4), verts=tbm.verts)
            self.target_mesh = core_mesh_from_bm(tbm)
            self.bvh = core.BVH(self.target_mesh)

        self.engine = core.SpringEngine(self.source_mesh, self.bvh)

        # import pdb
        # pdb.set_trace()
        self.structural_springs = self.engine.create_spring_group(structural_springs_indexes(bm))
        self.shear_springs = self.engine.create_spring_group(shear_spring_indexes(bm))
        self.bending_springs = self.engine.create_spring_group(bending_spring_indexes(bm, S.bending_distance))

        self.symmetry_map = self.engine.create_symmetry_map()

        self.ternary_links = self.engine.create_ternary_links(ternary_links_indexes(bm))
        self.ternary_links.displacements_update()

        self.quaternary_links = self.engine.create_quaternary_links(quaternary_link_indexes(bm))
        self.quaternary_links.lengths_update()

        running_op = self

        self.snapping_mask_update(context)
        self.simulation_mask_update(context)

        self.pin_cache = {}
        self.draw3d = DrawCallback()
        self.draw3d.point_size = 8
        self.draw3d.line_width = 3
        self.draw3d.setup_handler()

        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(1 / 60, window=context.window)
        self.perf_timer = PerfTimer()  # completelly unrelated to self._timer
        return {'RUNNING_MODAL'}

    def mouse_pin_set(self, context, event, create_empty=False):

        closest_distance = float('inf')
        closest_empty = None
        for pin_obj in S.source_ob.get('sw_pins', []):
            pin_2d = global_to_screen(pin_obj.location, context)
            if not pin_2d:
                continue  # behind camera
            mouse_2d = Vector((event.mouse_region_x, event.mouse_region_y))
            dist = (pin_2d - mouse_2d).length
            if dist < closest_distance:
                closest_empty = pin_obj
                closest_distance = dist

        if closest_empty and closest_distance < 30:
            bpy.ops.object.select_all(action='DESELECT')
            closest_empty.select_set(True)
            context.view_layer.objects.active = closest_empty
            return True

        result, location, normal, index = mouse_raycast(S.source_ob, context, event)

        if not result:
            return False

        if not S.source_ob.modifiers:
            index = min(S.source_ob.data.polygons[index].vertices,
                        key=lambda i: (self.get_shape(context).data[i].co - location).length)

        else:
            index, v_location = self.source_mesh.closest_vert(location)

        if not create_empty:
            self.mouse_pin_pos = S.source_ob.matrix_world @ location
            self.mouse_pin_delta = self.mouse_pin_pos - S.source_ob.matrix_world @ self.get_shape(context).data[index].co
            self.mouse_pin = core.SpringEnginePin(self.structural_springs, index, S.mouse_grab_size)

        else:
            empty = bpy.data.objects.new(S.source_ob.name + '_pin', None)

            empty.empty_display_type = 'SPHERE'
            radius_disp = self.structural_springs[index].avg_radius
            empty.empty_display_size = radius_disp * 2 * sum(S.source_ob.scale) / 3
            empty.show_in_front = True

            empty.location = S.source_ob.matrix_world @ self.get_shape(context).data[index].co

            empty['vert_idx'] = index
            bpy.context.collection.objects.link(empty)
            pins = list(S.source_ob.get('sw_pins', []))
            pins.append(empty)
            S.source_ob['sw_pins'] = pins
            bpy.ops.object.select_all(action='DESELECT')
            empty.select_set(True)
            bpy.context.view_layer.objects.active = empty
            bpy.ops.ed.undo_push(message='Add Pin')

        return True

    def mouse_pin_update(self, context, event):
        o, dir = get_mouse_ray(context, event)
        plane = context.space_data.region_3d.view_rotation @ Vector((0, 0, 1))
        self.mouse_pin_pos = intersect_line_plane(o, o + dir, self.mouse_pin_pos, plane)

    def mouse_pin_clear(self, context, event):
        self.mouse_pin_pos = None
        self.mouse_pin = None
        self.mouse_pin_delta = None

    def empty_pin_scale(self, empty):
        return max((sum(empty.scale) / 3) * 4, 1.000001)

    def pin_cache_update(self, context, event):
        mat = S.source_ob.matrix_world
        mat_inv = S.source_ob.matrix_world.inverted()

        S.source_ob['sw_pins'] = [
            empty for empty in S.source_ob.get('sw_pins', ())
            if empty and empty.name in context.scene.objects
        ]

        new_cache = {}
        seen_vert_idxs = set()

        def new_pin_reuse_cache(index, pin_type, scale, factor, location, n_rings):
            previous_cache = self.pin_cache.get(index, PinCacheData(None))
            previous_pin = previous_cache.engine_pin

            if previous_pin and previous_pin.n_rings == n_rings:
                new_cache[index] = PinCacheData(engine_pin=previous_pin,
                                                type=pin_type,
                                                scale=scale,
                                                factor=factor,
                                                world_pos=location)
            else:
                pin = core.SpringEnginePin(self.structural_springs, index, n_rings)
                new_cache[index] = PinCacheData(engine_pin=pin,
                                                type=pin_type,
                                                scale=scale,
                                                factor=factor,
                                                world_pos=location)

        for empty in S.source_ob.get('sw_pins', ()):
            index = empty['vert_idx']

            if empty == context.active_object and empty.select_get(view_layer=context.view_layer):
                pin_type = 'ACTIVE_PIN'
            else:
                pin_type = 'OBJ_PIN'

            vec = mat_inv @ empty.location
            vec -= Vector(self.engine.get_verts([index])[0])

            scale = self.empty_pin_scale(empty)
            n_rings = int(scale)

            new_pin_reuse_cache(index, pin_type, scale, 1, empty.location, n_rings)

        if self.mouse_pin_pos:
            self.mouse_pin_update(context, event)
            pin_type = 'MOUSE_PIN'
            index = self.mouse_pin.start_index
            new_cache[index] = PinCacheData(engine_pin=self.mouse_pin,
                                            type=pin_type,
                                            scale=self.mouse_pin.n_rings + 1,
                                            factor=1,
                                            world_pos=self.mouse_pin_pos - self.mouse_pin_delta)

        for axis in range(3):
            if S.mirror[axis]:
                for idx, pin_data in list(new_cache.items()):
                    mirr_idx = self.symmetry_map[idx][axis]
                    if mirr_idx == idx:
                        continue
                    location = mat_inv @ pin_data.world_pos
                    location[axis] *= -1
                    location = mat @ location
                    pin_type = 'MIRROR_' + pin_data.type

                    new_pin_reuse_cache(mirr_idx,
                                        pin_type,
                                        pin_data.scale,
                                        pin_data.factor,
                                        location,
                                        pin_data.engine_pin.n_rings)

        self.pin_cache = new_cache

    def pin_cache_apply(self, context, event, factor=1, mouse_factor=1):

        mat_inv = S.source_ob.matrix_world.inverted()

        for index, pin_data in self.pin_cache.items():
            vert_loc = Vector(self.engine.get_verts([index])[0])
            local_pos = mat_inv @ pin_data.world_pos
            if self.bvh and S.project_pins and S.snapping_force > 0 and not pin_data.type == 'MOUSE_PIN':
                f = S.snapping_force
                if self.snapping_mask:
                    f *= 1 - self.snapping_mask[index]
                local_pos = lerp(local_pos, Vector(self.bvh.find_nearest(local_pos)[0]), f)
            vec = local_pos - vert_loc
            if pin_data.type == 'MOUSE_PIN':
                f = mouse_factor
            else:
                f = factor
            pin_data.engine_pin.move(*(vec * pin_data.factor * f), pin_data.scale)

    def draw_pins(self, context, event):

        red = Vector((1, 0, 0, 0.5))
        green = Vector((0, 0.9, 0, 0.5))
        blue = Vector((0, 0, 0.5, 0.5))
        cyan = Vector((0, 0.2, 0.8, 0.5))
        purple = (cyan + red) / 2

        self.draw3d.clear_data()

        for index, pin_data in self.pin_cache.items():

            vert_loc = S.source_ob.matrix_world @ Vector(self.engine.get_verts([index])[0])

            if pin_data.type.startswith('MIRROR_'):
                self.draw3d.add_line(vert_loc, pin_data.world_pos, purple, cyan)
                self.draw3d.add_point(vert_loc, purple)
                self.draw3d.add_point(pin_data.world_pos, cyan)
            else:
                self.draw3d.add_line(vert_loc, pin_data.world_pos, purple, red)
                self.draw3d.add_point(vert_loc, purple)
                self.draw3d.add_point(pin_data.world_pos, red)

            if pin_data.type in {'ACTIVE_PIN', 'MOUSE_PIN'}:

                for i, ring in enumerate(pin_data.engine_pin):
                    indexes = list(ring)
                    locations = self.engine.get_verts(indexes)
                    f = 1 - max((pin_data.scale - i - 1) / (pin_data.scale - 1), 0)
                    color = lerp(red, green, smoothstep(f, 0, 0.7) ** 3)
                    color = lerp(color, blue, smoothstep(f, 0.7, 1))
                    color.w = (1 - f) ** 0.2
                    for loc in locations:
                        loc = S.source_ob.matrix_world @ Vector(loc)
                        self.draw3d.add_point(loc, color)

        self.draw3d.update_batch()

    def snapping_mask_update(self, context):
        vg_data = vertex_group_to_list(S.source_ob, S.snapping_group)
        self.snapping_mask = self.engine.create_mask(vg_data)

    def simulation_mask_update(self, context):
        vg_data = vertex_group_to_list(S.source_ob, S.simulation_group)
        self.simulation_mask = self.engine.create_mask(vg_data)

    def reset_simulation(self, context):
        vdata = [0] * len(self.engine)
        S.source_ob.data.vertices.foreach_get('co', vdata)
        self.engine.from_list(vdata)
        self.engine.kinetic_step(0)
        self.structural_springs.lengths_update()
        self.shear_springs.lengths_update()
        self.bending_springs.lengths_update()
        self.ternary_links.displacements_update()
        self.quaternary_links.lengths_update()

    def load_shape_to_engine(self, context):
        vdata = [0] * len(self.engine)
        self.get_shape(context).data.foreach_get('co', vdata)
        self.engine.from_list(vdata)
        self.engine.kinetic_step(0)

    def get_shape(self, context):
        sk = S.source_ob.data.shape_keys
        if sk and SW_SHAPE_KEY_NAME in sk.key_blocks:
            return sk.key_blocks[SW_SHAPE_KEY_NAME]
        else:
            if not sk or len(sk.key_blocks) == 0:
                S.source_ob.shape_key_add(name='Basis')
            shape = S.source_ob.shape_key_add(name=SW_SHAPE_KEY_NAME)
            shape.value = 1
            return shape

    error = None

    def modal(self, context, event):
        global running_op
        if self.error:
            print('operator runned twice')
            return {'FINISHED'}

        try:
            return self.modal_impl(context, event)
        except Exception as e:
            self.error = e
            self.stop(context)
            raise e

    def stop(self, context):
        self.draw3d.remove_handler()
        bpy.context.window_manager.event_timer_remove(self._timer)
        global running_op
        running_op = None

    def modal_impl(self, context, event):
        global running_op

        if not running_op or event.type == 'ESC'\
                or not S.source_ob\
                or not len(S.source_ob.data.vertices) * 3 == len(self.engine):

            self.stop(context)
            return {'FINISHED'}

        if not S.source_ob.mode == 'OBJECT':
            self.get_shape(context).mute = True
        else:
            self.get_shape(context).mute = False

        if not self.last_mode == 'OBJECT' and S.source_ob.mode == 'OBJECT':
            self.reset_simulation(context)
            self.load_shape_to_engine(context)
            self.simulation_mask_update(context)
            self.snapping_mask_update(context)

        self.last_mode = S.source_ob.mode

        if event.type == 'SPACE' and event.value == 'PRESS' and not event.ctrl and S.source_ob.mode == 'OBJECT':
            if event.shift:
                S.interact_mouse = not S.interact_mouse
            else:
                S.pause = not S.pause
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'TIMER':

            if not S.pause and S.source_ob.mode == 'OBJECT':

                with self.simulation_mask.masked_context(invert=S.simulation_group_invert):

                    for _ in range(S.simulation_steps):

                        for i, f in iter_float_factor(S.quad_smoothing, 1, 3, 10):
                            self.quaternary_links.smooth(f)

                        for i, f in iter_float_factor(S.shear_stiffness, 1, 3, 10):
                            if i > 0:
                                self.shear_springs.stiff_spring_force(f)
                            else:
                                self.shear_springs.soft_spring_force(f, deform_update=S.scale_plasticity,
                                                                     deform_restore=S.scale_restoration,
                                                                     min_deform=S.min_scaling,
                                                                     max_deform=S.max_scaling)

                        for i, f in iter_float_factor(S.bending_stiffness, 1, 3, 10):
                            if i > 0:
                                self.bending_springs.stiff_spring_force(f)
                            else:
                                self.bending_springs.soft_spring_force(f, deform_update=S.scale_plasticity,
                                                                       deform_restore=S.scale_restoration,
                                                                       min_deform=S.min_scaling,
                                                                       max_deform=S.max_scaling)

                        for i, f in iter_float_factor(S.structural_stiffness, 1, 3, 10):
                            if i > 0:
                                self.structural_springs.stiff_spring_force(f)
                            else:
                                self.structural_springs.soft_spring_force(f, deform_update=S.scale_plasticity,
                                                                          deform_restore=S.scale_restoration,
                                                                          min_deform=S.min_scaling,
                                                                          max_deform=S.max_scaling)

                    self.engine.kinetic_step(1 - S.damping)

                    for i, f in iter_float_factor(S.smooth, 0.5, 3, 5):
                        self.structural_springs.smooth(f)

                    for i, f in iter_float_factor(S.topologic_smooth, 0.5, 3, 5):
                        self.ternary_links.displacement_force(f)

                    self.engine.update_mesh_normals(1)

                    f = S.pin_force ** 4
                    for i in range(3):
                        self.pin_cache_update(context, event)
                        self.pin_cache_apply(context, event, factor=((i + 1) / 3) * f, mouse_factor=1)

                    with self.snapping_mask.masked_context(invert=S.snapping_group_invert):
                        if self.bvh and S.snapping_force > 0:
                            self.engine.snap_to_bvh(S.snapping_force ** 3, 20 - S.snapping_quality + 1, snapping_mode = S.snapping_mode)

                    self.symmetry_map.mirror(*S.mirror)

                self.get_shape(context).data.foreach_set('co', self.engine)
                S.source_ob.data.update()

                self.draw_pins(context, event)

            else:
                self.pin_cache_update(context, event)
                self.draw_pins(context, event)

            return {'PASS_THROUGH'} if not self.mouse_pin_pos else {'RUNNING_MODAL'}

        if event.type == S.mouse_button and event.value == 'PRESS' and S.interact_mouse:

            areas = areas_under_mouse(context, event)
            bad_region = False

            for area, regions in areas:
                if area.type == 'VIEW_3D':
                    r_types = set(r.type for r in regions)
                    if {'UI', 'HEADER', 'TOOLS'} & r_types:
                        bad_region = True

            if not bad_region and self.mouse_pin_set(context, event, create_empty=event.shift):
                return {'RUNNING_MODAL'}

        elif event.type == S.mouse_button and event.value == 'RELEASE':
            self.mouse_pin_clear(context, event)
            return {'PASS_THROUGH'}

        return {'PASS_THROUGH'}


@register_cls
class OBJECT_OT_apply_softwrap(bpy.types.Operator):
    bl_idname = 'object.apply_softwrap'
    bl_label = 'Apply'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Apply the deformation and exit simulation if running'

    @classmethod
    def poll(self, context):
        return get_settings(context).source_ob

    def execute(self, context):

        shapes = S.source_ob.data.shape_keys
        bpy.ops.object.pins_remove_softwrap()

        if shapes and SW_SHAPE_KEY_NAME in shapes.key_blocks:
            data = [0] * (len(S.source_ob.data.vertices) * 3)
            shapes.key_blocks[SW_SHAPE_KEY_NAME].data.foreach_get('co', data)
            S.source_ob.shape_key_remove(shapes.key_blocks[SW_SHAPE_KEY_NAME])

            if S.source_ob.data.shape_keys:
                if len(shapes.key_blocks) == 1:
                    S.source_ob.shape_key_remove(shapes.key_blocks[0])
                    S.source_ob.data.vertices.foreach_set('co', data)
                else:
                    shapes.key_blocks[0].data.foreach_set('co', data)
            else:
                S.source_ob.data.vertices.foreach_set('co', data)
        else:
            return {'CANCELLED'}

        if running_op:
            running_op.reset_simulation(context)

        return {'FINISHED'}


@register_cls
class OBJECT_OT_reset_softwrap(bpy.types.Operator):
    bl_idname = 'object.reset_softwrap'
    bl_label = 'Reset'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Reset the deformation'

    @classmethod
    def poll(self, context):
        return get_settings(context).source_ob

    def execute(self, context):

        if running_op:
            running_op.reset_simulation(context)

        shapes = S.source_ob.data.shape_keys
        if shapes and SW_SHAPE_KEY_NAME in shapes.key_blocks:
            S.source_ob.shape_key_remove(shapes.key_blocks[SW_SHAPE_KEY_NAME])
            if S.source_ob.data.shape_keys and len(shapes.key_blocks) == 1:
                S.source_ob.shape_key_remove(shapes.key_blocks[0])
            return {'FINISHED'}
        return {'CANCELLED'}


@register_cls
class OBJECT_OT_remove_pins_softwrap(bpy.types.Operator):
    bl_idname = 'object.pins_remove_softwrap'
    bl_label = 'Delete Pins'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Delete all pins assigned to this mesh'

    @classmethod
    def poll(self, context):
        return get_settings(context).source_ob

    def execute(self, context):

        for pin_obj in S.source_ob.get('sw_pins', []):
            bpy.data.objects.remove(pin_obj)

        S.source_ob['sw_pins'] = []
        del S.source_ob['sw_pins']
        return {'FINISHED'}


@handlers.persistent
def load_pre_handler(scene):
    S().stop_engine(bpy.context)
    DrawCallback.remove_all_handlers()


def register():
    handlers.load_pre.append(load_pre_handler)
    for cls in all_classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.softwrap2 = bpy.props.PointerProperty(type=SoftwrapSettings)


def unregister():
    handlers.load_pre.remove(load_pre_handler)
    for cls in all_classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.softwrap2
