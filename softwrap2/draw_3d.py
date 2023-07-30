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

'''
This is a utility module for drawing lines in the 3D viewport on Blender 2.8
using the GPU Api


The idea is to get rid of messy draw functions and data that is hard to keep track.
This class works directly like a callable draw handler and keeps track of all the geometry data.
'''

__all__ = ["DrawCallback"]

import bpy
import bgl
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
VERSION = bpy.app.version

vertex_shader = '''

uniform mat4 ModelViewProjectionMatrix;

#ifdef USE_WORLD_CLIP_PLANES
uniform mat4 ModelMatrix;
#endif

in vec3 pos;
in vec4 color;

out vec4 finalColor;

void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
    gl_Position.z -= 0.002;
    finalColor = color;

    #ifdef USE_WORLD_CLIP_PLANES
      world_clip_planes_calc_clip_distance((ModelMatrix * vec4(pos, 1.0)).xyz);
    #endif
}

'''


fragment_shader = """
in vec4 finalColor;
out vec4 fragColor;

void main()
{
  fragColor = finalColor;
  fragColor = blender_srgb_to_framebuffer_space(fragColor);
}
"""

point_fragment_shader = """
in vec4 finalColor;
in vec4 fragCoord;

out vec4 fragColor;

void main()
{
    vec2 coord = (gl_PointCoord - vec2(0.5, 0.5)) * 2.0;
    float fac = dot(coord, coord);
    if (fac > 0.5){
        discard;
    }
    fragColor = finalColor;
    fragColor = blender_srgb_to_framebuffer_space(fragColor);
}

"""


class DrawCallback:
    running_draws = set()

    def __init__(self):

        # Useful for rendering in the same space of an object
        self.matrix = Matrix().Identity(4)
        # X-ray mode, draw through solid objects
        self.draw_on_top = False
        # Blend mode to choose, set it to one of the blend constants.
        self.blend_mode = 'MULTIPLY'

        self.line_width = 1
        self.point_size = 3

        # Handler Placeholder
        self.draw_handler = None

        self.line_coords = []
        self.line_colors = []
        self.point_coords = []
        self.point_colors = []
        self._line_shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
        self._point_shader = gpu.types.GPUShader(vertex_shader, point_fragment_shader)
        self._line_batch = batch_for_shader(self._line_shader, 'LINES',
                                            {"pos": self.line_coords, "color": self.line_colors})
        self._point_batch = batch_for_shader(self._line_shader, 'POINTS',
                                             {"pos": self.point_coords, "color": self.line_colors})

    def __call__(self, *args, **kwargs):
        # __call__ Makes this object behave like a function.
        # So you can add it like a draw handler.
        self._draw()

    def setup_handler(self):
        # Utility function to easily add it as a draw handler
        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(self, (), "WINDOW", "POST_VIEW")
        self.__class__.running_draws.add(self)

    def remove_handler(self):
        # Utility function to remove the handler
        if not self.draw_handler:
            return
        bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, "WINDOW")
        self.draw_handler = None
        self.__class__.running_draws.remove(self)

    @classmethod
    def remove_all_handlers(cls):
        for draw in list(cls.running_draws):
            draw.remove_handler()

    def update_batch(self):
        # This takes the data rebuilds the shader batch.
        # Call it every time you clear the data or add new lines, otherwize,
        # You wont see changes in the viewport
        coords = [self.matrix @ Vector(coord) for coord in self.line_coords]
        self._line_batch = batch_for_shader(self._line_shader, 'LINES', {"pos": coords, "color": self.line_colors})
        coords = [self.matrix @ Vector(coord) for coord in self.point_coords]
        self._point_batch = batch_for_shader(self._point_shader, 'POINTS', {"pos": coords, "color": self.point_colors})

    def add_line(self, start, end, color1=(1, 0, 0, 1), color2=None):
        # Simple add_line function, support color gradients,
        # if only color1 is specified, it will be solid color (color1 on both ends)
        # This doesnt render a line, it just adds the vectors and colors to the data
        # so after calling update_batch(), it will be converted in a buffer Object
        self.line_coords.append(Vector(start))
        self.line_coords.append(Vector(end))
        self.line_colors.append(color1)
        if color2 is None:
            self.line_colors.append(color1)
        else:
            self.line_colors.append(color2)

    def add_point(self, location, color=(1, 0, 0, 1)):
        self.point_coords.append(location)
        self.point_colors.append(color)

    def clear_data(self):
        # just clear all the data
        self.line_coords = []
        self.line_colors = []
        self.point_coords = []
        self.point_colors = []

    def _start_drawing(self):
        # This handles all the settings of the renderer before starting the draw stuff
        if VERSION[1] >= 93:
            gpu.state.blend_set('ALPHA')

            # bgl.glDisable(bgl.GL_DEPTH_TEST)
            # if self.draw_on_top:
            if self.draw_on_top:
                gpu.state.depth_test_set('NONE')

            gpu.state.line_width_set(self.line_width)
            gpu.state.point_size_set(self.point_size)

        else:
            bgl.glEnable(bgl.GL_BLEND)

            if self.draw_on_top:
                bgl.glDisable(bgl.GL_DEPTH_TEST)

            bgl.glLineWidth(self.line_width)
            bgl.glPointSize(self.point_size)

    def _stop_drawing(self):

        # just reset some OpenGL stuff to not interfere with other drawings in the viewport
        # its not absolutely necessary but makes it safer.
        if VERSION[1] >= 93:
            pass

        else:
            bgl.glDisable(bgl.GL_BLEND)
            bgl.glLineWidth(1)
            bgl.glPointSize(1)
            if self.draw_on_top:
                bgl.glEnable(bgl.GL_DEPTH_TEST)

    def _draw(self):
        # This should be called by __call__,
        # just regular routines for rendering in the viewport as a draw_handler
        self._start_drawing()
        self._line_shader.bind()
        self._line_batch.draw(self._line_shader)
        self._point_shader.bind()
        self._point_batch.draw(self._point_shader)
        self._stop_drawing()


if __name__ == "__main__":
    # Simple example, run it on blender's text editor.

    # create a new instance of the class
    draw = DrawCallback()
    # add lines to ir
    draw.add_line((10, 0, 0), (-10, 0, 0), color1=(1, 0, 0, 1), color2=(0, 0, 1, 1))
    draw.add_line((0, 0, 0), (0, 0, 5), color1=(0, 1, 0, 1), color2=(0, 1, 1, 1))
    # enable X ray mode/see through objects and set Blend mode to Additive
    draw.draw_on_top = True
    draw.blend_mode = ADDITIVE_BLEND
    # set line width to 5
    draw.line_width = 5
    # Important, update batch always when adding
    # new lines, otherwize they wont render.
    draw.update_batch()
    # setup draw handler, optionally, you can call bpy.SpaceView3D.draw_handler_add()
    draw.setup_handler()
