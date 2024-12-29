#!/bin/sh

glslangValidator -V shader_horizontal.vert -o vert_h.spv
glslangValidator -V shader_vertical.vert -o vert_v.spv
glslangValidator -V shader.frag -o frag.spv
