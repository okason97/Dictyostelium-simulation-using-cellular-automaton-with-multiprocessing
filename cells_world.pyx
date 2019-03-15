# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import cython
import numpy as np
from cython cimport parallel
from random import randint
from libc.stdlib cimport rand, abs
cimport openmp

@cython.boundscheck(False)
@cython.wraparound(False)
def run_world(double[:,:] world not None,
              int[:,:] entity_map not None,
              int[:,:] dictys not None,
              int num_threads = 4):
    cdef int n_dictys, new_x, new_y, vision_x, vision_y, influence_range
    cdef int i, j, k, energy, x, y
    cdef double max, vision_value, chem_food, chem_decay
    cdef openmp.omp_lock_t entity_lock, world_lock
    if world.shape[0] * world.shape[1] < dictys.shape[0]:
        raise ValueError('Too many dictys')
    n_dictys = dictys.shape[0]
    vision = []
    influence_range = 100
    chem_decay = 0.01
    chem_food = 0.1
    openmp.omp_init_lock(&entity_lock)
    openmp.omp_init_lock(&world_lock)
    with nogil, cython.boundscheck(False), cython.wraparound(False):
        for i in parallel.prange(n_dictys, schedule = 'static', 
                                 num_threads = num_threads):
            x = dictys[i,0]
            y = dictys[i,1]
            energy = dictys[i,2]
            # cell's world view
            # search for strongest chem signal
            max = 0
            vision_value = world[x-1, y] 
            if x != 0 and vision_value > max:
                max = vision_value
                new_x = x-1
                new_y = y
            vision_value = world[x+1, y]
            if x != world.shape[0] - 1 and vision_value > max:
                max = vision_value
                new_x = x+1
                new_y = y
            if y != 0:
                vision_value = world[x, y-1]
                if vision_value > max:
                    max = vision_value
                    new_x = x
                    new_y = y-1
                if x != 0:
                    vision_value = world[x-1, y-1]
                    if vision_value > max:
                        max = vision_value
                        new_x = x-1
                        new_y = y-1
                if x != world.shape[0] - 1:
                    vision_value = world[x+1, y-1]
                    if vision_value > max:
                        max = vision_value
                        new_x = x+1
                        new_y = y-1
            if y != world.shape[1] - 1:
                vision_value = world[x, y+1]
                if vision_value > max:
                    max = vision_value
                    new_x = x
                    new_y = y+1
                if x != 0:
                    vision_value = world[x-1, y+1]
                    if vision_value > max:
                        max = vision_value
                        new_x = x-1
                        new_y = y+1
                if x != world.shape[0] - 1:
                    vision_value = world[x+1, y+1]
                    if vision_value > max:
                        max = vision_value
                        new_x = x+1
                        new_y = y+1
            # found a signal, go there are look for food
            if max != 0:
                # move
                openmp.omp_set_lock(&entity_lock)
                # found food
                if entity_map[new_x, new_y] != 2:
                    if entity_map[new_x, new_y] == 1:
                        energy = energy + 4
                        openmp.omp_set_lock(&world_lock)
                        world[new_x,new_y] = world[new_x,new_y] - chem_food
                        if new_x != 0:
                            world[new_x-1,new_y] = world[new_x-1,new_y] - chem_decay
                        if new_x != world.shape[0] - 1:
                            world[new_x+1,new_y] = world[new_x+1,new_y] - chem_decay
                        if new_y != 0:
                            world[new_x,new_y-1] = world[new_x,new_y-1] - chem_decay
                            if new_x != 0:
                                world[new_x-1,new_y-1] = world[new_x-1,new_y-1] - chem_decay
                            if new_x != world.shape[0] - 1:
                                world[new_x+1,new_y-1] = world[new_x+1,new_y-1] - chem_decay
                        if new_y != world.shape[1] - 1:
                            world[new_x,new_y+1] = world[new_x,new_y+1] - chem_decay
                            if new_x != 0:
                                world[new_x-1,new_y+1] = world[new_x-1,new_y+1] - chem_decay
                            if new_x != world.shape[0] - 1:
                                world[new_x+1,new_y+1] = world[new_x+1,new_y+1] - chem_decay
                        openmp.omp_unset_lock(&world_lock)
                    entity_map[new_x, new_y] = 2
                    entity_map[x, y] = 0
                    x = new_x
                    y = new_y
                else:
                    if new_x == x:
                        new_x = new_x + 1 - 2*(rand() % (1 + 1 - 0) + 0)
                    elif new_y == y:
                        new_y = new_y + 1 - 2*(rand() % (1 + 1 - 0) + 0)
                    else:
                        if rand() & 1:
                            new_x = x
                        else:
                            new_y = y
                    if entity_map[new_x, new_y] != 2:
                        if entity_map[new_x, new_y] == 1:
                            energy = energy + 4
                            openmp.omp_set_lock(&world_lock)
                            world[new_x,new_y] = world[new_x,new_y] - chem_food
                            if new_x != 0:
                                world[new_x-1,new_y] = world[new_x-1,new_y] - chem_decay
                            if new_x != world.shape[0] - 1:
                                world[new_x+1,new_y] = world[new_x+1,new_y] - chem_decay
                            if new_y != 0:
                                world[new_x,new_y-1] = world[new_x,new_y-1] - chem_decay
                                if new_x != 0:
                                    world[new_x-1,new_y-1] = world[new_x-1,new_y-1] - chem_decay
                                if new_x != world.shape[0] - 1:
                                    world[new_x+1,new_y-1] = world[new_x+1,new_y-1] - chem_decay
                            if new_y != world.shape[1] - 1:
                                world[new_x,new_y+1] = world[new_x,new_y+1] - chem_decay
                                if new_x != 0:
                                    world[new_x-1,new_y+1] = world[new_x-1,new_y+1] - chem_decay
                                if new_x != world.shape[0] - 1:
                                    world[new_x+1,new_y+1] = world[new_x+1,new_y+1] - chem_decay
                            openmp.omp_unset_lock(&world_lock)
                        entity_map[new_x, new_y] = 2
                        entity_map[x, y] = 0
                        x = new_x
                        y = new_y
                openmp.omp_unset_lock(&entity_lock)
            # didn't find any chem signal go to a random place
            else:
                if x == 0:
                    new_x = x + rand() % (1 + 1 - 0) + 0
                elif x == world.shape[0] - 1:
                    new_x = x + rand() % (0 + 1 - -1) + -1
                else:
                    new_x = x + rand() % (1 + 1 - -1) + -1
                if y == 0:
                    new_y = y + rand() % (1 + 1 - 0) + 0
                elif y == world.shape[0] - 1:
                    new_y = y + rand() % (0 + 1 - -1) + -1
                else:
                    new_y = y + rand() % (1 + 1 - -1) + -1
                # if the cell doesn't collide with another cell move
                # move
                openmp.omp_set_lock(&entity_lock)
                if entity_map[new_x, new_y] != 2:
                    entity_map[new_x, new_y] = 2
                    entity_map[x, y] = 0
                    x = new_x
                    y = new_y
                openmp.omp_unset_lock(&entity_lock)
            # cells spend their energy to live
            energy = energy - 1
            # starving or found cAMP
            # food signal goes from 0 to 1, more than 1 is cAMP
            if energy <= 0 or max >= 1:
                openmp.omp_set_lock(&world_lock)
                # generate cluster order signal to form slug
                for k in range(-influence_range, influence_range + 1):
                    for j in range(-influence_range, influence_range + 1):
                        if 0 <= x+k < world.shape[0] and 0 <= y+j < world.shape[1]:
                            if abs(k)<abs(j):
                                world[x+k,y+j] = world[x+k,y+j] + influence_range + 1 - abs(j)
                            else:
                                world[x+k,y+j] = world[x+k,y+j] + influence_range + 1 - abs(k)
                openmp.omp_unset_lock(&world_lock)
            # update cell's info
            dictys[i, 0] = x
            dictys[i, 1] = y
            dictys[i, 2] = energy
    return 0