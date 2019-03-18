import cells_world
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from profilehooks import profile

@profile
def run_world(world, entity_map, dictys, num_threads = 2):
    cells_world.run_world(world, entity_map, dictys, num_threads)
    return world, entity_map, dictys

m = 128
num_threads = 6
world = np.zeros((m,m))
entity_map = np.zeros((m,m), dtype=np.intc)
n_dictys = 1024
n_food = 1024
starting_energy = 10
dictys = np.full((n_dictys,3),starting_energy, dtype=np.intc)

# initialize
np.put(entity_map,np.random.choice(np.where(entity_map.flatten() == 0)[0], n_food, replace=False),1)
chem_decay = 0.01
chem_food = 0.1
for x, y in np.argwhere(entity_map == 1):
    world[x,y] = chem_food
    if x != 0:
        world[x-1,y] = world[x-1,y] + chem_decay
    if x != world.shape[0] - 1:
        world[x+1,y] = world[x+1,y] + chem_decay
    if y != 0:
        world[x,y-1] = world[x,y-1] + chem_decay
        if x != 0:
            world[x-1,y-1] = world[x-1,y-1] + chem_decay
        if x != world.shape[0] - 1:
            world[x+1,y-1] = world[x+1,y-1] + chem_decay
    if y != world.shape[1] - 1:
        world[x,y+1] = world[x,y+1] + chem_decay
        if x != 0:
            world[x-1,y+1] = world[x-1,y+1] + chem_decay
        if x != world.shape[0] - 1:
            world[x+1,y+1] = world[x+1,y+1] + chem_decay
np.put(entity_map,np.random.choice(np.where(entity_map.flatten() == 0)[0], n_dictys, replace=False),2)
dictys[:,:-1] = tuple(zip(*np.where(entity_map == 2)))

plt.ion()
fig_world = plt.figure()
ax_world = fig_world.add_subplot(111)
ax_world.imshow(world)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(entity_map)
plt.draw()
# for humans to see, cells are too fast
plt.pause(0.3)

for i in range(50):

    # update
    world, entity_map, dictys = run_world(world, entity_map, dictys, num_threads)

    ax = fig.add_subplot(111)
    ax.imshow(entity_map)
    plt.draw()
    # plt.pause(0.1)
# time.sleep(5)