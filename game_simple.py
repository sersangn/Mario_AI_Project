from PIL import Image
from numpy import asarray
import numpy as np

image = Image.open('block2.png')

data = asarray(image)

data = np.dot(data[..., :3],[.2989, .5870, .1140])
data = np.round(data).astype(np.uint8)

BLOCK2 = data

def get_block():
    return BLOCK2

def get_simple_game_state(state):
    simple_game_state = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    
    x = 0
    y = 0
    
    for i in range(0,240,16):
        
        for j in range(0,256,16):
            print(x, y)
            
            tile = state[0]
            tile = tile[i:i+16, j:j+16]
            if (tile == BLOCK2).all():
                simple_game_state[x][y] = 1
                
            y = y+1
            
        x = x+1
        y=0
    
    
    # tile = np.asarray(state[0])
    # tile = tile[224:240, 16:32]
    
    
    # print(tile)
    simple_game_state = np.asarray(simple_game_state)
    print(simple_game_state)
    return simple_game_state
    
    # print(simple_game_state)
    return simple_game_state
    
    
# print(BLOCK2)