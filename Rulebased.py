from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np

# Rule-Based Agent implementation for Gym Super Mario Bros World 1 Level 1

# 23599356 Sersang Ngedup
# 22957747 Kirsty Straiton

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

#Screen Height, width and match threshold constants
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block3.png"],
        "floorblock":["block2.png"],
        "stair":["block4.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def make_action(screen, info, step, env, prev_action):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]
    

################Above code was written by Lauren Gee and modified by us##########################


################ Below code is our rule based agent ##########################

    #DEFAULT ACTION
    action = 3
    #-------------------ENEMY LOCATION CODE-------------------#

    ##Code for jumping according to enemy locations
    if len(enemy_locations)>0:
        for enemy in enemy_locations:
            if enemy[0][0]>mario_locations[0][0][0]: ##dont need this
                if(enemy[0][0]-mario_locations[0][0][0]<40) and (enemy[0][1]>=180) and mario_locations[0][0][0] <= enemy[0][0]:
                    action=4

    #-------------------PIPE CODE------------------------------#
    pipeindex = 0
    pipelist = []
    pipe = False
    for index, element in enumerate(block_locations):
            if 'pipe' in element:
                pipe = True
                pipelist.append(index)

    #Code for jumping with pipe
    if pipe:
        for pipeindex in pipelist:
            if block_locations[pipeindex][0][0]<mario_locations[0][0][0]:
                continue
            if(block_locations[pipeindex][0][0]-mario_locations[0][0][0]<30):
                action=2

    #-------------------DEALING WITH GAPS CODE-------------------#
    botlevel = []
    stairs = []
    for block in block_locations:
        if block[2]=="floorblock" and block[0][1]==224:
            botlevel.append(block)
        if block[2]=="stair":
            stairs.append(block)

    if len(botlevel) <= 13:
        prevblock = botlevel[0]
        for block in botlevel:
            if ((block[0][0] - prevblock[0][0]) !=0 )and ((block[0][0] - prevblock[0][0]!=16)):
                if prevblock[0][0] <= mario_locations[0][0][0] and mario_locations[0][0][0] <= block[0][0]:
                    # print("GAP!!!")
                    action= 2
            prevblock= block

    #-------------------DEALING WITH STAIRS CODE-------------------#

    if len(stairs) > 0:
        for stair in stairs:
            if stair[0][1] == 192 and stair[0][0]>mario_locations[0][0][0]:
                if(stair[0][0]-mario_locations[0][0][0]<40) and mario_locations[0][0][0] <= stair[0][0]:
                    action=4

        if step % 20 == 0:
            if prev_action == 4:
                action = 0

######Stops mario from always pressing jump and getting stuck
#--------Every 10 frames you switch action to 0
    if step % 10 == 0:
        if prev_action == 4 and len(stairs) == 0:
            action = 0
        # print(mario_locations)
    if step % 20 == 0:
        if prev_action == 2:
            action = 0
    
    #Stops mario from flying off screen when he's at the top of the stairs
    if mario_locations[0][0][1] <= 66:
        action = 3
    
    return action

####################Implementing the rule based agent#################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()

obs = None
done = True
env.reset()
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        env.reset()
env.close()
