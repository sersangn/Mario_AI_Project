from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

# code for locating objects on the screen in super mario bros
# by Lauren Gee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID      = False
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    # "sky":{
    #     "sky":["sky.png"] ###incase i want to try the sky method
    # },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png","koopaC.png", "koopaD.png"],
    },
    "block": {
        "block": ["block1.png", "block3.png"],
        "floorblock":["block2.png"],
        "stair":["block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
        # "sky": ["sky.png"]
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
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

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
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
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ", # sky blue colour
    (0,     0,   0): " ", # black
    (252, 252, 252): "'", # white / cloud colour
    (248,  56,   0): "M", # red / mario colour
    (228,  92,  16): "%", # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()),reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour): # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]
    
    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x+i, y)] = name_str[i%len(name_str)]
                pixels[(x+i, y+height-1)] = name_str[(i+height-1)%len(name_str)]
            for i in range(1, height-1):
                pixels[(x, y+i)] = name_str[i%len(name_str)]
                pixels[(x+width-1, y+i)] = name_str[(i+width-1)%len(name_str)]

    # print the screen to terminal
    print("-"*SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))

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
    
    #      [((x,y), (width,height))]
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

def make_action(screen, info, step, env, prev_action,gap):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # You probably don't want to print everything I am printing when you run
    # your code, because printing slows things down, and it puts a LOT of
    # information in your terminal.

    # Printing the whole grid is slow, so I am only printing it occasionally,
    # and I'm only printing it for debug purposes, to see if I'm locating objects
    # correctly.
    if PRINT_GRID and step % 100 == 0:
        print_grid(screen, object_locations)
        # If printing the grid doesn't display in an understandable way, change
        # the settings of your terminal (or anaconda prompt) to have a smaller
        # font size, so that everything fits on the screen. Also, use a large
        # terminal window / whole screen.

        # object_locations contains the locations of all the objects we found
        print(object_locations)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]


    # List of locations of sky:
    # sky_locations = object_locations["sky"]

    # pipe_locations = []
    # pipe_locations.append(_locate_pipe(screen))

    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
    
    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
            print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")

        # Or you could do it this way:
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
            print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
        print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in object_locations are screen coordinates.
        # Top left corner of screen is (0, 0), top right corner is (255, 0).
        # Here's how you can get Mario's screen coordinates:
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            print("Mario's location on screen:",
                  mario_x, mario_y, f"({object_name} mario)")
        
        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        print("Mario's location in world:",
              mario_world_x, mario_world_y, f"({mario_status} mario)")

    # TODO: Write code for a strategy, such as a rule based agent.

    # Choose an action from the list of available actions.
    # For example, action = 0 means do nothing
    #              action = 1 means press ' ' button
    #              action = 2 means press 'right' and 'A' buttons at the same time
    #check if you're in the middle of a manuever - add something to keep track of 
    #step variable -- tells you what time, step+10
        
    #DEFAULT ACTION
    action = 3
    #-------------------ENEMY LOCATION CODE-------------------#

    ##Code for jumping according to enemy locations
    if len(enemy_locations)>0:
        for enemy in enemy_locations:
            if enemy[0][0]>mario_locations[0][0][0]:
                if(enemy[0][0]-mario_locations[0][0][0]<40) and (enemy[0][1]>=180) and mario_locations[0][0][0] <= enemy[0][0]:
                    action=4

    #-------------------PIPE CODE-------------------#
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
            # print("pipe", pipeindex, block_locations[pipeindex]) 
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
    #got some exception error but program continued... might introduce errors later

    if len(botlevel) <= 13:
        prevblock = botlevel[0]
        for block in botlevel:
            if ((block[0][0] - prevblock[0][0]) !=0 )and ((block[0][0] - prevblock[0][0]!=16)):
                if prevblock[0][0] <= mario_locations[0][0][0] and mario_locations[0][0][0] <= block[0][0]:
                    action= 2
            prevblock= block

    #-------------------DEALING WITH STAIRS CODE-------------------#

    if len(stairs) > 0:
        action = 4
        if step % 20 == 0:
            if prev_action == 4:
                action = 0

    #-------------------DEALING WITH GAPS CODE - SKY METHOD-------------------#

    # print(sky_locations)

    # if len(botlevel) <= 13:
    #     prevblock = botlevel[0]
    #     for block in botlevel:
    #         if ((block[0][0] - prevblock[0][0]) !=0 )and ((block[0][0] - prevblock[0][0]!=16)):
    #             if prevblock[0][0]+32 <= mario_locations[0][0][0] or mario_locations[0][0][0] <= block[0][0]:
    #                 print("GAP HERE")
    #                 action= 4
    #         prevblock= block

    #Trying to deal with four goombas
    # if len(enemy_locations)==3:
    #     for enemy in enemy_locations:
    #         if enemy[0][0]>mario_locations[0][0][0]:
    #             if(enemy[0][0]-mario_locations[0][0][0]<5) and (mario_locations[0][0][0]<enemy[0][0]) and (enemy[0][1]<=193):
    #                 print("HEREEEEE")
    #                 action = 6
    #                 return action


######Stops mario from always pressing jump and getting stuck
#--------Every 10 frames you switch action to 0
    if step % 10 == 0:
        if prev_action == 4 and len(stairs) == 0:
            action = 0
        # print(mario_locations)
    if step % 20 == 0:
        if prev_action == 2:
            action = 0

    # for floors in floor:
    #     if floors[0][0]==0:
    #         print(floor)
    # print("MARIO", mario_locations)
    # print("FLOOR", botlevel)

    ##-----------------------NEW GAP METHOD------------------------#
    ##--------DEALING WITH THIS USING THE LENGTH _ NOT THAT GREAT
    # #####OLD GAP METHOD
    # if len(floor) <= 26:
    #     gap=gap+1
    #     if gap>20:
    #         action=4
    #         print("FLOOR", floor)
    #         print("LENGTH", len(floor))
    #         print("MARIO", mario_locations)
    #         return action, gap
    #     # if floor[0][0][0]==13:
    #     #     # print("here!!!")
    #     #     action=4

    # ##This parameter is good cause it makes mario jump right before the gap somehow
    # if len(floor) > 28:
    #     gap = 0
    
    ####Trying out going left if stuck --> come back to this
    # if step%30==0:
    #     prevlocation= mario_locations
    #     if mario_locations[0][0][0]-prevlocation[0][0][0]<3:
    #         print(prevlocation)
    #         print(mario_locations)
    #         action = 6

    # # if mario_locations[0][0][1]<160:
    # #     action = 0
    # if left:
    
    if mario_locations[0][0][1] <= 66:
        action = 3
    return action, gap

################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state = env.reset()

obs = None
done = True
env.reset()
gap = 0
for step in range(100000):
    if obs is not None:
        gap = make_action(obs, info, step, env, action, gap)[1]
        action = make_action(obs, info, step, env, action, gap)[0]
    else:
        #action = env.action_space.sample()
        action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print("dead")
        gap = 0
        env.reset()
env.close()
