import numpy as np


def generate_circle_of_fifth_distances():
    """
    distances from the note 0
    :return:
    """
    dist_dict = {0:0}
    curr_dist = 0
    curr_notes = [0]
    next_notes = []
    while len(dist_dict) < 12:
        curr_dist += 1
        for note in curr_notes:
            if (((note - 5) % 12) not in next_notes):
                next_notes += [((note - 5) % 12)]
            if (((note + 5) % 12) not in next_notes):
                next_notes += [((note + 5) % 12)]
            if (((note - 7) % 12) not in next_notes):
                next_notes += [((note - 7) % 12)]
            if (((note + 7) % 12) not in next_notes):
                next_notes += [((note + 7) % 12)]
        curr_notes = [item for item in next_notes]
        for note in curr_notes:
            if note not in dist_dict.keys():
                dist_dict[note] = curr_dist
        next_notes = []
    return dist_dict


def scalar_COF_metric(x, y):
    a = x % 12
    b = y % 12
    dist_dict = generate_circle_of_fifth_distances()
    key = abs(a - b)
    return dist_dict[key]

def harmonic_metric(x, y):
    a = x % 12
    b = y % 12
    dist_dict = {0:0, 1:15, 2:10, 3:3, 4:2, 5:1, 6:20, 7:1, 8:2, 9:3, 10:10, 11:15}
    key = abs(a - b)
    return dist_dict[key]
