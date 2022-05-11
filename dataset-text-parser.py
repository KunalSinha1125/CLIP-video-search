
# this function parses AllVideoDescriptions and outputs a dictionary
# which has the name of the video as key and a list of string descriptions as values

text_dir = "AllVideoDescriptions.txt"

def export_descriptions():
    vid2tex = {}
    with open(text_dir) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#' or line.isspace():
                continue
            splits = line.split()
            vid_name = splits[0]
            if vid_name not in vid2tex.keys():
                vid2tex[vid_name] = [' '.join(splits[1:])]
            else:
                vid2tex[vid_name].append(' '.join(splits[1:]))
    return vid2tex