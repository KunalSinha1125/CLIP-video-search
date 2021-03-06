
# this function parses AllVideoDescriptions and outputs a dictionary
# which has the name of the video as key and a list of string descriptions as values

text_dir = "AllVideoDescriptions.txt"

def export_descriptions():
    vid2tex = {}
    tex2vid = {}
    with open(text_dir) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#' or line.isspace():
                continue
            splits = line.split()
            vid_name = splits[0]
            vid_des = ' '.join(splits[1:])
            if vid_name not in vid2tex.keys():
                vid2tex[vid_name] = [vid_des]
            else:
                vid2tex[vid_name].append(vid_des)
            if vid_des not in tex2vid.keys():
                tex2vid[vid_des] = [vid_name]
            else:
                tex2vid[vid_des].append(vid_name)
    return vid2tex, tex2vid
