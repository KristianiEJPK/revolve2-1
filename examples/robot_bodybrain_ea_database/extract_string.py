
def extract_string(string):
    # Get max_parts
    splitted = string.split("|")
    max_parts = int(splitted[0])
    # Only core? or should we get the building plan?
    if max_parts == 1:
        dict_coord = None
    elif len(splitted) == 2:
        # ---- Get coordinate data
        substring = splitted[1]
        substring_split = substring.split("-")
        # Fill dictionary with building plan
        # --> {poslin: ["B" or "H", {attachment_point: rotation_index}]}
        dict_coord = {}
        i = 0
        while (i != len(substring_split)) and (substring_split[i] != ""):
            # Linear coordinate
            coord = int(substring_split[i])
            # Information for that coordinate (type, attachment points and orientations)
            info = substring_split[i + 1]
            # Set type of module (Brick or Hinge)
            dict_coord[coord] = []
            dict_coord[coord].append(info[0])
            # Set attachment points and orientations
            if len(info[1:]) > 1:
                dict_coord[coord].append({})
                for j in range(int((len(info) - 1) / 2)):
                    dict_coord[coord][1][int(info[1 + int(j * 2)])] = int(info[1 + (int(j * 2) + 1)])
            else:
                dict_coord[coord].append({})
            # Increase i
            i += 2
    
    return max_parts, dict_coord