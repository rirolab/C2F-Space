import numpy as np
import cv2

# def get_rect(scene_graph, objects):
#     rect_string = ""
#     for obj_idx, obj in enumerate(objects):
#         bbox = scene_graph["objects_info"][obj]["bbox"]
#         x1, y1, x2, y2 = bbox
#         center = [round((x1 + x2) / 2), round((y1 + y2) / 2)]
#         width = round(x2 - x1)
#         height = round(y2 - y1)
#         extents = '{idx}. {o} : (center = {c}, width = {w}, height = {h})\n'.format(
#             idx=obj_idx, o=obj, c=center, w=width, h=height)
#         rect_string += extents
#     return rect_string

def get_rect(scene_graph, objects):
    rect_string = ""
    for obj_idx, obj in enumerate(objects):
        bbox = scene_graph["objects_info"][obj]["bbox"]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # w = x2 - x1
        # h = y2 - y1
        extents = '{idx}. {o} : (x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2})\n'.format(
            idx=obj_idx, o=obj, x1=x1, y1=y1, x2=x2, y2=y2)
        rect_string += extents
    return rect_string

def get_min_enc_rot_rect(scene_graph, objects, img_x, img_y):

    min_enc_rot_rect_string = ""
    for obj_idx, obj in enumerate(objects):
        mask = np.array(scene_graph["objects_info"][obj]["mask"], dtype=np.uint8)
        mask = mask.reshape(img_y, img_x)
        if is_circular(mask):
            height, width = len(mask), len(mask[0])
            # Initialize extreme points with opposite values to ensure updates
            leftmost, rightmost, bottommost, topmost = width - 1, 0, 0, height - 1
            for y in range(height):
                for x in range(width):
                    if mask[y][x] == 1:  # Check if pixel belongs to the object
                        leftmost = min(leftmost, x)
                        rightmost = max(rightmost, x)
                        bottommost = max(bottommost, y)
                        topmost = min(topmost, y)
            extents = ''     
            extents = '{idx}. {o} : (x = {l}, x = {r}, y = {b}, y = {t})\n'.format(idx=obj_idx, o=obj,\
                        l=leftmost, r=rightmost, b=bottommost, t=topmost)
            min_enc_rot_rect_string += extents
        else:            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the minimum enclosing rotated rectangle
                rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Convert the list of lists into a numpy array
                corners = np.array(box)
                
                # Find the top-left corner (min x + min y)
                top_left = corners[np.argmin(np.sum(corners, axis=1))]

                # Find the bottom-right corner (max x + max y)
                bottom_right = corners[np.argmax(np.sum(corners, axis=1))]

                # Remove top-left and bottom-right to find the other two points
                remaining = np.array([point for point in corners if not np.array_equal(point, top_left) and not np.array_equal(point, bottom_right)])
                
                # Determine which is top-right and bottom-left
                if remaining[0][0] < remaining[1][0]:
                    bottom_left = remaining[0]
                    top_right = remaining[1]
                else:
                    bottom_left = remaining[1]
                    top_right = remaining[0]

                extents = ''
                extents = '{idx}. {o} : (bottom_left = {l}, top_left = {r}, top_right = {b}, bottom_right = {t})\n'.format(idx=obj_idx, o=obj,\
                        l=list(bottom_left), r=list(top_left), b=list(top_right), t=list(bottom_right))
                
                min_enc_rot_rect_string += extents

    return min_enc_rot_rect_string

def is_circular(mask):

    # Assume mask is a binary segmentation mask with the object as white (255) and background as black (0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the object
    contour = max(contours, key=cv2.contourArea)

    # Calculate area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Determine if the object is circular based on a threshold
    is_circular = circularity > 0.85

    return is_circular