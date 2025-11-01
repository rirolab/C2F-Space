import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import cv2
from skimage import measure
import math
from copy import deepcopy

def add_grid(img, grid_size, grid_color, grid_lines, save_path=None):
    """Function to add grids to the image.

    Args:
        img (numpy.ndarray): RGB image array.
        grid_size (int): Size of the grid in pixels.
        grid_color (list): RGB values for the color of the grid.
        grid_lines (bool): grid_lines on the image to be drawn or not drawn

    Returns:
        numpy.ndarray: Image with grids added
    """
    
    # Extract the image size
    if grid_size != 0:

        img_height, img_width, _ = np.shape(img)

        # Create a figure with the same size as the original image
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

        # Display the image
        ax.imshow(img)
        # Set the grid with the provided color and size
        ax.set_xticks(np.arange(0, img_width, grid_size))
        ax.set_yticks(np.arange(0, img_height, grid_size))
        
        # Set the grid color
        if grid_lines:
            ax.grid(color=grid_color, linestyle='-', linewidth=1)
        else:
            ax.grid(visible=False)

        # Label the axes with larger font size
        ax.set_xlabel('x axis', fontsize=20)  # Set a larger font size for the x-axis label
        ax.set_ylabel('y axis', fontsize=20)  # Set a larger font size for the y-axis label

        # Increase the size of the tick labels (axis numbers)
        ax.tick_params(axis='x', labelsize=15)  # Set larger size for x-axis numbers
        ax.tick_params(axis='y', labelsize=15)  # Set larger size for y-axis numbers

        # Save the image with grids
        if save_path is None:
            if grid_lines:
                save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grids.png"
            else:
                save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grid_without_gridlines.png"
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.25)

        # Close the figure to free memory
        plt.close(fig)
    else:
        if save_path is None:
            if grid_lines:
                save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grids.png"
            else:
                save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grid_without_gridlines.png"
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp")
        plt.imsave(save_path, img)

def add_grid_for_val(img, grid_size, grid_color, save_path=None):
    """Function to add grids to the image.

    Args:
        img (numpy.ndarray): RGB image array.
        grid_size (int): Size of the grid in pixels.
        grid_color (list): RGB values for the color of the grid.
        grid_lines (bool): grid_lines on the image to be drawn or not drawn

    Returns:
        numpy.ndarray: Image with grids added
    """
    if grid_size != 0:
        # Extract the image size
        img_height, img_width, _ = np.shape(img)

        # Create a figure with the same size as the original image
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

        # Display the image
        ax.imshow(img)

        # Set the grid with the provided color and size
        ax.set_xticks(np.arange(0, img_width, grid_size))
        ax.set_yticks(np.arange(0, img_height, grid_size))
        
        # Set the grid color
        ax.grid(color=grid_color, linestyle='-', linewidth=1)


        # Label the axes with larger font size
        ax.set_xlabel('x axis', fontsize=20)  # Set a larger font size for the x-axis label
        ax.set_ylabel('y axis', fontsize=20)  # Set a larger font size for the y-axis label

        # Increase the size of the tick labels (axis numbers)
        ax.tick_params(axis='x', labelsize=15)  # Set larger size for x-axis numbers
        ax.tick_params(axis='y', labelsize=15)  # Set larger size for y-axis numbers

        # Save the image with grids    
        if save_path is None:
            save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grids_for_val.png"
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.25)

        # Close the figure to free memory
        plt.close(fig)
    else:
        if save_path is None:
            save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grids_for_val.png"
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp")
        plt.imsave(save_path, img)

def add_rel_grid_for_val(img, grid_size, grid_color, center_point, save_path=None):
    """Function to add grids to the image.

    Args:
        img (numpy.ndarray): RGB image array.
        grid_size (int): Size of the grid in pixels.
        grid_color (list): RGB values for the color of the grid.
        grid_lines (bool): grid_lines on the image to be drawn or not drawn

    Returns:
        numpy.ndarray: Image with grids added
    """
    if grid_size != 0:
        # Extract the image size
        img_height, img_width, _ = np.shape(img)
        cx, cy = center_point

        # Create a figure with the same size as the original image
        fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

        # Display the image
        ax.imshow(img)

        x_offset = math.ceil((cx) / grid_size)
        x_start = cx - x_offset * grid_size
        x_end = cx + math.ceil((img_width - cx) / grid_size) * grid_size
        x_positions = np.arange(x_start, x_end + 1, grid_size)
        for x in x_positions:
            ax.axvline(x=x, color=grid_color, linestyle='-', linewidth=1)
        
        y_offset = math.ceil((cy) / grid_size)
        y_start = cy - y_offset * grid_size
        y_end = cy + math.ceil((img_height - cy) / grid_size) * grid_size
        y_positions = np.arange(y_start, y_end + 1, grid_size)
        for y in y_positions:
            ax.axhline(y=y, color=grid_color, linestyle='-', linewidth=1)
        
        valid_x_ticks = [x for x in x_positions if 0 <= x <= img_width]
        x_labels = [f"{x - cx:.0f}" for x in valid_x_ticks]
        ax.set_xticks(valid_x_ticks)
        ax.set_xticklabels(x_labels, fontsize=15)
        ax.set_xlabel('x axis (relative to center)', fontsize=20)

        valid_y_ticks = [y for y in y_positions if 0 <= y <= img_height]
        y_labels = [f"{y - cy:.0f}" for y in valid_y_ticks]
        ax.set_yticks(valid_y_ticks)
        ax.set_yticklabels(y_labels, fontsize=15)
        ax.set_ylabel('y axis (relative to center)', fontsize=20)
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        
        # Save the image with grids    
        if save_path is None:
            save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grids_for_val.png"
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.25)

        # Close the figure to free memory
        plt.close(fig)
    else:
        if save_path is None:
            save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp/grids_for_val.png"
            if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp"):
                os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/../../output/tmp")
        plt.imsave(save_path, img)


def save_image(img, stage, datum_id, folder_name):
    """Function to save image

    Args:
        img (numpy.ndarray): RGB image array.
        stage (str): Stage of the pipeline from where the RGB image is from.
        datum_id (int): Unique id given to each image.
        folder_name (str): dataset name 
    """
    crnt_file_path = os.path.dirname(os.path.abspath(__file__))
    output_path = crnt_file_path + '/../../'
    save_dir = output_path + folder_name +  'pipeline_outputs/{}'.format(datum_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        
    
    else:
        img = Image.fromarray(img)
        img.save(save_dir+'/{s}.png'.format(s=stage))

def draw_coordinates_on_image_with_labels(image_path, coordinates, point_size, point_color, label_size):
    """
    Draws a list of coordinates on an image with labels inside the circle.

    Parameters:
        image_path (str): Path to the image.
        coordinates (list): List of coordinates in the format [[X1, Y1], [X2, Y2], ...].
        point_size (int): Size of the circle around the points.
        point_color (tuple): Color of the points and circle in BGR format.
        label_size (int): Font scale for the labels.

    Returns:
        numpy.ndarray: The image with the drawn points and labels.
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be found.")

    # Iterate through coordinates and draw them on the image
    for i, (x, y) in enumerate(coordinates):
        # Draw a circle around the point
        cv2.circle(image, (int(x), int(y)), point_size, point_color, 2)

        # Put the label inside the circle
        label = str(i + 1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_size, 2)[0]
        text_x = int(x) - text_size[0] // 2
        text_y = int(y) + text_size[1] // 2
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    label_size, point_color, thickness=2)
        
        # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def draw_coordinates_on_image(image_path, coordinates, point_size, point_color):
    """
    Draws a list of semi-transparent filled circles at the specified coordinates on an image.

    Parameters:
        image_path (str): Path to the image.
        coordinates (list): List of coordinates in the format [[X1, Y1], [X2, Y2], ...].
        point_size (int): Radius of the filled circles.
        point_color (tuple): Color of the points in BGR format.
        alpha (float): Transparency factor for the circles (0.0 fully transparent, 1.0 fully opaque).

    Returns:
        numpy.ndarray: The image with the drawn filled circles.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be found.")

    # Create an overlay image for drawing the circles
    overlay = image.copy()

    # Iterate through coordinates and draw filled circles on the overlay
    for x, y in coordinates:
        cv2.circle(overlay, (int(x), int(y)), point_size, point_color, thickness=-1)

    alpha = 0.7
    # Blend the overlay with the original image using the alpha value
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def draw_ellipse_on_image(image_path, coordinates, axes_length, angle, point_color):
    """
    Draws a list of semi-transparent filled circles at the specified coordinates on an image.

    Parameters:
        image_path (str): Path to the image.
        coordinates (list): List of coordinates in the format [[X1, Y1], [X2, Y2], ...].
        axes_length (list): Radius of the filled circles.
        point_color (tuple): Color of the points in BGR format.
        alpha (float): Transparency factor for the circles (0.0 fully transparent, 1.0 fully opaque).

    Returns:
        numpy.ndarray: The image with the drawn filled circles.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at path {image_path} could not be found.")

    # Create an overlay image for drawing the circles
    overlay = image.copy()

    # Iterate through coordinates and draw filled circles on the overlay
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        a, b = axes_length[i]
        cv2.ellipse(overlay, (int(x), int(y)), (int(a), int(b)), angle[i], 0, 360, point_color, thickness=-1)

    alpha = 0.7
    # Blend the overlay with the original image using the alpha value
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def save_final_image(iter_num, datum_id, folder_name, spatial_instruction):
    """Function to save the final image 

    Args:
        strat (str): Stage of the pipeline from where the RGB image is from.
        N (int): max number of iterations
        datum_id (int): Unique id given to each image-instruction pair.
        folder_name (str): dataset name
        spatial_instruction (str): input instruction
    """
    crnt_file_path = os.path.dirname(os.path.abspath(__file__))
    output_path = crnt_file_path + '/../../' + folder_name

    # Read the last iter image and save the image
    pth = output_path + "pipeline_outputs/{}/generation_output_iter_{}.png".format(datum_id, iter_num)
    img = cv2.imread(pth)
            
    # Create a border around the image and add the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1

    # Determine the maximum width for the text
    image_height, image_width, _ = np.shape(img)
    max_text_width = image_width - 100  

    # Wrap the text to fit within the image width
    wrapped_text = wrap_text(spatial_instruction, max_text_width)

    # Calculate the total height needed for the text
    line_height = cv2.getTextSize('Test', font, font_scale, thickness)[0][1] \
                                    + 10  # Adding some line spacing
    total_text_height = len(wrapped_text) * line_height

    # Add a border with sufficient space for the wrapped text
    top_border = total_text_height + 20  # Add some padding
    bottom_border = 50  # Bottom border 
    left_border = 50  # Left border
    right_border = 50  # Right border

    bordered_image = cv2.copyMakeBorder(img, top=top_border, \
        bottom=bottom_border, left=left_border, right=right_border, \
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Step 5: Add the wrapped text to the bordered image
    y = top_border - total_text_height + line_height - 10  # Initial y position

    for line in wrapped_text:
        cv2.putText(bordered_image, line, (left_border, y), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height  # Move to the next line
    save_dir = output_path+"final/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        
    cv2.imwrite(save_dir + datum_id + ".png", bordered_image)

def wrap_text(text, max_width):
    """Function to wrap the text into a new line if it is long

    Args:
        text (str): text to be wrapped
        max_width (int): maximum width for the text

    Returns:
        str: wrapped text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1

    words = text.split(' ')
    lines = []
    current_line = ''
    
    for word in words:
        test_line = current_line + word + ' '
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width > max_width:
            lines.append(current_line)
            current_line = word + ' '
        else:
            current_line = test_line
    
    lines.append(current_line)  # add the last line
    return lines

def overlay_boundaries_on_image(image, scene_graph, objects_in_collision_with):    
    
    # List of colors for overlays (distinct and not red)
    colors = [

        (0, 255, 102)
    ]
    
    # Copy the original image to avoid modifying it directly
    overlay_image = deepcopy(image)

    for obj in objects_in_collision_with:
        mask = np.array(scene_graph['objects_info'][obj]['mask'])
        mask = np.squeeze(mask)
        # Ensure we have enough colors for all masks
        color = colors[0]
        
        # Find contours for the current mask
        contours = measure.find_contours(mask)
        
        for contour in contours:
            contour = np.round(contour).astype(int)
            contour = contour[:, ::-1]  # Switch (row, col) to (x, y)
            cv2.polylines(overlay_image, [contour], isClosed=True, color=color, thickness=5)
            break

    return overlay_image

def plot_polygon_mask(img, polygon_coords):
    """Function to add polygon mask over the image.

    Args:
        img (numpy.ndarray): RGB image array.
        polygon_coords (list): List containing the polygon coordinates.

    Returns:
        numpy.ndarray: Image with polygon mask added
    """
    n = len(polygon_coords)
    #normalised_color = math.trunc(255/n)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Generate masks for the given polygon coordinates
    for coordinates in polygon_coords:
        cent=(sum([p[0] for p in coordinates])/len(coordinates),\
             sum([p[1] for p in coordinates])/len(coordinates))
        coordinates.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
        coordinates = [tuple(coordinate) for coordinate in coordinates]
        blank_img = Image.new('L', (img.shape[1], img.shape[0]))
        ImageDraw.Draw(blank_img).polygon(coordinates, fill=1)
        mask += np.array(blank_img)
    
    # Define the green color for the mask
    mask_color = np.array([255, 0, 12])

    # Create a copy of the original image
    new_img = deepcopy(img)

    alpha = 0.6

    # Apply the mask to the image
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] != 0:
                new_img[i, j, :] = (1 - alpha) * img[i, j, :] + alpha * mask_color

    return new_img

def find_collision_with_objects_point(img, scene_graph, region_center, region_radius):

    height = img.shape[0]
    width = img.shape[1]
  
    blank_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(blank_img)
    
    cx, cy = region_center[0]
    left   = cx - region_radius
    top    = cy - region_radius
    right  = cx + region_radius
    bottom = cy + region_radius
    
    draw.ellipse([left, top, right, bottom], fill=1)
    
    circle_mask = np.array(blank_img, dtype=np.uint8)
    
    objects_in_collision_with = []

    for key, value in scene_graph["objects_info"].items():

        object_mask = np.array(value['mask'])
        intersection_area = np.logical_and(object_mask, circle_mask).sum()
        object_mask_area = object_mask.sum()
        region_mask_area = circle_mask.sum()
        overlap_with_object = (intersection_area / object_mask_area) * 100
        overlap_with_region = (intersection_area / region_mask_area) * 100
        if overlap_with_object >=2 or overlap_with_region >= 2:
            objects_in_collision_with.append(key)
        else:
            pass

    return objects_in_collision_with

def find_collision_with_objects_ellipse(img, scene_graph, region_center, axes_length, angle):

    height = img.shape[0]
    width = img.shape[1]
    
    circle_mask = np.zeros((height, width), dtype=np.uint8)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    for i in range(len(angle)):
    
        theta = np.deg2rad(angle[i])
        cx, cy = region_center[i]
        a, b = axes_length[i]
        
        # 좌표를 중심 기준으로 이동
        x_shift = x - cx
        y_shift = y - cy

        # 회전된 좌표계로 변환
        x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
        y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)

        # 타원 방정식: (x_rot / a)^2 + (y_rot / b)^2 <= 1
        mask = ((x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1)

        # 마스크로 이미지 생성
        circle_mask[mask] = 1
        

    # if len(region_center) == 1 and len(region_center[0]) == 2:
    #     cx, cy = region_center[0]
    # else:
    #     cx, cy = 0, 0
    
    # if len(axes_length) == 1 and len(axes_length[0]) == 2:
    #     a, b = axes_length[0]
    # else:
    #     a, b = 0, 0
    
    # # a, b = axes_length[0]
    # # a /= 2
    # # b /= 2

    # # 좌표를 중심 기준으로 이동
    # x_shift = x - cx
    # y_shift = y - cy

    # # 회전된 좌표계로 변환
    # x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
    # y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)

    # # 타원 방정식: (x_rot / a)^2 + (y_rot / b)^2 <= 1
    # mask = ((x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1)

    # # 마스크로 이미지 생성
    # circle_mask[mask] = 1
    
    
    objects_in_collision_with = []

    for key, value in scene_graph["objects_info"].items():

        object_mask = np.array(value['mask'])
        intersection_area = np.logical_and(object_mask, circle_mask).sum()
        object_mask_area = object_mask.sum()
        region_mask_area = circle_mask.sum()
        overlap_with_object = (intersection_area / object_mask_area) * 100
        overlap_with_region = (intersection_area / region_mask_area) * 100
        if overlap_with_object >=2 or overlap_with_region >= 2:
            objects_in_collision_with.append(key)
        else:
            pass

    return objects_in_collision_with

def find_collision_with_objects_polygon(img, scene_graph, polygon_coords):

    polygon_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Generate masks for the given polygon coordinates
    for coordinates in polygon_coords:
        cent=(sum([p[0] for p in coordinates])/len(coordinates),\
                sum([p[1] for p in coordinates])/len(coordinates))
        coordinates.sort(key=lambda p: math.atan2(p[1]-cent[1],p[0]-cent[0]))
        coordinates = [tuple(coordinate) for coordinate in coordinates]
        blank_img = Image.new('L', (img.shape[1], img.shape[0]))
        ImageDraw.Draw(blank_img).polygon(coordinates, fill=1)
        polygon_mask += np.array(blank_img)
    
    objects_in_collision_with = []

    for key, value in scene_graph["objects_info"].items():

        object_mask = np.array(value['mask'])
        intersection_area = np.logical_and(object_mask, polygon_mask).sum()
        object_mask_area = object_mask.sum()
        region_mask_area = polygon_mask.sum()
        overlap_with_object = (intersection_area / object_mask_area) * 100
        overlap_with_region = (intersection_area / region_mask_area) * 100
        if overlap_with_object >=2 or overlap_with_region >= 2:
            objects_in_collision_with.append(key)
        else:
            pass

    return objects_in_collision_with