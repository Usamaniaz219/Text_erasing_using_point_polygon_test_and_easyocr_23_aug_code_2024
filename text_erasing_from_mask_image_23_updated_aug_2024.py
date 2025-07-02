import numpy as np
import cv2
import os
import time
import EasyOcr_Bounding_box_into_mask
from EasyOcr_Bounding_box_into_mask import main
from shapely.geometry import Polygon
from shapely.validation import make_valid



retain_contours1 = []

def retain_contours(source_image,mask_image,bounding_boxes):
    image_Gray = cv2.cvtColor(source_image,cv2.COLOR_BGR2GRAY)
    source_image_copy = source_image.copy()
    height, width = image_Gray.shape[:2]
    # Create a blank (black) image
    blank_image = np.zeros((height, width), dtype=np.uint8)
    blank_image1_copy = blank_image.copy()
    thresh1 = cv2.adaptiveThreshold(image_Gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    for box in bounding_boxes:
        # blank_image_bbox = blank_image.copy()
        # blank_image_copy = blank_image.copy()
        box = np.array(box, dtype=np.int32)
        box = box.reshape((-1, 1, 2))
        # cv2.fillPoly(blank_image_bbox, [box], 255)
        # intersections_with_bbox = cv2.bitwise_and(blank_image_bbox,mask_image)
        # intersection_area_bbox = np.sum(intersections_with_bbox)
        # bbox_mask_area = np.sum(blank_image_bbox)
        # print("bbox_mask_intersection_area",bbox_mask_area)
        # if bbox_mask_area==0:
        #     return 0
        # intersection_percentage_bbox = intersection_area_bbox/bbox_mask_area
        # # print("intersection percentage",intersection_percentage_bbox)
        # if intersection_percentage_bbox >=0:
           
        cv2.fillPoly(blank_image, [box], 255)
            # cv2.imwrite("merged_image.jpg",output)
            # cv2.imwrite("blank_image.jpg",blank_image_bbox)
    output = cv2.bitwise_and(thresh1,blank_image)
    
    return output


def text_erasing_using_pointpolygon_text(source_mask_image, text_mask):

    # Apply binary thresholding to the masks
    _, zone_mask = cv2.threshold(source_mask_image, 10, 255, cv2.THRESH_BINARY)
    _, text_mask = cv2.threshold(text_mask, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed_mask_image = cv2.morphologyEx(zone_mask,cv2.MORPH_CLOSE,kernel)
    # Find contours in the masks
    zone_contours, _ = cv2.findContours(closed_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_contours, _ = cv2.findContours(text_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store inner points
    inner_point_list = []

    # Initialize an output mask
    output_mask = np.zeros_like(zone_mask)
    
    # Initialize combined_mask to None
    combined_mask = None

    # Iterate through each text contour
    for text_contour in text_contours:
        pts_tuple_inside_cnt = []

        # Check if each point in the text contour lies within any zone contour
        for zone_contour in zone_contours:
            for pt in text_contour:
                pt_tuple = tuple(int(x) for x in pt[0])

                # Check if the point is inside the zone contour
                result = cv2.pointPolygonTest(zone_contour, pt_tuple, False)
                if result > 0:
                    pts_tuple_inside_cnt.append(pt_tuple)

            # If points were found inside the zone contour, calculate the intersection percentage
            if pts_tuple_inside_cnt:
                intersection_percent = len(pts_tuple_inside_cnt) / len(text_contour)
                if intersection_percent >= 0:
                    pts_contour = np.array(pts_tuple_inside_cnt, dtype=np.int32).reshape((-1, 1, 2))
                    inner_point_list.append(pts_contour)
                # Clear the list after each zone contour check
                pts_tuple_inside_cnt.clear()

    # Draw the resulting contours on the output mask and save the images
    for contour in inner_point_list:
        cnt = np.array(contour, dtype=np.int32)
        cnt = cnt.reshape((-1, 1, 2))
        cv2.fillPoly(output_mask, [cnt], (255, 255, 255))

    # Check if output_mask has any non-zero values before combining
    if np.any(output_mask):
        combined_mask = cv2.bitwise_or(zone_mask, output_mask)
    else:
        combined_mask = zone_mask.copy()  # Default to zone_mask if no contours were found

    return combined_mask, output_mask, text_mask

def process_image(source_image_path, source_mask_path, output_dir):
    ori_image_name = os.path.splitext(os.path.basename(source_image_path))[0]
    print(f"Processing ori image name image: {ori_image_name}")

    source_image = cv2.imread(source_image_path)
    
    # cv2.imwrite("mask-temp.jpg",source_image)
    if source_image is None:
        print(f"Error reading mask image: {source_image_path}")
        return None
    
 
    mask_image = cv2.imread(source_mask_path)

    if mask_image is None:
        print(f"Error reading bounding box image: {source_mask_path}")
        return None
    mask_image = cv2.cvtColor(mask_image,cv2.COLOR_BGR2GRAY)
    bounding_boxes, output_image = main(source_image, tile_width, tile_height)
    output = retain_contours(source_image,mask_image,bounding_boxes)
    # intersected_contours = retain_intersected_contours(retained_contours,mask_image)
    # intersected_contours = retain_intersected_contours(retained_contours,mask_image)
    combined_mask,output_text_mask,text_pixel_mask= text_erasing_using_pointpolygon_text(mask_image,output)

    # if intersected_contours is not None:
    #     print("intersected_retained_contours",len(intersected_contours))
    #     print("Length of retained Contours",len(retained_contours))
    # merged_result,text_mask = draw_intersected_contours_on_mask_image(mask_image,intersected_contours)
    # if intersected_contours is not None:
    #     intersected_contours.clear()
    # if retained_contours is not None: 
    #     retained_contours.clear()
    return combined_mask,output_text_mask,text_pixel_mask



def process_images(input_dir, output_dir, bounding_box_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    file_count = 0
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # original_file_list.append(filename)
                file_count += 1
                image_path = os.path.join(input_dir, filename)
                ori_image_name = os.path.splitext(os.path.basename(image_path))[0]
                print("mask image name :",ori_image_name)
            
                for root,dirs, files in os.walk(bounding_box_dir):              
                    # all_masks = os.listdir(bounding_box_dir)
                    for dir in dirs:
                        if dir==ori_image_name:
                            dir1 = os.path.join(root,dir)
                            # mask_dirs.append(dir1)
                            
                            all_masks = os.listdir(dir1)
                            masks_renamed = [mask.replace(".jpg","").replace(".png","") for mask in all_masks]

                            for renamed_mask in masks_renamed:
                                mask_path = f"{bounding_box_dir}/{dir}/{renamed_mask}.jpg"
                                output_,text_mask,text_pixel_mask = process_image(image_path, mask_path, output_dir)
                                if output_ is None:
                                    continue

                                # output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
                                output_subdir = os.path.join(output_dir, ori_image_name)
                                # output_subdir = f"{output_subdir}_{renamed_mask}_intersection_of_0.1"
                                os.makedirs(output_subdir, exist_ok=True)
                                
                                output_file_path = os.path.join(output_subdir, f"{renamed_mask}_output_mask.jpg")
                                output_file_path1 = os.path.join(output_subdir,f"{renamed_mask}_text_mask.jpg")
                                output_file_path2 = os.path.join(output_subdir,f"{renamed_mask}_text_pixel_mask.jpg")
                                cv2.imwrite(output_file_path,output_)
                                cv2.imwrite(output_file_path1,text_mask)
                                cv2.imwrite(output_file_path2,text_pixel_mask)
                                
                                print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")
                        continue
                    break    



if __name__ == "__main__":
    tile_width = 1024
    tile_height = 1024


    # input__image_directory = '/home/usama/Data_13_aug_2024_temp_latest/A/'
    # mask_image_dir = '/home/usama/Data_13_aug_2024_temp_latest/B/'
    # output_directory = '/media/usama/6EDEC3CBDEC389B3/22_aug_2024_results_221/'
    
    process_images(input__image_directory, output_directory, mask_image_dir)


