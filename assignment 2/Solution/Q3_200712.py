

import cv2
import numpy as np
def boundary_error(rectangles):
    rectangle=sorted(rectangles, key=lambda x: x[0])
    distances=[]
    n=len(rectangle)
    for i in range(n-1):
        dist=abs(rectangle[i][2]-rectangle[i+1][0])
        distances.append(dist)
    distances=sorted(distances)
    avg=0
    for i in range(len(distances)-1):
        avg=avg+distances[i]
    avg=avg/(len(distances)-1)
    if (distances[len(distances)-1]>(4*avg)) and distances[len(distances)-1]>15:
        return False
    else:
        return True

def calculateAreaRectangle(rect):
    l0=rect[0]
    l1=rect[1]
    l2=rect[2]
    l3=rect[3]
    ans=(l2-l0)*(l3-l1)
    return ans

def calculateSumOfRect(square):
    square=sorted(square)
    n=len(square)
    ans=0
    for i in range(n):
        ans=ans+square[i]
    return ans

def isRavanOneReal(square):
    n=len(square)
    areaSqua1=0
    areaSqua2=0
    areaSquaIndex1=0
    areaSquaIndex2=0
    Squa1=[]
    Squa2=[]
    for i in range(n):
        currArea=calculateAreaRectangle(square[i])
        if currArea>=areaSqua1:
            areaSqua2=areaSqua1
            areaSquaIndex2=areaSquaIndex1
            Squa2=Squa1
            areaSqua1=currArea
            Squa1=square[i]
            areaSquaIndex1=i
        elif currArea>=areaSqua2:
            areaSqua2=currArea
            Squa2=square[i]
            areaSquaIndex2=i
    leftFaces=0
    rightFaces=0
    for i in range(n):
        if i!=areaSquaIndex1 and i!=areaSquaIndex2:
            if square[i][0]<square[areaSquaIndex1][0]:
                leftFaces=leftFaces+1
            else:
                rightFaces=rightFaces+1
    if leftFaces!=3 or rightFaces!=4:
        return False
    else:
        return True

def remove_duplicates(input_array):
    input_array = np.array(input_array)
    unique_elements = np.unique(input_array)
    return unique_elements
def soort(rect):
    for i in range(len(rect)):
        for j in range(i+1,len(rect)):
            t=rect[i]
            rect[i]=rect[j]
            rect[j]=t
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)

    return merge(left_half, right_half)

def merge(left, right):
    result = np.array([], dtype=left.dtype)
    left_index, right_index = 0, 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            result = np.append(result, left[left_index])
            left_index += 1
        else:
            result = np.append(result, right[right_index])
            right_index += 1

    result = np.append(result, left[left_index:])
    result = np.append(result, right[right_index:])
    return result

def isRavanTwoReal(rectangle):
    rectangle_sorted = sorted(rectangle, key=lambda x: x[0])
    dist=0
    index=0
    for i in range(1,len(rectangle_sorted)):
        if (rectangle_sorted[i][0]-rectangle_sorted[i-1][0])>=dist:
            dist=rectangle_sorted[i][0]-rectangle_sorted[i-1][0]
            index=i
    if index!=5 or len(rectangle_sorted)!=11:
        return rectangle_sorted,False
    else:
        return rectangle_sorted,True
    
def is_subset(rect1,rect2):
    a=rect1[0]>=rect2[0]
    b=rect1[2]<=rect2[2]
    if a and b:
        return True
    else:
        return False
    
def remove_largest(square):
    i=0
    maximum_area=0
    for j in range(len(square)):
        current_area=(square[j][2]-square[j][0])*(square[j][3]-square[j][1])
        if current_area>=maximum_area:
            maximum_area=current_area
            i=j
    answer=[]
    for j in range(len(square)):
        if j!=i:
            answer.append(square[j])
    return answer

def process_rectangle(square):
    ans=[]
    n=len(square)
    for i in range(n):
        flagBool=True
        for j in range(n):
            if j!=i and is_subset(square[i],square[j]):
                flagBool=False
                break
        if flagBool==True:
            ans.append(square[i])
    return ans

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image

def k_means_segmentation(image, k=3):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    final=cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels=final[1]
    centers=final[2]
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image, labels, centers


def apply_watershed(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255] 
    return image


def mask_darkest_segment(image, labels, centers):
    darkest_segment_index = np.argmin(np.linalg.norm(centers, axis=1))
    labels_reshaped = labels.reshape(image.shape[0], image.shape[1])
    darkest_segment_mask = labels_reshaped == darkest_segment_index
    darkest_segment_mask = darkest_segment_mask.astype(np.uint8)
    inverse_mask = 1 - darkest_segment_mask
    white_image = np.ones_like(image) * 255
    masked_image = cv2.bitwise_and(white_image, white_image, mask=inverse_mask)
    return masked_image

def apply_median_filter(image, kernel_size=3):
    image_rgb=image
    median_filtered_image = cv2.medianBlur(image_rgb, kernel_size)
    return image_rgb, median_filtered_image

def apply_dilation(image,kernel_size=(3,3)):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return image, eroded_image

def apply_erosion(image, kernel_size=(3, 3)):
    kernel=np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]],dtype=np.uint8)
    eroded_image = cv2.dilate(image, kernel, iterations=5)
    return image, eroded_image


def apply_erosion2(image, kernel_size=(3, 3)):
    kernel=np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]],dtype=np.uint8)
    eroded_image = cv2.dilate(image, kernel, iterations=1)
    return image, eroded_image

def count_connected_components(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_with_boxes = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    rectan=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectan.append([x,y,x+w,y+h])
    rectan=remove_largest(rectan)
    rectan=process_rectangle(rectan)
    for rect in rectan:
        cv2.rectangle(image_with_boxes, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
    return image,image_with_boxes,rectan

def is_ravan_green(image):
    lower_green = np.array([40, 40, 40])  
    upper_green = np.array([80, 255, 255]) 
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    green_pixel_count = cv2.countNonZero(green_mask)
    thresholdGreen = 100  
    green_ravan = green_pixel_count > thresholdGreen
    return green_ravan


def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    class_name = 'fake'
    all_images=[]
    all_images_name=[]
    image_path=audio_path
    image = cv2.imread(image_path)
    whichRavan=is_ravan_green(image)
    if whichRavan:
        k_value = 4 
        segmented_image, labels, centers = k_means_segmentation(image, k_value)
        all_images.append(segmented_image)
        all_images_name.append("segmented_image")
        masked_image = mask_darkest_segment(segmented_image, labels, centers)
        all_images.append(masked_image)
        all_images_name.append("masked_image")
        _,smoothed_image= apply_median_filter(masked_image)
        all_images.append(smoothed_image)
        all_images_name.append("smoothed_image")
        _,eroded_image=apply_erosion2(smoothed_image)
        all_images.append(eroded_image)
        all_images_name.append("eroded_image")
        _,again_smoothed_image= apply_median_filter(eroded_image)
        all_images.append(again_smoothed_image)
        all_images_name.append("again_smoothed_image")
        threshold_value=100
        _,final_image = cv2.threshold(again_smoothed_image, threshold_value, 255, cv2.THRESH_BINARY)
        all_images.append(final_image)
        all_images_name.append("final_image")
        _,boxed_image,rectan=count_connected_components(final_image)
        ans2=boundary_error(rectan)
        ans=isRavanOneReal(rectan)
        if ans and ans2:
            class_name="real"
        else:
            class_name="fake"
        all_images.append(boxed_image)
        all_images_name.append("boxed_image")
        original_image = cv2.imread(image_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        all_images.append(original_image_rgb)
        all_images_name.append("original_image_rgb")

        n=len(all_images)
        num_rows = 2  
        num_cols = (n+ num_rows - 1) // num_rows
    else:
        k_value = 4  
        segmented_image, labels, centers = k_means_segmentation(image, k_value)
        all_images.append(segmented_image)
        all_images_name.append("segmented_image")
        masked_image = mask_darkest_segment(segmented_image, labels, centers)
        all_images.append(masked_image)
        all_images_name.append("masked_image")
        _,smoothed_image= apply_median_filter(masked_image)
        all_images.append(smoothed_image)
        all_images_name.append("smoothed_image")
        _,eroded_image=apply_erosion(smoothed_image)
        all_images.append(eroded_image)
        all_images_name.append("eroded_image")
        _,again_smoothed_image= apply_median_filter(eroded_image)
        all_images.append(again_smoothed_image)
        all_images_name.append("again_smoothed_image")
        threshold_value=100
        _,final_image = cv2.threshold(again_smoothed_image, threshold_value, 255, cv2.THRESH_BINARY)
        all_images.append(final_image)
        all_images_name.append("final_image")
        _,boxed_image,rectan=count_connected_components(final_image)
        _,ans=isRavanTwoReal(rectan)
        if ans:
            class_name="real"
        else:
            class_name="fake"
        all_images.append(boxed_image)
        all_images_name.append("boxed_image")
        original_image = cv2.imread(image_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        all_images.append(original_image_rgb)
        all_images_name.append("original_image_rgb")

        n=len(all_images)
        num_rows = 2  
        num_cols = (n+ num_rows - 1) // num_rows
    return class_name
