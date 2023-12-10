import cv2
import numpy as np
def intercept(x1,y1,x2,y2,x):
    a=(x-x1)*(y2-y1)
    a=a/(x2-x1)
    a=a+y1
    return a
def solve(input_path):
    input_image = cv2.imread(input_path)
    height,width,channels=input_image.shape

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is not None:
        max_rho = 0
        max_theta = 0
        for line in lines:
            rho, theta = line[0]
            if rho > max_rho:
                max_rho = rho
                max_theta = theta
        max_theta=lines[0][0][1]
        angle = np.degrees(max_theta) - 90.0
        height, width = input_image.shape[:2]
        new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
        new_height = int(height * abs(np.cos(np.radians(angle))) + width * abs(np.sin(np.radians(angle))))

        canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255 
        center_x = width / 2  
        center_y = height / 2  
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

        tx = (new_width - width) / 2
        ty = (new_height - height) / 2

        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        rotated_image = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))
        output_image = rotated_image.copy()
        return output_image
    else:
        input_image = cv2.imread(input_path)
        return input_image
def solve3(input_image): 
    
    height,width,channels=input_image.shape
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        max_rho = 0
        max_theta = 0
        for line in lines:
            rho, theta = line[0]
            if rho > max_rho:
                max_rho = rho
                max_theta = theta
        angle = np.degrees(max_theta) - 90.0
        height, width = input_image.shape[:2]
        new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
        new_height = int(height * abs(np.cos(np.radians(angle))) + width * abs(np.sin(np.radians(angle))))
        canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255 
        center_x = width / 2  
        center_y = height / 2  
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        tx = (new_width - width) / 2
        ty = (new_height - height) / 2
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        rotated_image = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))
        output_image = rotated_image.copy()
        return output_image
    else:
        return input_image
def solve2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None and len(lines) > 0:
        # print('looooo')
        rho, theta = lines[0, 0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        roi_above = gray[max(0, y1 -5):y1, x1:x2]
        roi_below = gray[y2:min(y2 + 5, gray.shape[0]), x1:x2]
        threshold = 100
        dark_pixels_above = np.sum(roi_above < threshold)
        dark_pixels_below = np.sum(roi_below < threshold)
        if dark_pixels_above > dark_pixels_below:
            image = cv2.rotate(image, cv2.ROTATE_180)
    return image
def detect_sun(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=30
    )
    if circles is not None:
        return True
    else:
        return False
def kmeans_segmentation(image, k=2):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image
def create_lava_mask(image, threshold=30):
    lower_orange = np.array([0, 100, 200], dtype=np.uint8)
    upper_orange = np.array([80, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(image, lower_orange, upper_orange)
    mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]

    return mask
def solution(image_path):
    image=cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    height, width, _ =image.shape
    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    flag=detect_sun(image_path)
    if flag:
        final_image=solve2(solve3(solve(input_path=image_path)))
        final_image = np.zeros((height, width, 3), dtype=np.uint8)
        return final_image
    else:
        lower_red = np.array([0, 0, 115], dtype=np.uint8)
        upper_red = np.array([82, 250, 255], dtype=np.uint8)
        red_orange_mask = cv2.inRange(image, lower_red, upper_red)
        contours, _ = cv2.findContours(red_orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_image = red_orange_mask.copy()
            cv2.drawContours(largest_contour_image, [largest_contour], -1, 255, thickness = cv2.FILLED)
            final_image = cv2.merge((largest_contour_image, largest_contour_image, largest_contour_image))
        else:
            image = cv2.imread(image_path)
            segmented_image = kmeans_segmentation(image)
            lava_mask = create_lava_mask(segmented_image)
            result = np.zeros_like(image)
            result[lava_mask != 0] = [255, 255, 255]  # White for lava region
            final_image = result

        # print(cv2.contourArea(largest_contour))
        
    ######################################################################  
    return final_image
