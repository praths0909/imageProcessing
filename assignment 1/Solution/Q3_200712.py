import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # image = cv2.imread(image_path)
    # return image
    return solve2(solve3(solve(input_path=image_path)))
def intercept(x1,y1,x2,y2,x):
    a=(x-x1)*(y2-y1)
    a=a/(x2-x1)
    a=a+y1
    return a
def solve(input_path):
    input_image = cv2.imread(input_path)
    # print(input_image.shape)
    height,width,channels=input_image.shape
    # plt.imshow(input_image)
    # plt.show()
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find edges
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Determine the angle of the most prominent line
    if lines is not None:
        # print(lines)
        max_rho = 0
        max_theta = 0
        for line in lines:
            rho, theta = line[0]
            if rho > max_rho:
                max_rho = rho
                max_theta = theta
        max_theta=lines[0][0][1]
        # Calculate the angle between the most prominent line and the horizontal axis
        angle = np.degrees(max_theta) - 90.0
        # print(angle)
        # height, width = input_image.shape[:2]
        # new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
        # new_height = int(height * abs(np.cos(np.radians(angle))) + width * abs(np.sin(np.radians(angle))))

        # # Create a white canvas of the calculated dimensions
        # canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 255 for white color
        #     # Rotate the entire image to make the text horizontal
        # rotated_image = cv2.warpAffine(input_image, cv2.getRotationMatrix2D((input_image.shape[1] / 2, input_image.shape[0] / 2), angle, 1), (input_image.shape[1], input_image.shape[0]))
        # plt.imshow(rotated_image)
        # plt.show()
        # # Calculate the transformation matrix for rotation and translation
        # rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), angle, 1)

        # # Perform the rotation and translation onto the canvas
        # rotated_image = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))
        # if abs(angle)<5:
        #     reference_line = rotated_image.shape[0] // 2
        #     above_reference = np.sum(rotated_image[:reference_line, :] > 128)
        #     below_reference = np.sum(rotated_image[reference_line:, :] > 128)

        #     # Adjust the orientation by 180 degrees if below_reference > above_reference
        #     if below_reference > above_reference:
        #         rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)

        # Calculate the dimensions of the rotated image
        # Calculate the dimensions of the rotated image
        height, width = input_image.shape[:2]
        new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
        new_height = int(height * abs(np.cos(np.radians(angle))) + width * abs(np.sin(np.radians(angle))))

        # Create a white canvas of the calculated dimensions
        canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 255 for white color

        # Calculate the transformation matrix for rotation
        center_x = width / 2  # Center of rotation along the original image's width
        center_y = height / 2  # Center of rotation along the original image's height
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

        # Calculate the translation values to properly center the rotated image
        tx = (new_width - width) / 2
        ty = (new_height - height) / 2

        # Update the translation values in the transformation matrix
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # Perform the rotation and translation onto the canvas
        rotated_image = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))

        # Create the final output image
        output_image = rotated_image.copy()
        # cv2.imwrite("Result_test/"+str(i)+"a.png",output_image)
        # plt.imshow(rotated_image)
        # plt.show()

        # Display the output image (optional)
        # print(output_image.shape)
        # plt.imshow(output_image)
        # plt.show()
        return output_image
    else:
        input_image = cv2.imread(input_path)
        return input_image
def solve3(input_image):
    
    height,width,channels=input_image.shape
    # plt.imshow(input_image)
    # plt.show()
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find edges
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Determine the angle of the most prominent line
    if lines is not None:
        # print(lines)
        max_rho = 0
        max_theta = 0
        for line in lines:
            rho, theta = line[0]
            if rho > max_rho:
                max_rho = rho
                max_theta = theta
        # max_theta=lines[0][0][1]
        # Calculate the angle between the most prominent line and the horizontal axis
        angle = np.degrees(max_theta) - 90.0
        # print(angle)
        # height, width = input_image.shape[:2]
        # new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
        # new_height = int(height * abs(np.cos(np.radians(angle))) + width * abs(np.sin(np.radians(angle))))

        # # Create a white canvas of the calculated dimensions
        # canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 255 for white color
        #     # Rotate the entire image to make the text horizontal
        # rotated_image = cv2.warpAffine(input_image, cv2.getRotationMatrix2D((input_image.shape[1] / 2, input_image.shape[0] / 2), angle, 1), (input_image.shape[1], input_image.shape[0]))
        # plt.imshow(rotated_image)
        # plt.show()
        # # Calculate the transformation matrix for rotation and translation
        # rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), angle, 1)

        # # Perform the rotation and translation onto the canvas
        # rotated_image = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))
        # if abs(angle)<5:
        #     reference_line = rotated_image.shape[0] // 2
        #     above_reference = np.sum(rotated_image[:reference_line, :] > 128)
        #     below_reference = np.sum(rotated_image[reference_line:, :] > 128)

        #     # Adjust the orientation by 180 degrees if below_reference > above_reference
        #     if below_reference > above_reference:
        #         rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)

        # Calculate the dimensions of the rotated image
        # Calculate the dimensions of the rotated image
        height, width = input_image.shape[:2]
        new_width = int(width * abs(np.cos(np.radians(angle))) + height * abs(np.sin(np.radians(angle))))
        new_height = int(height * abs(np.cos(np.radians(angle))) + width * abs(np.sin(np.radians(angle))))

        # Create a white canvas of the calculated dimensions
        canvas = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 255 for white color

        # Calculate the transformation matrix for rotation
        center_x = width / 2  # Center of rotation along the original image's width
        center_y = height / 2  # Center of rotation along the original image's height
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

        # Calculate the translation values to properly center the rotated image
        tx = (new_width - width) / 2
        ty = (new_height - height) / 2

        # Update the translation values in the transformation matrix
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # Perform the rotation and translation onto the canvas
        rotated_image = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))

        # Create the final output image
        output_image = rotated_image.copy()
        # cv2.imwrite("Result_test/"+str(i)+"a.png",output_image)
        # plt.imshow(rotated_image)
        # plt.show()

        # Display the output image (optional)
        # print(output_image.shape)
        # plt.imshow(output_image)
        # plt.show()
        return output_image
    else:
        return input_image
def solve2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Draw the detected lines on the original image
    # for rho, theta in lines[:, 0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
        
    #     # Draw the line on the original image
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     plt.imshow(image)
    #     plt.show()
    # print(lines)
    # print('lol')
    if lines is not None and len(lines) > 0:
        # print('looooo')
        rho, theta = lines[0, 0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Calculate the coordinates of the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Draw the line on the original image
        # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.circle(image, (x1, y1), 5, (0, 255, 255), 2)
        # cv2.circle(image, (x2, y2), 20, (0, 255, 255), 10)
        # cv2.circle(image, (25, 25), 20, (0, 255, 255), 10)
        # plt.imshow(image)
        # plt.show()

        # Define a region of interest (ROI) above and below the line
        # print(x1,y1,x2,y2)
        # a1=intercept(x1,y1,x2,y2,0+10)
        # a2=intercept(x1,y1,x2,y2,gray.shape[0]-10)
        # x1=0+10
        # y1=int(a1)
        # x2=gray.shape[0]-10
        # y2=int(a2)
        # print(x1,y1,x2,y2)
        # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        roi_above = gray[max(0, y1 -5):y1, x1:x2]
        roi_below = gray[y2:min(y2 + 5, gray.shape[0]), x1:x2]
        # print(gray.shape)
        # roi_above = gray[y1-5:y1,x1:x2]
        # roi_below = gray[y2:y2+5,x1:x2]
        # print(roi_above)
        # print(roi_below)

        # Set a threshold to classify dark pixels
        threshold = 100

        # Count the number of dark pixels above and below the line
        dark_pixels_above = np.sum(roi_above < threshold)
        dark_pixels_below = np.sum(roi_below < threshold)
        # print(dark_pixels_above,dark_pixels_below)
        # If more dark pixels are above the line, rotate the image by 180 degrees
        if dark_pixels_above > dark_pixels_below:
            image = cv2.rotate(image, cv2.ROTATE_180)
    # plt.imshow(image)
    # plt.show()
    # cv2.imwrite("Result_test/"+str(i)+".png",image)
    return image