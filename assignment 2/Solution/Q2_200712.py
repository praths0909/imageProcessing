import cv2
import numpy as np
def isequal(a,b):
    d=abs(a-b)
    if d<5:
        return True
    else:
        return False
    
def calculate_gaussian(i,j,sigma):
    a=(i)**2
    b=(j)**2
    c=(2 * sigma**2)
    kernel_size=-(a + b) / c
    return np.exp(kernel_size)
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

def cross_bilateral_filter(image, guide_image, kernel_size, sig_color, sig_spa):
    result = np.zeros_like(image, dtype=np.float32)
    rows, columns, _ = image.shape

    for i in range(rows):
        for j in range(columns):
            i_min_value=0
            if((i-kernel_size)>0):
                i_min_value=i-kernel_size
            i_max_value=i+kernel_size
            if rows<i_max_value:
                i_max_value=rows
            j_min_value=0
            if((j-kernel_size)>0):
                j_min_value=j-kernel_size
            j_max_value=j+kernel_size
            if columns < j_max_value:
                j_max_value=columns

            imagePatches = image[i_min_value:i_max_value+1, j_min_value:j_max_value+1]
            flashPatches = guide_image[i_min_value:i_max_value+1, j_min_value:j_max_value+1]

            spatial_weights = calculate_gaussian(i-i_min_value,j-j_min_value,sig_spa)

            intensity_weights = np.exp(-np.clip(np.sum((guide_image[i, j] - flashPatches)**2, axis=2), -100, 100) / (2 * sig_color**2))

            weights = spatial_weights * intensity_weights
            chec=np.sum(weights)
            if chec != 0:
                weights /= chec

            for c in range(3):
                result[i, j, c] = np.sum(weights * imagePatches[:, :, c])

    return result.astype(np.uint8)

def apply_operations(array1,array2):
    addition_result = array1 + array2
    subtraction_result = array1 - array2
    elementwise_multiplication_result = array1 * array2
    elementwise_division_result = array1 / array2

def transpose_array(array1):
    array1_transposed = np.transpose(array1)
    return array1_transposed

def multiply_array(array1,array2):
    matrix_multiplication_result = np.dot(array1, array2)
    return matrix_multiplication_result

def calculate_sum_array(array1):
    sum_result = np.sum(array1)
    return sum_result

def calculate_mean_array(array1):
    mean_result = np.mean(array1)
    return mean_result

def std_deviation(array1):
    std_deviation_result = np.std(array1)
    return std_deviation


def find_details(flash):
    e = 0.02
    a=flash+e
    return a/(cv2.GaussianBlur(flash,(23,23),2.2)+e)

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image_path1=image_path_a
    image_path2=image_path_b
    non_flash= cv2.imread(image_path1)
    flash_image= cv2.imread(image_path2)
    kernel_size=11
    sig_color=2.2
    sig_spa=0.6
    ans=cross_bilateral_filter(non_flash, flash_image, kernel_size, sig_color, sig_spa)


    return ans




