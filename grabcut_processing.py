import cv2
import numpy as np

def apply_grabcut(img, rect):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_img = img * mask2[:, :, np.newaxis]
    return segmented_img, mask2

def remove_selected_region(image, mask):
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

def refine_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask

def inpaint_image(image, mask, method=cv2.INPAINT_NS):
    refined_mask = refine_mask(mask)
    inpainted_img = cv2.inpaint(image, refined_mask, 3, method)
    return inpainted_img
