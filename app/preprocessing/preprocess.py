import cv2


def resize_image(image, size=(256, 256)):
    resized = cv2.resize(image, size)

    return resized