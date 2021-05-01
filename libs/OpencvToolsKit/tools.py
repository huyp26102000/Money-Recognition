import cv2

def apply_threshold(thresh):
    """
    Apply OTSU threshold
    """
#     ret, thresh = cv2.threshold(thresh, 250, 255, cv2.THRESH_TOZERO)
    # ret, thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
     
    # thresh = cv2.GaussianBlur(thresh,(11,11),0)
    # thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,6)
    

    
    # ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

    # plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
#     plt.title('After applying OTSU threshold')
    # plt.show()
    return thresh


def shi_tomashi(gray_image):
    """
    Use Shi-Tomashi algorithm to detect corners
    Args:
        image: np.array
    Returns:
        corners: list
    """
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_image, 4, 0.1, 100)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    # print('\nThe corner points are...\n')

    # im = image.copy()
    # for index, c in enumerate(corners):
    #     x, y = c
    #     cv2.circle(im, (x, y), 50, 255, -1)
    #     character = chr(65 + index)
    #     print(character, ':', c)
    #     cv2.putText(im, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0 , 255), 2, cv2.LINE_AA)
    # plt.figure(figsize=(10,10))
    # plt.imshow(im)
    # plt.title('Corner Detection: Shi-Tomashi')
    # plt.show()
    return corners