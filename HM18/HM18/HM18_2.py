# ----------- Pattern recognition: object detection and tracking -----------------

'''
визначення заповнених торгівельних полиць за технологіями image recognition

Package            Version
------------------ -----------
matplotlib         3.6.2
numpy              1.24.1
opencv-python      3.4.18.65

'''

import cv2
from matplotlib import pyplot as plt


def image_read(FileIm):
    image = cv2.imread(FileIm)
    plt.imshow(image)
    plt.show()

    return image


def image_processing_window(image):

    # зміст етапів обробки визначається властивостями вхідних зображень та обє'кту ідентифікації !!!

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                  # корекція кольору
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                        # Гаусова фільтрація
    edged = cv2.Canny(gray, 10, 250)                                # фільтр Кенні - векторизація
    plt.imshow(edged)
    plt.show()

    return edged



def image_contours(image_entrance):
    cnts = cv2.findContours(image_entrance.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    return cnts


def image_recognition(image_entrance, image_cont, file_name):
    total = 0
    for c in image_cont:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        # ---------------- головна ідея ідентифікації ---------------------------
        # ---------- якщо у контура 4 вершини то це обє'кт ----------------------
        if len(approx) == 3:
            cv2.drawContours(image_entrance, [approx], -1, (0, 255, 0), 3)
            total += 1

    print("Знайдено {0} сегмент(а) трикутних об'єктів".format(total))
    cv2.imwrite(file_name, image_entrance)
    plt.imshow(image_entrance)
    plt.show()

    return

if __name__ == '__main__':


    image_entrance = image_read("cats.jpg")
    image_exit = image_processing_window(image_entrance)
    image_cont = image_contours(image_exit)
    image_recognition(image_entrance, image_cont, "cats_3.jpg")

  