import cv2
import fitz
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'

file = 'C://Users//heito//Downloads//28 Dec 2022, 4_25_48 AM.pdf'

pdf_file = fitz.open(file)

for page in pdf_file:
    pix = page.get_pixmap(matrix=fitz.Identity, dpi=None,
                          colorspace=fitz.csRGB, clip=None, alpha=True, annots=True)
    path_save = "C://Users//heito//Downloads//samplepdfimage-%i.png" % page.number
    pix.save(path_save)

    original_image = cv2.imread(path_save, cv2.IMREAD_GRAYSCALE)

    binary_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #All types of THERSH -> test others if BINARY dind't work
    ret, thresh1 = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(binary_image, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(binary_image, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(binary_image, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [binary_image, thresh1, thresh2, thresh3, thresh4, thresh5]

    #Plot to show all the THRESH types
    # for i in range(6):
    #     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    # Find table borders
    contours, _ = cv2.findContours(images[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea, default=0)
    vimage = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    vimage = cv2.drawContours(vimage, [largest_contour], 0, (0, 255, 0), 1)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = original_image[y : y + h, x : x + w]

    # resize to width=900
    scale_factor = 900.0 / cropped_image.shape[1]
    cropped_image = cv2.resize(cropped_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

    mask = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    height, width = mask.shape[:2]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 2, 1))
    horizontal_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 2, 3))
    horizontal_mask = cv2.erode(mask, horizontal_kernel)
    horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel2, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 2))
    vertical_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, height // 2))
    vertical_mask = cv2.erode(mask, vertical_kernel)
    vertical_mask = cv2.dilate(vertical_mask, vertical_kernel2, iterations=3)

    hor_ver_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    cropped_image[np.nonzero(hor_ver_mask)] = 255

    text = pytesseract.image_to_string(mask, config="--psm 3").replace('\n\n', '\n')

    with open("results.txt", 'a') as txt:
        txt.write(text + '\n')
        txt.close()

print("Finished")