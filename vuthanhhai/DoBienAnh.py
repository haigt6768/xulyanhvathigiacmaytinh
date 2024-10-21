import cv2
import numpy as np

# Đọc ảnh
img = cv2.imread('Img/images.jpg', cv2.IMREAD_GRAYSCALE)

# Tạo các ma trận Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Tính độ lớn gradient
magnitude = np.sqrt(sobelx**2 + sobely**2)

# Áp dụng ngưỡng (ví dụ)
thresh = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Hiển thị kết quả
cv2.imshow('Edge Detected Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()