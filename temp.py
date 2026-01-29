import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取灰階圖片
img = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 2D 傅立葉轉換
F = np.fft.fft2(img)

# 3. 把低頻移到中心（只是為了「看」，不是傅立葉的一部分）
F_shift = np.fft.fftshift(F)

# 4. 取 magnitude（因為是複數）
magnitude = np.abs(F_shift)

# 5. log 壓縮（不然你只會看到黑）
magnitude_log = np.log(1 + magnitude)

# 6. 顯示
plt.figure(figsize=(6, 6))
plt.imshow(magnitude_log, cmap='gray')
plt.title("Fourier Transform (Magnitude Spectrum)")
plt.axis("off")
plt.show()
