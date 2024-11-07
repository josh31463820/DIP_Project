from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2

img1 = cv2.imread('original.png')
img2 = cv2.imread("images/reconstructed_image.png")
# img3 = cv2.imread('result_no_map_no_g.png')

p_m = compare_psnr(img1, img2)
# p_nm = compare_psnr(img1, img3)

print('p_m = ',p_m)
# print('p_nm = ',p_nm)
