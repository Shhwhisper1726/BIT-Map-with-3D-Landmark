import cv2
import os
from PIL import Image

images_path = './data/nerf_llff_data/stone0/images/' # 原图路径
output_dir = './data/nerf_lllf_data/stone0/' # resize后路径

factor = 8 # 降采样倍数

images_list = os.listdir(images_path)
img = Image.open(images_path + images_list[0])
(W,H) = (img.width,img.height) #[W,H]
print("image_size : ",(W ,H))


for image_name in images_list:
    # print(images_path+image_name)
    img = cv2.imread(images_path+image_name)
    img_resize = cv2.resize(img, (W//factor, H//factor))
    path = "H:/BITMAP/nerf-pytorch-master/data/nerf_llff_data/stone0/images_8/" + image_name
    print(path)
    cv2.imwrite(path, img_resize)
    print(image_name , " done")

cv2.imwrite(".\images_8\fuckyou2.jpg", img_resize)
# cv2.imshow("bitch", img_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("all images done")
