import os
import binascii
import numpy as np
import math
import cv2
import codeVis

# convert code to image

if __name__ == "__main__":

    for filepath,dirnames,filenames in os.walk(r'your_java_path'):

        for filename in filenames:
            project_name = filename.split('.txt')[0]
            f_path = os.path.join(filepath, filename)
            f = open(f_path, encoding='UTF-8')
            for line in f_path:
                if os.path.exists(f_path):
                    im = codeVis.get_new_img(f_path)  # gray image
                    print(f_path)  # generate gray image

                    path_save = f_path + ''.join(line[:-3]).replace('/', '_') + '.png'
                    path_save = path_save.replace('.txt', '')
                    cv2.imwrite(path_save, im)

# Gray diagram
def get_new_img(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])

    x = len(fh)
    x_width = math.sqrt(x)
    x_width = math.ceil(x_width)
    y = x_width * x_width - x

    fh = np.pad(fh, (0, y), 'constant', constant_values=0)
    img = np.array(fh).reshape(x_width, x_width)

    return img

def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    size = fsize/float(1024)
    return round(size,2)


