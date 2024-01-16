

import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os

dataset_folder = r'C:\Users\paras\Downloads\QR Code.v2i.multiclass\train'
output_file = 'qr_code_text.txt'

cap = cv2.VideoCapture(0)

with open(output_file, 'a') as file:
    while (cap.isOpened()):
        _, img = cap.read()

        decoded_codes = decode(img)

        for code in decoded_codes:
            decoded_data = code.data.decode('utf-8')
            file.write(f'Decoded Data: {decoded_data}\n')
            file.write('\n')

            points = (np.array([code.polygon], np.int32))
            points = (points.reshape((-1, 1, 2)))
            points2 = code.rect
            if decoded_data :
                cv2.polylines(img, [points], True, (0,255,0), 5)
                cv2.putText(img, decoded_data, (points2[0], points2[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (0,255,0), 2)
            else:
                cv2.polylines(img, [points], True, (0,0,255), 5)
                cv2.putText(img, 'UnAuthorized', (points2[0], points2[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2)

        cv2.imshow('image', img)
        if cv2.waitKey(1)& 0xff == ord('q'):
            break

    for filename in os.listdir(dataset_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(dataset_folder, filename)
            
            img = cv2.imread(img_path)

            decoded_codes = decode(img)

            for code in decoded_codes:
                decoded_data = code.data.decode('utf-8')
                file.write(f'Image: {img_path}\n')
                file.write(f'Decoded Data: {decoded_data}\n')
                file.write('\n')

            cv2.imshow('img', img)
            cv2.waitKey(1000)

cv2.destroyAllWindows()
