"""
preprocessing.py
-----------------
Module tien xu ly anh - giai thich CHI TIET TUNG DONG.
"""

import cv2          # Thu vien xu ly anh/video chinh, viet tat cua "Computer Vision"
import numpy as np  # Thu vien xu ly so/ma tran, dung de lam viec voi du lieu anh (la mang so)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Chuyen anh mau (BGR) sang anh xam."""
    # cv2.cvtColor: ham doi he mau cua OpenCV
    # COLOR_BGR2GRAY: ma lenh bao "doi tu BGR (mau OpenCV dung) sang GRAY (xam)"
    # Ket qua: moi pixel tu 3 gia tri (B,G,R) -> chi con 1 gia tri (do sang)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray  # tra ket qua ve cho noi goi ham


def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Can bang histogram de tang do tuong phan."""
    # cv2.equalizeHist: tu dong tinh histogram (so luong pixel theo tung muc xam 0-255)
    # roi "keo gian" lai cho phan bo deu hon -> anh ro net hon
    # CHI dung duoc voi anh 1 kenh (anh xam), khong dung truc tiep voi anh mau
    equalized = cv2.equalizeHist(gray)
    return equalized


def median_denoise(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Loc trung vi de khu nhieu muoi tieu."""
    if kernel_size % 2 == 0:        # kernel (cua so loc) BAT BUOC phai la so le (3,5,7..)
        kernel_size += 1            # neu nguoi dung truyen so chan, tu dong +1 cho thanh le
    # cv2.medianBlur: voi moi pixel, xet 1 o vuong kernel_size x kernel_size quanh no,
    # lay GIA TRI TRUNG VI (median) cua tat ca pixel trong o do thay cho pixel trung tam.
    denoised = cv2.medianBlur(gray, kernel_size)
    return denoised


def preprocess(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Goi lan luot 3 buoc tien xu ly tren 1 frame."""
    gray = to_gray(frame)                        # Buoc 1: anh mau -> anh xam
    equalized = equalize_histogram(gray)         # Buoc 2: tang tuong phan
    denoised = median_denoise(equalized, kernel_size)  # Buoc 3: khu nhieu
    return denoised  # tra ve anh xam da xu ly xong, san sang cho Threshold/Sobel


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cach dung: python preprocessing.py duong_dan_anh.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Khong doc duoc anh tai: {img_path}")
        sys.exit(1)

    result = preprocess(frame)

    cv2.imshow("Anh goc", frame)
    cv2.imshow("Sau tien xu ly", result)
    print("Nhan phim bat ky tren cua so anh de dong...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
