"""
classical_detector.py
-----------------------
Phep bien doi thu cong: Thresholding + Sobel + tim box.
Giai thich CHI TIET TUNG DONG.
"""

import cv2
import numpy as np


def iterative_global_threshold(gray: np.ndarray, epsilon: float = 0.5) -> int:
    """Tu tim nguong T bang phuong phap lap."""
    T = float(np.mean(gray))   # B1: T khoi tao = trung binh do sang toan anh

    while True:                        # lap vo han, se "break" khi hoi tu
        g1 = gray[gray > T]            # lay TAT CA pixel co do sang LON HON T -> nhom "sang"
        g2 = gray[gray <= T]           # lay TAT CA pixel co do sang NHO HON/BANG T -> nhom "toi"

        if len(g1) == 0 or len(g2) == 0:
            # neu 1 trong 2 nhom khong co pixel nao (anh toan sang hoac toan toi)
            # thi khong tinh duoc trung binh -> dung lap de tranh loi chia 0
            break

        m1 = np.mean(g1)               # trung binh do sang cua nhom "sang"
        m2 = np.mean(g2)               # trung binh do sang cua nhom "toi"
        T_new = (m1 + m2) / 2.0        # nguong moi = trung diem cua 2 trung binh

        if abs(T_new - T) < epsilon:
            # neu nguong moi gan nhu khong doi so voi nguong cu (sai khac < epsilon)
            # nghia la da HOI TU -> chot gia tri va dung lap
            T = T_new
            break
        T = T_new                      # chua hoi tu -> cap nhat T va lap tiep

    return int(T)                      # tra ve so nguyen (0-255) de dung cho threshold


def apply_threshold(gray: np.ndarray, T: int = None) -> np.ndarray:
    """Ap dung nguong T de tao anh nhi phan."""
    if T is None:
        T = iterative_global_threshold(gray)   # neu khong truyen T, tu tinh bang ham tren

    # cv2.threshold(anh, nguong, gia_tri_gan, kieu):
    # voi kieu THRESH_BINARY: pixel > T -> gan 255 (trang); pixel <= T -> gan 0 (den)
    # ham tra ve 2 gia tri: (nguong_thuc_te_dung, anh_ket_qua) -> ta chi can anh ket qua
    _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    return binary


def sobel_edge(gray: np.ndarray) -> np.ndarray:
    """Phat hien bien bang dao ham Sobel theo 2 huong x, y."""
    # cv2.Sobel(anh, kieu_du_lieu_dau_ra, dao_ham_x, dao_ham_y, ksize):
    # dong nay: dx=1, dy=0 -> tinh dao ham theo huong NGANG (phat hien bien DUNG/doc)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # dong nay: dx=0, dy=1 -> tinh dao ham theo huong DUNG (phat hien bien NGANG)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # CV_64F: dung kieu so thuc 64-bit de khong bi mat dau am cua dao ham (anh xam goc la uint8 khong co dau)

    # Cong huong 2 dao ham bang cong thuc Pythagore -> ra "do lon gradient" tai moi pixel
    # gradient lon = do sang thay doi manh = nhieu kha nang la bien vat the
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # gradient co the > 255 (vi tinh tu so thuc), can gioi han lai ve khoang 0-255
    # va doi sang kieu uint8 (so nguyen khong dau 8-bit) de hien thi/luu nhu anh thuong
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude


def find_candidate_boxes(binary: np.ndarray, edge: np.ndarray,
                          edge_threshold: int = 50,
                          min_area: int = 500) -> list:
    """Ket hop mask threshold + mask bien, tim contour -> bounding box."""
    # bien anh bien (gia tri lien tuc 0-255) thanh anh nhi phan (chi 0 hoac 255)
    # de co the OR (gop) voi mask threshold
    _, edge_binary = cv2.threshold(edge, edge_threshold, 255, cv2.THRESH_BINARY)

    # bitwise_or: pixel duoc giu lai (255) neu NO LA 255 O IT NHAT 1 trong 2 mask
    # -> vung duoc ca threshold va sobel "dong y" la vat the deu duoc giu
    combined = cv2.bitwise_or(binary, edge_binary)

    # tao 1 "kernel" hinh vuong 5x5 toan so 1, dung de lam phep toan hinh thai hoc
    kernel = np.ones((5, 5), np.uint8)
    # MORPH_CLOSE = phep DAN (dilate) roi MON (erode): giup noi cac vung trang gan nhau
    # thanh 1 vung lien tuc, dong thoi lam day cac lo nho ben trong vat the
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # findContours: tim cac "duong vien" (contour) la tap hop diem tao thanh 1 vung kin
    # RETR_EXTERNAL: chi lay duong vien NGOAI CUNG (bo qua lo/vien ben trong)
    # CHAIN_APPROX_SIMPLE: nen gon duong vien, chi giu cac diem can thiet (giam dung luong)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []                           # danh sach rong de chua box ket qua
    for cnt in contours:                 # lap qua tung duong vien tim duoc
        area = cv2.contourArea(cnt)      # tinh dien tich (so pixel) ben trong duong vien
        if area < min_area:
            continue                     # qua nho -> coi la nhieu, bo qua, khong them vao boxes
        x, y, w, h = cv2.boundingRect(cnt)   # tinh khung chu nhat NHO NHAT bao quanh duong vien
        boxes.append((x, y, w, h))       # them box (x,y,w,h) vao danh sach ket qua

    return boxes, combined               # tra ve ca danh sach box va anh mask de debug


def detect(frame_preprocessed: np.ndarray) -> tuple:
    """Ham tong hop: nhan anh da tien xu ly, tra ve box va mask."""
    binary = apply_threshold(frame_preprocessed)         # B1: tao mask threshold
    edge = sobel_edge(frame_preprocessed)                # B2: tao anh bien sobel
    boxes, debug_mask = find_candidate_boxes(binary, edge)  # B3: gop lai va tim box
    return boxes, debug_mask


if __name__ == "__main__":
    import sys
    from preprocessing import preprocess

    if len(sys.argv) < 2:
        print("Cach dung: python classical_detector.py duong_dan_anh.jpg")
        sys.exit(1)

    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print("Khong doc duoc anh.")
        sys.exit(1)

    pre = preprocess(frame)         # tien xu ly anh truoc khi phat hien
    boxes, mask = detect(pre)       # chay toan bo buoc phat hien

    output = frame.copy()           # copy anh goc de khong ve len anh goc thuc su
    for (x, y, w, h) in boxes:
        # ve khung chu nhat mau xanh la (0,255,0), do day duong vien = 2 pixel
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Tim duoc {len(boxes)} vung ung vien.")
    cv2.imshow("Mask ket hop", mask)
    cv2.imshow("Ket qua phat hien", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
