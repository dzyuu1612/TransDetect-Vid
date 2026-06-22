"""
optical_flow.py
-----------------
Lucas-Kanade Optical Flow - giai thich CHI TIET TUNG DONG.
"""

import cv2
import numpy as np


def get_good_features(gray: np.ndarray, max_corners: int = 100) -> np.ndarray:
    """Tim diem goc tot de theo doi (Shi-Tomasi)."""
    points = cv2.goodFeaturesToTrack(
        gray,                # anh xam dau vao
        maxCorners=max_corners,   # so diem toi da tra ve (lay diem "tot" nhat truoc)
        qualityLevel=0.3,    # diem co do "tot" (gia tri rieng cua ma tran Harris) phai
                              # >= 0.3 * do_tot_cua_diem_tot_nhat moi duoc giu lai
        minDistance=7,       # 2 diem duoc giu lai phai cach nhau it nhat 7 pixel
                              # (tranh chon nhieu diem qua sat nhau, du thua thong tin)
        blockSize=7          # kich thuoc o vuong dung de tinh do "tot" cua moi diem
    )
    return points   # tra ve mang toa do (N,1,2), hoac None neu khong tim duoc diem nao


def track_points(prev_gray: np.ndarray, curr_gray: np.ndarray,
                  prev_points: np.ndarray) -> tuple:
    """Theo doi diem tu frame truoc sang frame hien tai."""
    lk_params = dict(
        winSize=(15, 15),    # kich thuoc cua so tim kiem quanh moi diem (cang lon
                              # cang bat duoc chuyen dong lon, nhung cham hon)
        maxLevel=2,           # so tang "pyramid" (anh thu nho dan): giup tim duoc
                              # ca chuyen dong nho (tang got) va chuyen dong lon (tang tho)
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        # dieu kien dung qua trinh toi uu noi bo cua thuat toan:
        # dung khi: lap qua 10 lan, HOAC sai so < 0.03 (cai nao toi truoc thi dung)
    )

    # calcOpticalFlowPyrLK: ham chinh cua Lucas-Kanade trong OpenCV
    # input: 2 anh xam lien tiep + vi tri diem can theo doi o frame truoc
    # output: vi tri moi cua tung diem (curr_points) + trang thai theo doi (status)
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )

    status = status.reshape(-1)   # dua status ve mang 1 chieu (tu (N,1) thanh (N,))
    # status[i] == 1 nghia la diem i duoc tim thay thanh cong o frame moi
    # status[i] == 0 nghia la diem i bi mat (vd: bi che, ra ngoai khung hinh)

    good_prev = prev_points.reshape(-1, 2)[status == 1]   # chi giu cac diem THEO DOI THANH CONG (vi tri cu)
    good_curr = curr_points.reshape(-1, 2)[status == 1]   # vi tri moi tuong ung cua cac diem do

    return good_prev, good_curr


def draw_motion_vectors(frame: np.ndarray, good_prev: np.ndarray,
                         good_curr: np.ndarray) -> np.ndarray:
    """Ve mui ten the hien huong chuyen dong."""
    output = frame.copy()   # copy de khong ve truc tiep len bien frame goc
    # zip(): ghep 2 mang lai, lap dong thoi qua tung cap diem cu - diem moi tuong ung
    for (xp, yp), (xc, yc) in zip(good_prev, good_curr):
        p1 = (int(xp), int(yp))    # toa do diem o frame TRUOC (lam tron ve so nguyen)
        p2 = (int(xc), int(yc))    # toa do diem o frame HIEN TAI
        # ve mui ten tu p1 -> p2, mau do (0,0,255 trong BGR), do day 2, dau mui ten dai 40% duong
        cv2.arrowedLine(output, p1, p2, (0, 0, 255), 2, tipLength=0.4)
        # ve 1 cham tron mau xanh duong tai vi tri moi de de nhin diem hien tai
        cv2.circle(output, p2, 3, (255, 0, 0), -1)   # -1 = to dac (fill) hinh tron
    return output


if __name__ == "__main__":
    import sys

    # sys.argv[1] = duong dan video, neu khong co thi dung 0 (webcam mac dinh)
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)   # mo video/webcam de doc tung frame

    ret, prev_frame = cap.read()     # doc frame DAU TIEN
    if not ret:
        print("Khong doc duoc video/camera.")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)   # doi frame dau ve anh xam
    prev_points = get_good_features(prev_gray)                  # tim diem goc tot ban dau

    while True:                       # lap qua tat ca frame con lai
        ret, frame = cap.read()       # doc 1 frame moi
        if not ret:
            break                     # het video (khong doc duoc nua) -> dung lap

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # doi frame hien tai ve xam

        if prev_points is None or len(prev_points) < 5:
            # neu mat het diem theo doi (con qua it diem), tim lai diem moi tu frame truoc
            prev_points = get_good_features(prev_gray)

        if prev_points is not None:
            good_prev, good_curr = track_points(prev_gray, curr_gray, prev_points)
            output = draw_motion_vectors(frame, good_prev, good_curr)
            # cap nhat diem theo doi = vi tri moi, doi kieu du lieu cho dung voi
            # ham calcOpticalFlowPyrLK o vong lap ke tiep (yeu cau shape (N,1,2), kieu float32)
            prev_points = good_curr.reshape(-1, 1, 2).astype(np.float32)
        else:
            output = frame   # khong co diem nao de theo doi -> hien anh goc, khong ve gi them

        cv2.imshow("Lucas-Kanade Optical Flow", output)
        prev_gray = curr_gray   # frame hien tai tro thanh "frame truoc" cho vong lap ke tiep

        # waitKey(30): cho 30ms truoc khi doc phim bam, tuong ung ~33 FPS hien thi
        # neu nguoi dung bam phim 'q' thi dung chuong trinh
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()             # giai phong video/webcam
    cv2.destroyAllWindows()   # dong tat ca cua so dang mo
