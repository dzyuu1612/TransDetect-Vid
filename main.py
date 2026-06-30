"""
main.py — CLI cua TransDetect-Vid (code path CHINH thong nhat).
Moi logic xu ly nam trong package src.transdetect.pipeline.
"""
import argparse
from src.transdetect import pipeline


def main():
    p = argparse.ArgumentParser(description="TransDetect-Vid: chay pipeline phat hien doi tuong trong video.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["classical", "yolo"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    a = p.parse_args()
    if a.method == "classical":
        pipeline.run_classical(a.input, a.output)
    else:
        pipeline.run_yolo(a.input, a.output, model_path=a.model, conf=a.conf, iou=a.iou)


if __name__ == "__main__":
    main()
