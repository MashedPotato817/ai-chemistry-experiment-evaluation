#!/usr/bin/env python3
"""Use a YOLO checkpoint to sample-detect an experiment video.

Default values match the teacher-provided files.  The script processes every
``--frame-stride`` frame (20 by default, i.e. one frame per second for this
video) and produces an annotated MP4, a per-detection CSV, and a summary.

Example (from the project root):
  .\\.venv-cpython\\Scripts\\python.exe model_tests\\fs_s_best1592\\detect_teacher_video.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "fs_s_best1592.pt"
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "1-1.01_74_fs.mp4"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "1-1.01_74_fs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 实验视频抽帧检测")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=1088, help="推理尺寸（训练尺寸为 1088）")
    parser.add_argument("--frame-stride", type=int, default=20, help="每隔多少原始帧检测一次")
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或设备编号")
    return parser.parse_args()


def ensure_readable(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description}不存在：{path}")


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride 必须大于或等于 1")
    ensure_readable(args.model, "模型文件")
    ensure_readable(args.video, "视频文件")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video = args.output_dir / "annotated_sampled.mp4"
    csv_path = args.output_dir / "detections.csv"
    summary_path = args.output_dir / "summary.json"

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sampled_fps = fps / args.frame_stride
    writer = cv2.VideoWriter(
        str(annotated_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sampled_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{annotated_video}")

    print(f"加载模型：{args.model}")
    model = YOLO(str(args.model))
    print(f"输入视频：{args.video.name}，{width}x{height}，{fps:.2f} FPS，{total_frames} 帧")
    print(f"抽样策略：每 {args.frame_stride} 帧检测一次；输出视频帧率 {sampled_fps:.2f} FPS")

    class_counts: Counter[str] = Counter()
    sampled_frames = 0
    frames_with_detections = 0
    total_detections = 0
    elapsed_inference = 0.0
    started = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "timestamp_s", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2",
        ])

        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % args.frame_stride != 0:
                frame_id += 1
                continue

            infer_start = time.perf_counter()
            result = model.predict(
                frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False
            )[0]
            elapsed_inference += time.perf_counter() - infer_start
            sampled_frames += 1
            timestamp_s = frame_id / fps if fps else 0.0

            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(result.names[class_id])
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                csv_writer.writerow([
                    frame_id, f"{timestamp_s:.3f}", class_id, class_name,
                    f"{confidence:.4f}", f"{x1:.1f}", f"{y1:.1f}",
                    f"{x2:.1f}", f"{y2:.1f}",
                ])
                class_counts[class_name] += 1
                total_detections += 1

            if len(result.boxes):
                frames_with_detections += 1
            writer.write(result.plot())

            if sampled_frames % 50 == 0:
                print(f"已检测 {sampled_frames} 帧（原视频 {frame_id + 1}/{total_frames} 帧）")
            frame_id += 1

    cap.release()
    writer.release()
    wall_time = time.perf_counter() - started
    summary = {
        "model": str(args.model),
        "video": str(args.video),
        "device": args.device,
        "confidence_threshold": args.conf,
        "inference_size": args.imgsz,
        "frame_stride": args.frame_stride,
        "source_video": {
            "width": width, "height": height, "fps": fps, "total_frames": total_frames,
            "duration_s": total_frames / fps if fps else None,
        },
        "sampled_frames": sampled_frames,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "detections_by_class": dict(class_counts.most_common()),
        "inference_time_s": round(elapsed_inference, 3),
        "average_inference_ms": round(elapsed_inference / sampled_frames * 1000, 2) if sampled_frames else None,
        "wall_time_s": round(wall_time, 3),
        "outputs": {"annotated_video": str(annotated_video), "detections_csv": str(csv_path)},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
