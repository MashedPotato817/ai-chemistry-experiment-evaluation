#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs_s_best.pt 模型综合测试脚本

测试内容：
1. 模型加载与基本信息
2. 推理性能（GPU/CPU）
3. 各置信度阈值下的检测行为
4. 不同图像尺寸的推理时间
5. 时序分析器集成测试
6. 评分引擎集成测试
7. 将所有结果写入 model_test_report.md
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# 添加项目根目录下的 src 到路径（兼容从 model_tests/ 或项目根目录运行）
_ROOT = Path(__file__).parent.parent  # 项目根目录
sys.path.insert(0, str(_ROOT / "src"))

MODEL_PATH = "models/fs_s_best.pt"
REPORT_PATH = "model_tests/model_test_report.md"

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def ok(msg: str):
    print(f"  ✅ {msg}")

def warn(msg: str):
    print(f"  ⚠️  {msg}")

def fail(msg: str):
    print(f"  ❌ {msg}")

# ── 测试模块 ──────────────────────────────────────────────────────────────────

def test_model_load():
    """测试1：模型加载"""
    section("测试1：模型加载")
    results = {}

    try:
        from ultralytics import YOLO
        import torch

        t0 = time.time()
        model = YOLO(MODEL_PATH)
        load_time = time.time() - t0

        results["status"] = "success"
        results["load_time_s"] = round(load_time, 3)
        results["model_type"] = type(model.model).__name__
        results["task"] = getattr(model, "task", "detect")
        results["num_classes"] = len(model.names)
        results["classes"] = dict(model.names)
        results["file_size_mb"] = round(Path(MODEL_PATH).stat().st_size / 1024 / 1024, 2)

        # 参数量
        total_params = sum(p.numel() for p in model.model.parameters())
        results["total_params"] = total_params
        results["total_params_M"] = round(total_params / 1e6, 2)

        # GPU 信息
        results["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["gpu_name"] = torch.cuda.get_device_name(0)
            results["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )

        # Checkpoint 详细信息（训练轮次、best_fitness、train_args）
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                if "epoch" in state:
                    results["checkpoint_epoch"] = state["epoch"]
                if "best_fitness" in state:
                    bf = state["best_fitness"]
                    results["best_fitness"] = round(float(bf), 4) if bf is not None else None
                if "train_args" in state:
                    args = state["train_args"]
                    items = (
                        vars(args).items() if hasattr(args, "__dict__")
                        else args.items() if hasattr(args, "items")
                        else []
                    )
                    results["train_args"] = {k: str(v) for k, v in items}
        except Exception as e:
            results["checkpoint_warning"] = str(e)

        ok(f"模型加载成功，耗时 {load_time:.3f}s")
        ok(f"类别数量: {results['num_classes']}")
        ok(f"参数量: {results['total_params_M']}M")
        if results.get("checkpoint_epoch") is not None:
            epoch_val = results['checkpoint_epoch']
            epoch_display = f"{epoch_val}（导出模型，非训练断点）" if epoch_val == -1 else str(epoch_val)
            ok(f"训练轮次: {epoch_display}")
        if results.get("best_fitness") is not None:
            ok(f"最佳适应度: {results['best_fitness']}")
        if results["cuda_available"]:
            ok(f"GPU: {results['gpu_name']}")

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"模型加载失败: {e}")

    return results


def test_inference_performance():
    """测试2：推理性能"""
    section("测试2：推理性能")
    results = {}

    try:
        from ultralytics import YOLO
        import torch

        model = YOLO(MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results["device"] = device

        # 测试不同输入尺寸
        sizes = [640, 1088, 1280]
        size_results = {}

        for size in sizes:
            dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

            # 预热
            for _ in range(3):
                model(dummy, verbose=False, device=device)

            # 正式测试 20 次
            times = []
            for _ in range(20):
                t0 = time.time()
                model(dummy, verbose=False, device=device)
                times.append(time.time() - t0)

            avg_ms = np.mean(times) * 1000
            std_ms = np.std(times) * 1000
            fps = 1000 / avg_ms

            size_results[size] = {
                "avg_ms": round(avg_ms, 1),
                "std_ms": round(std_ms, 1),
                "fps": round(fps, 1),
                "min_ms": round(min(times) * 1000, 1),
                "max_ms": round(max(times) * 1000, 1),
            }
            ok(f"{size}×{size}: {avg_ms:.1f}ms ± {std_ms:.1f}ms  ({fps:.1f} FPS)")

        results["size_benchmarks"] = size_results
        results["status"] = "success"

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"性能测试失败: {e}")

    return results


def test_confidence_thresholds():
    """测试3：不同置信度阈值下的检测行为"""
    section("测试3：置信度阈值测试")
    results = {}

    try:
        from ultralytics import YOLO
        import torch

        model = YOLO(MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 使用训练尺寸 1088
        dummy = np.random.randint(0, 255, (1088, 1088, 3), dtype=np.uint8)
        thresholds = [0.1, 0.25, 0.3, 0.5, 0.7, 0.9]
        threshold_results = {}

        for conf in thresholds:
            preds = model(dummy, verbose=False, device=device, conf=conf)
            n_detections = sum(len(r.boxes) for r in preds)
            threshold_results[conf] = n_detections
            ok(f"conf={conf}: 检测到 {n_detections} 个目标（随机图像）")

        results["threshold_results"] = threshold_results
        results["recommended_threshold"] = 0.5
        results["status"] = "success"
        results["note"] = "随机图像检测数量仅供参考，实际场景会有真实检测结果"

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"阈值测试失败: {e}")

    return results


def test_temporal_analyzer():
    """测试4：时序分析器集成测试"""
    section("测试4：时序分析器集成测试")
    results = {}

    try:
        from chemistry_lab.detection.temporal_analyzer import (
            TemporalAnalyzer, EXPERIMENT_STEPS_SEQUENCE, STEP_INDICATORS
        )
        from chemistry_lab.detection.models import Detection, BoundingBox

        analyzer = TemporalAnalyzer()

        # 模拟"添加药品"步骤的检测结果
        def make_detection(cls_id, name, x1, y1, x2, y2, conf=0.85):
            return Detection(
                class_id=cls_id,
                class_name=name,
                confidence=conf,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                timestamp=time.time(),
                frame_id=1,
            )

        # 场景1：准备器材阶段
        frame1 = [
            make_detection(0, "锥形瓶", 100, 100, 300, 400),
            make_detection(4, "集气瓶", 400, 100, 600, 350),
            make_detection(9, "酒精灯", 650, 200, 800, 380),
        ]
        analyzer.add_detections(frame1)
        step1 = analyzer.get_current_step()
        ok(f"场景1（锥形瓶+集气瓶+酒精灯）→ 推断步骤: {step1}")

        # 场景2：添加药品阶段
        frame2 = [
            make_detection(16, "手", 200, 150, 350, 300),
            make_detection(15, "药勺", 220, 160, 320, 260),
            make_detection(14, "二氧化锰固体", 300, 200, 450, 350),
            make_detection(0, "锥形瓶", 100, 100, 300, 400),
        ]
        analyzer.add_detections(frame2)
        step2 = analyzer.get_current_step()
        ok(f"场景2（手+药勺+二氧化锰）→ 推断步骤: {step2}")

        # 场景3：收集气体阶段
        frame3 = [
            make_detection(4, "集气瓶", 400, 100, 600, 350),
            make_detection(18, "气泡", 420, 300, 460, 340),
            make_detection(17, "水柱", 430, 200, 450, 320),
        ]
        analyzer.add_detections(frame3)
        step3 = analyzer.get_current_step()
        ok(f"场景3（集气瓶+气泡+水柱）→ 推断步骤: {step3}")

        # 器材完整性检查
        completeness = analyzer.check_apparatus_completeness()
        missing = analyzer.get_missing_apparatus()
        ok(f"器材完整性: {sum(completeness.values())}/{len(completeness)} 已检测")
        ok(f"缺失器材: {missing}")

        # 统计信息
        stats = analyzer.get_statistics()
        ok(f"追踪物体数: {stats['tracked_objects']}")
        ok(f"历史帧数: {stats['history_length']}")

        results["status"] = "success"
        results["step_inference"] = {
            "scene1_detected": step1,
            "scene2_detected": step2,
            "scene3_detected": step3,
        }
        results["apparatus_completeness"] = {
            "detected": sum(completeness.values()),
            "total": len(completeness),
            "missing": missing,
        }
        results["total_steps"] = len(EXPERIMENT_STEPS_SEQUENCE)
        results["step_indicators_count"] = len(STEP_INDICATORS)

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"时序分析器测试失败: {e}")

    return results


def test_scoring_engine():
    """测试5：评分引擎集成测试"""
    section("测试5：评分引擎集成测试")
    results = {}

    try:
        from chemistry_lab.scoring.scoring_engine import ScoringEngine, DEFAULT_EXPERIMENT_STATS

        engine = ScoringEngine()
        experiment_type = "气体制备与收集实验"
        mu, sigma = engine.get_experiment_statistics(experiment_type)

        ok(f"默认统计参数: μ={mu}s ({mu/60:.1f}min), σ={sigma}s")

        # 测试多个场景
        scenarios = [
            {"name": "优秀学生", "time": 360, "errors": 0, "questions": 1, "quality": 0.95, "safety": True,  "safety_events": 0, "steps": 9},
            {"name": "良好学生", "time": 480, "errors": 2, "questions": 3, "quality": 0.80, "safety": True,  "safety_events": 1, "steps": 8},
            {"name": "合格学生", "time": 600, "errors": 5, "questions": 6, "quality": 0.65, "safety": True,  "safety_events": 2, "steps": 7},
            {"name": "需改进",   "time": 780, "errors": 8, "questions": 10,"quality": 0.45, "safety": False, "safety_events": 4, "steps": 5},
        ]

        scenario_results = []
        for s in scenarios:
            result = engine.calculate_complete_score(
                experiment_type=experiment_type,
                actual_time=s["time"],
                error_count=s["errors"],
                question_count=s["questions"],
                operation_quality=s["quality"],
                safety_compliance=s["safety"],
                safety_event_count=s["safety_events"],
                steps_completed=s["steps"],
                total_steps=9,
            )
            scenario_results.append({
                "name": s["name"],
                "time_s": s["time"],
                "s1": round(result.s1_time_score, 3),
                "s2": round(result.s2_performance_score, 3),
                "final": round(result.final_score, 3),
                "grade": result.get_grade(),
                "percentile": round(result.percentile_rank, 1),
            })
            ok(
                f"{s['name']:6s}: {result.final_score:.3f}/2.00 "
                f"({result.get_grade()}) | "
                f"S1={result.s1_time_score:.3f} S2={result.s2_performance_score:.3f} | "
                f"超越{result.percentile_rank:.1f}%"
            )

        results["status"] = "success"
        results["default_mu"] = mu
        results["default_sigma"] = sigma
        results["scenarios"] = scenario_results

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"评分引擎测试失败: {e}")

    return results


def test_detector_integration():
    """测试6：检测器完整流程测试"""
    section("测试6：检测器完整流程测试")
    results = {}

    try:
        from chemistry_lab.detection.yolo_detector import YOLODetector
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = YOLODetector(
            model_path=MODEL_PATH,
            confidence_threshold=0.3,
            device=device,
        )

        # 用训练尺寸图像测试
        dummy = np.random.randint(0, 255, (1088, 1088, 3), dtype=np.uint8)

        # 运行 30 帧
        all_detections = []
        for i in range(30):
            dets = detector.detect_objects(dummy)
            all_detections.append(len(dets))

        stats = detector.get_statistics()
        ok(f"处理帧数: {stats['frame_count']}")
        ok(f"平均 FPS: {stats['average_fps']:.1f}")
        ok(f"平均推理时间: {stats['average_inference_time']*1000:.1f}ms")
        ok(f"每帧平均检测数: {np.mean(all_detections):.1f}")

        results["status"] = "success"
        results["frames_processed"] = stats["frame_count"]
        results["average_fps"] = round(stats["average_fps"], 1)
        results["avg_inference_ms"] = round(stats["average_inference_time"] * 1000, 1)
        results["avg_detections_per_frame"] = round(float(np.mean(all_detections)), 1)
        results["device"] = device

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"检测器集成测试失败: {e}")

    return results


def test_class_comparison():
    """测试7：模型类别与项目预设类别对比"""
    section("测试7：模型类别与项目预设类别对比")
    results = {}

    # 与 yolo_detector.py 中的 EXPERIMENT_CLASSES 保持一致
    PROJECT_CLASSES = {
        0:  '锥形瓶',
        1:  '双孔橡胶塞',
        2:  '直角导管',
        3:  '橡胶管',
        4:  '集气瓶',
        5:  '玻璃片',
        6:  '止水夹',
        7:  '长颈漏斗',
        8:  '废液缸',
        9:  '酒精灯',
        10: '火柴',
        11: '洗瓶',
        12: '熄灭木条',
        13: '燃烧木条',
        14: '二氧化锰固体',
        15: '药勺',
        16: '手',
        17: '水柱',
        18: '气泡',
    }

    try:
        from ultralytics import YOLO

        model = YOLO(MODEL_PATH)
        names = model.names  # {id: name}

        model_set = set(names.values())
        project_set = set(PROJECT_CLASSES.values())

        matched = model_set & project_set
        only_in_model = model_set - project_set
        only_in_project = project_set - model_set

        ok(f"匹配类别 ({len(matched)}): {matched if matched else '无'}")
        if only_in_model:
            warn(f"模型独有 ({len(only_in_model)}): {only_in_model}")
        else:
            ok("无模型独有类别（完全匹配）")
        if only_in_project:
            warn(f"项目预设但模型没有 ({len(only_in_project)}): {only_in_project}")
            warn("建议：更新 yolo_detector.py 中的 EXPERIMENT_CLASSES 以匹配模型实际类别")
        else:
            ok("项目预设类别全部被模型覆盖")

        results["status"] = "success"
        results["matched"] = sorted(matched)
        results["only_in_model"] = sorted(only_in_model)
        results["only_in_project"] = sorted(only_in_project)
        results["project_classes"] = PROJECT_CLASSES

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        fail(f"类别对比测试失败: {e}")

    return results


def analyze_model(model_path: str = MODEL_PATH):
    """独立命令行模式：快速打印模型分析摘要（不生成报告文件）"""
    print("=" * 60)
    print("🔍 YOLO 模型分析工具")
    print("=" * 60)

    from pathlib import Path as _Path
    path = _Path(model_path)
    if not path.exists():
        fail(f"模型文件不存在: {model_path}")
        sys.exit(1)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n📁 文件信息")
    print(f"   路径      : {path.resolve()}")
    print(f"   文件大小  : {file_size_mb:.2f} MB")

    print(f"\n⏳ 正在加载模型...")
    try:
        from ultralytics import YOLO
    except ImportError:
        fail("未安装 ultralytics，请运行: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)
    print("✅ 模型加载成功")

    # 基本信息
    print(f"\n📊 模型基本信息")
    print(f"   模型类型  : {type(model.model).__name__}")
    print(f"   任务类型  : {getattr(model, 'task', 'detect')}")

    # 检测类别
    print(f"\n🏷️  检测类别")
    names = model.names
    if names:
        print(f"   类别数量  : {len(names)}")
        for cls_id, cls_name in sorted(names.items()):
            print(f"      [{cls_id:2d}] {cls_name}")

    # Checkpoint 详细信息
    print(f"\n🧠 Checkpoint 信息")
    try:
        import torch
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            print(f"   Checkpoint 键: {list(state.keys())}")
            if "epoch" in state:
                print(f"   训练轮次  : {state['epoch']}")
            if "best_fitness" in state:
                bf = state["best_fitness"]
                print(f"   最佳适应度: {bf:.4f}" if bf is not None else "   最佳适应度: 未记录")
            if "train_args" in state:
                args = state["train_args"]
                items = (
                    vars(args).items() if hasattr(args, "__dict__")
                    else args.items() if hasattr(args, "items")
                    else []
                )
                print(f"   训练参数  :")
                for k, v in items:
                    print(f"      {k}: {v}")
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"   总参数量  : {total_params:,} ({total_params/1e6:.2f}M)")
    except Exception as e:
        warn(f"无法读取详细结构: {e}")

    # 推理性能（快速10次）
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_label = f"GPU ({torch.cuda.get_device_name(0)})" if device == "cuda" else "CPU"
    except Exception:
        device, device_label = "cpu", "CPU"

    print(f"\n⚡ 推理性能测试（{device_label}，随机图像 640×640）")
    try:
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        model(dummy_image, verbose=False, device=device)  # 预热
        times = []
        for _ in range(10):
            t0 = time.time()
            model(dummy_image, verbose=False, device=device)
            times.append(time.time() - t0)
        avg_ms = np.mean(times) * 1000
        fps = 1.0 / np.mean(times)
        print(f"   平均推理时间: {avg_ms:.1f} ms")
        print(f"   理论 FPS    : {fps:.1f}")
    except Exception as e:
        warn(f"性能测试失败: {e}")

    print(f"\n{'=' * 60}")
    print("✅ 分析完成")
    print("=" * 60)


# ── 报告生成 ──────────────────────────────────────────────────────────────────

def generate_report(all_results: dict) -> str:
    """生成 Markdown 测试报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r = all_results

    load = r.get("load", {})
    perf = r.get("performance", {})
    conf = r.get("confidence", {})
    temporal = r.get("temporal", {})
    scoring = r.get("scoring", {})
    detector = r.get("detector", {})
    class_cmp = r.get("class_comparison", {})

    # 整体状态
    all_passed = all(v.get("status") == "success" for v in r.values())
    overall = "✅ 全部通过" if all_passed else "⚠️ 部分测试失败"

    lines = []
    lines.append(f"# fs_s_best.pt 模型测试报告\n")
    lines.append(f"**测试时间：** {now}  ")
    lines.append(f"**整体状态：** {overall}  ")
    lines.append(f"**模型路径：** `{MODEL_PATH}`\n")
    lines.append("---\n")

    # ── 1. 模型基本信息 ──
    lines.append("## 1. 模型基本信息\n")
    if load.get("status") == "success":
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 文件大小 | {load.get('file_size_mb')} MB |")
        lines.append(f"| 模型类型 | {load.get('model_type')} |")
        lines.append(f"| 任务类型 | {load.get('task')} |")
        lines.append(f"| 参数量 | {load.get('total_params_M')} M ({load.get('total_params'):,}) |")
        if load.get("checkpoint_epoch") is not None:
            epoch_val = load.get('checkpoint_epoch')
            epoch_display = f"{epoch_val}（导出模型，非训练断点）" if epoch_val == -1 else str(epoch_val)
            lines.append(f"| 训练轮次 | {epoch_display} |")
        if load.get("best_fitness") is not None:
            lines.append(f"| 最佳适应度 | {load.get('best_fitness')} |")
        lines.append(f"| 类别数量 | {load.get('num_classes')} |")
        lines.append(f"| 加载耗时 | {load.get('load_time_s')} s |")
        lines.append(f"| CUDA 可用 | {'是' if load.get('cuda_available') else '否'} |")
        if load.get("gpu_name"):
            lines.append(f"| GPU 型号 | {load.get('gpu_name')} |")
            lines.append(f"| GPU 显存 | {load.get('gpu_memory_gb')} GB |")
        lines.append("")

        lines.append("### 检测类别（19个）\n")
        lines.append("| ID | 类别名称 | ID | 类别名称 |")
        lines.append("|----|---------|----|---------|")
        classes = load.get("classes", {})
        ids = sorted(classes.keys())
        for i in range(0, len(ids), 2):
            id1 = ids[i]
            name1 = classes[id1]
            if i + 1 < len(ids):
                id2 = ids[i + 1]
                name2 = classes[id2]
                lines.append(f"| {id1} | {name1} | {id2} | {name2} |")
            else:
                lines.append(f"| {id1} | {name1} | — | — |")
        lines.append("")
    else:
        lines.append(f"> ❌ 测试失败：{load.get('error')}\n")

    # ── 2. 推理性能 ──
    lines.append("## 2. 推理性能\n")
    if perf.get("status") == "success":
        lines.append(f"**推理设备：** {perf.get('device', 'unknown').upper()}\n")
        lines.append("| 输入尺寸 | 平均耗时 | 标准差 | 最小 | 最大 | FPS |")
        lines.append("|---------|---------|--------|------|------|-----|")
        for size, data in perf.get("size_benchmarks", {}).items():
            lines.append(
                f"| {size}×{size} | {data['avg_ms']} ms | ±{data['std_ms']} ms | "
                f"{data['min_ms']} ms | {data['max_ms']} ms | {data['fps']} |"
            )
        lines.append("")
        lines.append("> 训练输入尺寸为 **1088×1088**，推理时建议使用相同尺寸以获得最佳精度。\n")
    else:
        lines.append(f"> ❌ 测试失败：{perf.get('error')}\n")

    # ── 3. 置信度阈值 ──
    lines.append("## 3. 置信度阈值分析\n")
    if conf.get("status") == "success":
        lines.append("| 置信度阈值 | 检测数量（随机图像） |")
        lines.append("|-----------|-------------------|")
        for thresh, count in conf.get("threshold_results", {}).items():
            lines.append(f"| {thresh} | {count} |")
        lines.append("")
        lines.append(f"> **推荐阈值：** `{conf.get('recommended_threshold')}`  \n")
        lines.append(f"> {conf.get('note')}\n")
    else:
        lines.append(f"> ❌ 测试失败：{conf.get('error')}\n")

    # ── 4. 时序分析器 ──
    lines.append("## 4. 时序分析器集成测试\n")
    if temporal.get("status") == "success":
        lines.append(f"**实验步骤总数：** {temporal.get('total_steps')}  ")
        lines.append(f"**步骤指示器数：** {temporal.get('step_indicators_count')}\n")

        lines.append("### 步骤推断测试\n")
        lines.append("| 场景 | 检测器材 | 推断步骤 |")
        lines.append("|------|---------|---------|")
        si = temporal.get("step_inference", {})
        lines.append(f"| 场景1 | 锥形瓶、集气瓶、酒精灯 | {si.get('scene1_detected')} |")
        lines.append(f"| 场景2 | 手、药勺、二氧化锰固体 | {si.get('scene2_detected')} |")
        lines.append(f"| 场景3 | 集气瓶、气泡、水柱 | {si.get('scene3_detected')} |")
        lines.append("")

        ac = temporal.get("apparatus_completeness", {})
        lines.append(f"**器材完整性：** {ac.get('detected')}/{ac.get('total')} 已检测  ")
        missing = ac.get("missing", [])
        if missing:
            lines.append(f"**缺失器材：** {', '.join(missing)}\n")
        else:
            lines.append("**缺失器材：** 无\n")
    else:
        lines.append(f"> ❌ 测试失败：{temporal.get('error')}\n")

    # ── 5. 评分引擎 ──
    lines.append("## 5. 评分引擎集成测试\n")
    if scoring.get("status") == "success":
        mu = scoring.get("default_mu")
        sigma = scoring.get("default_sigma")
        lines.append(f"**默认统计参数：** μ = {mu}s（{mu/60:.1f}分钟），σ = {sigma}s\n")

        lines.append("| 场景 | 用时 | S1得分 | S2得分 | 最终得分 | 等级 | 超越百分比 |")
        lines.append("|------|------|--------|--------|---------|------|-----------|")
        for s in scoring.get("scenarios", []):
            m, sec = divmod(s["time_s"], 60)
            lines.append(
                f"| {s['name']} | {int(m)}分{sec}秒 | {s['s1']} | {s['s2']} | "
                f"**{s['final']}** | {s['grade']} | {s['percentile']}% |"
            )
        lines.append("")
    else:
        lines.append(f"> ❌ 测试失败：{scoring.get('error')}\n")

    # ── 6. 检测器完整流程 ──
    lines.append("## 6. 检测器完整流程测试\n")
    if detector.get("status") == "success":
        lines.append(f"| 项目 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 推理设备 | {detector.get('device', '').upper()} |")
        lines.append(f"| 处理帧数 | {detector.get('frames_processed')} |")
        lines.append(f"| 平均 FPS | {detector.get('average_fps')} |")
        lines.append(f"| 平均推理时间 | {detector.get('avg_inference_ms')} ms |")
        lines.append(f"| 每帧平均检测数 | {detector.get('avg_detections_per_frame')}（随机图像，预期为 0） |")
        lines.append("")
    else:
        lines.append(f"> ❌ 测试失败：{detector.get('error')}\n")

    # ── 7. 类别对比 ──
    lines.append("## 7. 模型类别与项目预设类别对比\n")
    if class_cmp.get("status") == "success":
        matched = class_cmp.get("matched", [])
        only_model = class_cmp.get("only_in_model", [])
        only_project = class_cmp.get("only_in_project", [])
        lines.append(f"| 对比项 | 数量 | 类别 |")
        lines.append(f"|--------|------|------|")
        lines.append(f"| ✅ 匹配 | {len(matched)} | {', '.join(matched) if matched else '无'} |")
        lines.append(f"| 🆕 模型独有 | {len(only_model)} | {', '.join(only_model) if only_model else '无'} |")
        lines.append(f"| ⚠️ 项目预设但模型没有 | {len(only_project)} | {', '.join(only_project) if only_project else '无'} |")
        lines.append("")
        if only_model or only_project:
            lines.append("> 💡 建议：更新 `yolo_detector.py` 中的 `EXPERIMENT_CLASSES` 以匹配模型实际类别\n")
        else:
            lines.append("> ✅ 模型类别与项目预设完全一致，无需更新\n")
    else:
        lines.append(f"> ❌ 测试失败：{class_cmp.get('error')}\n")

    # ── 8. 总结与建议 ──
    lines.append("## 8. 总结与建议\n")

    lines.append("### 模型能力\n")
    lines.append("- 模型基于 **YOLOv11s** 架构，在导师自制数据集 `fs_new` 上训练 1200 轮")
    lines.append("- 可识别气体制备与收集实验中的 **19 种器材**，覆盖完整实验流程")
    lines.append("- 训练使用 4 卡 GPU 并行，AdamW 优化器，余弦学习率调度\n")

    lines.append("### 部署建议\n")
    if load.get("cuda_available"):
        lines.append("- ✅ 已检测到 GPU，建议使用 `YOLO_DEVICE=cuda` 进行推理")
    else:
        lines.append("- ⚠️ 未检测到 GPU，当前使用 CPU 推理，建议配置 CUDA 环境")

    lines.append("- 推理输入尺寸建议设置为 **1088×1088**（与训练一致）")
    lines.append("- 置信度阈值建议使用 **0.5**，可根据实际场景调整")
    lines.append("- 实时视频流建议目标 **30 FPS**，GPU 推理完全满足\n")

    lines.append("### 已完成的代码适配\n")
    lines.append("- `yolo_detector.py`：EXPERIMENT_CLASSES 已更新为 19 个类别")
    lines.append("- `temporal_analyzer.py`：步骤推断、安全检测、器材完整性检查已针对新模型重写")
    lines.append("- `llm_service.py`：实验名称、步骤、降级指导已更新")
    lines.append("- `scoring_engine.py`：默认统计参数已更新为气体制备实验的估计值")
    lines.append("- `pages.py`：UI 已更新，支持步骤导航和实验结束评分展示")
    lines.append("- `.env`：`YOLO_MODEL_PATH`、`YOLO_DEVICE`、`YOLO_INPUT_SIZE` 已配置\n")

    lines.append("---")
    lines.append(f"*报告生成时间：{now}*")

    return "\n".join(lines)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  fs_s_best.pt 模型综合测试")
    print("="*60)
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  报告输出: {REPORT_PATH}")

    all_results = {}

    all_results["load"]             = test_model_load()
    all_results["performance"]      = test_inference_performance()
    all_results["confidence"]       = test_confidence_thresholds()
    all_results["temporal"]         = test_temporal_analyzer()
    all_results["scoring"]          = test_scoring_engine()
    all_results["detector"]         = test_detector_integration()
    all_results["class_comparison"] = test_class_comparison()

    # 生成并写入报告
    section("生成测试报告")
    report = generate_report(all_results)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    ok(f"报告已写入: {REPORT_PATH}")

    # 统计
    passed = sum(1 for v in all_results.values() if v.get("status") == "success")
    total  = len(all_results)
    print(f"\n  测试结果: {passed}/{total} 通过")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 支持 --analyze 参数快速运行独立分析（不生成报告）
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
        analyze_model(model_path)
    else:
        main()
