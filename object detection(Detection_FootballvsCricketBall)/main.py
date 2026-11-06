from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_NAMES = {0: 'cricket ball', 1: 'football'}


def inspect_dataset(data: Path) -> dict:
    """원본을 수정하지 않고 이미지/라벨 쌍과 YOLO 좌표를 검사한다."""
    from PIL import Image

    report = {'data': str(data), 'splits': {}, 'errors': [], 'warnings': []}
    for split in ('train', 'valid'):
        images_dir, labels_dir = data / 'images' / split, data / 'labels' / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            report['errors'].append(f'{split}: images/labels directory missing')
            continue
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = sorted(labels_dir.glob('*.txt'))
        image_ids = Counter(p.stem for p in images)
        label_ids = {p.stem for p in labels}
        counts = Counter()
        empty = 0
        if not images:
            report['errors'].append(f'{split}: no images')
        for stem, count in image_ids.items():
            if count > 1:
                report['errors'].append(f'{split}: duplicate image stem {stem}')
        for stem in sorted(set(image_ids) ^ label_ids):
            report['errors'].append(f'{split}: unmatched image/label {stem}')
        for path in images:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as exc:
                report['errors'].append(f'{path}: {exc}')
        for path in labels:
            try:
                lines = path.read_text(encoding='utf-8').splitlines()
            except (OSError, UnicodeError) as exc:
                report['errors'].append(f'{path}: {exc}')
                continue
            empty += not any(line.strip() for line in lines)
            for line_no, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError('expected class x_center y_center width height')
                    cls = int(fields[0])
                    x, y, w, h = map(float, fields[1:])
                    if cls not in (0, 1):
                        raise ValueError('class must be 0 or 1')
                    if not all(math.isfinite(v) for v in (x, y, w, h)):
                        raise ValueError('non-finite coordinate')
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError('coordinates outside normalized YOLO range')
                    if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                        report['warnings'].append(f'{path.name}:{line_no}: box extends beyond image')
                    counts[cls] += 1
                except ValueError as exc:
                    report['errors'].append(f'{path}:{line_no}: {exc}')
        report['splits'][split] = {
            'images': len(images), 'labels': len(labels),
            'objects_per_class': dict(sorted(counts.items())), 'empty_labels': empty,
        }
    overlap = {p.stem for p in (data/'images/train').glob('*')} & {
        p.stem for p in (data/'images/valid').glob('*')}
    if overlap:
        report['warnings'].append(f'{len(overlap)} image stems occur in both splits')
    return report


def load_yolo(model: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install dependencies: python -m pip install -r requirements.txt') from exc
    return YOLO(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='데이터 검사 및 JSON 보고서 저장')
    train = sub.add_parser('train', help='검사 후 YOLO 학습')
    for cmd in (check, train):
        cmd.add_argument('--data', type=Path, default=ROOT/'data')
    train.add_argument('--names', nargs=2, default=list(CLASS_NAMES.values()), metavar=('CLASS_0', 'CLASS_1'))
    train.add_argument('--model', default='yolo11n.pt')
    train.add_argument('--epochs', type=int, default=50)
    train.add_argument('--batch', type=int, default=8)
    train.add_argument('--imgsz', type=int, default=640)
    train.add_argument('--device', default='0', help='cpu 또는 GPU 번호(기본값: 0)')
    train.add_argument('--workers', type=int, default=0)
    pred = sub.add_parser('predict', help='학습된 가중치로 이미지/동영상 예측')
    pred.add_argument('--weights', type=Path, required=True)
    pred.add_argument('--source', required=True)
    pred.add_argument('--conf', type=float, default=0.25)
    pred.add_argument('--device', default='0')
    args = parser.parse_args()
    if args.command == 'predict':
        if not args.weights.is_file():
            parser.error(f'Weights not found: {args.weights}')
        if not 0 <= args.conf <= 1:
            parser.error('--conf must be between 0 and 1')
        model = load_yolo(str(args.weights.resolve()))
        if set(model.names) != set(CLASS_NAMES):
            parser.error('Expected a trained two-class ball detection model')
        model.model.names = CLASS_NAMES.copy()
        for _ in model.predict(source=args.source, conf=args.conf, device=args.device,
                               save=True, stream=True, project=str(ROOT/'runs'), name='predict'):
            pass
        return
    data = args.data.resolve()
    report = inspect_dataset(data)
    out = ROOT/'reports'
    out.mkdir(exist_ok=True)
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    (out/'dataset_report.json').write_text(report_text, encoding='utf-8')
    print(report_text)
    if report['errors']:
        raise SystemExit('Dataset validation failed. See reports/dataset_report.json')
    if args.command == 'check':
        return
    if min(args.epochs, args.batch, args.imgsz) <= 0 or args.workers < 0:
        parser.error('epochs/batch/imgsz must be positive; workers must be nonnegative')
    # JSON quoted strings are also valid YAML; handles spaces and Unicode paths.
    config = out/'dataset.yaml'
    config.write_text(
        f'path: {json.dumps(data.as_posix())}\ntrain: images/train\nval: images/valid\n'
        f'names: {json.dumps(args.names, ensure_ascii=False)}\n', encoding='utf-8')
    model = load_yolo(args.model)
    model.train(data=str(config), epochs=args.epochs, batch=args.batch,
                imgsz=args.imgsz, device=args.device, workers=args.workers,
                seed=42, project=str(ROOT/'runs'), name='ball_detection', plots=True)


if __name__ == '__main__':
    main()
