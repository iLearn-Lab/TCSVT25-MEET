#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np


NUM_FOLDS = 5
IMAGES_PER_FOLD = 1000
CAPTIONS_PER_IMAGE = 5
CAPTIONS_PER_FOLD = IMAGES_PER_FOLD * CAPTIONS_PER_IMAGE


def read_lines(path: Path) -> List[str]:
    with path.open('r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def split_json_records(records, start: int, end: int):
    if isinstance(records, list):
        return records[start:end]
    raise TypeError('Only list-style JSON files are supported.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split MS-COCO testall precomputed features into 5 official 1K folds.'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/coco_precomp'),
        help='Directory containing testall_ims.npy and testall_caps.txt.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Where to write test0~test4 files. Defaults to input dir.',
    )
    parser.add_argument(
        '--source-prefix',
        default='testall',
        help='Prefix of the source files, e.g. testall.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing fold files.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate and print the split plan without writing files.',
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    prefix = args.source_prefix

    ims_path = input_dir / f'{prefix}_ims.npy'
    caps_txt_path = input_dir / f'{prefix}_caps.txt'
    ids_txt_path = input_dir / f'{prefix}_ids.txt'
    caps_json_path = input_dir / f'{prefix}_caps.json'
    precaps_path = input_dir / f'{prefix}_precaps_stan.txt'

    if not ims_path.exists():
        raise FileNotFoundError(f'Missing source image array: {ims_path}')
    if not caps_txt_path.exists():
        raise FileNotFoundError(f'Missing source caption file: {caps_txt_path}')

    images = np.load(ims_path)
    caps_txt = read_lines(caps_txt_path)
    ids_txt = read_lines(ids_txt_path) if ids_txt_path.exists() else None
    caps_json = None
    if caps_json_path.exists():
        with caps_json_path.open('r', encoding='utf-8') as f:
            caps_json = json.load(f)
    precaps_txt = read_lines(precaps_path) if precaps_path.exists() else None

    expected_unique_images = NUM_FOLDS * IMAGES_PER_FOLD
    expected_caps = NUM_FOLDS * CAPTIONS_PER_FOLD

    if images.shape[0] == expected_unique_images:
        repeated_image_layout = False
    elif images.shape[0] == expected_caps:
        repeated_image_layout = True
    else:
        raise ValueError(
            f'Expected either {expected_unique_images} unique images or '
            f'{expected_caps} repeated-image rows in {ims_path.name}, '
            f'got {images.shape[0]}.'
        )

    if len(caps_txt) != expected_caps:
        raise ValueError(
            f'Expected {expected_caps} captions in {caps_txt_path.name}, got {len(caps_txt)}.'
        )
    if ids_txt is not None and len(ids_txt) != expected_caps:
        raise ValueError(
            f'Expected {expected_caps} ids in {ids_txt_path.name}, got {len(ids_txt)}.'
        )
    if caps_json is not None and len(caps_json) != expected_caps:
        raise ValueError(
            f'Expected {expected_caps} records in {caps_json_path.name}, got {len(caps_json)}.'
        )
    if precaps_txt is not None and len(precaps_txt) != expected_caps:
        raise ValueError(
            f'Expected {expected_caps} lines in {precaps_path.name}, got {len(precaps_txt)}.'
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(NUM_FOLDS):
        cap_start = fold * CAPTIONS_PER_FOLD
        cap_end = (fold + 1) * CAPTIONS_PER_FOLD

        if repeated_image_layout:
            img_start = cap_start
            img_end = cap_end
        else:
            img_start = fold * IMAGES_PER_FOLD
            img_end = (fold + 1) * IMAGES_PER_FOLD

        fold_prefix = f'test{fold}'

        outputs = [
            output_dir / f'{fold_prefix}_ims.npy',
            output_dir / f'{fold_prefix}_caps.txt',
        ]
        if ids_txt is not None:
            outputs.append(output_dir / f'{fold_prefix}_ids.txt')
        if caps_json is not None:
            outputs.append(output_dir / f'{fold_prefix}_caps.json')
        if precaps_txt is not None:
            outputs.append(output_dir / f'{fold_prefix}_precaps_stan.txt')

        if not args.force:
            exists = [p for p in outputs if p.exists()]
            if exists:
                joined = ', '.join(str(p) for p in exists)
                raise FileExistsError(
                    f'Fold {fold} outputs already exist: {joined}. Use --force to overwrite.'
                )

        layout = 'repeated-image' if repeated_image_layout else 'unique-image'
        print(
            f'layout={layout} fold={fold} '
            f'images[{img_start}:{img_end}] -> {img_end - img_start}, '
            f'captions[{cap_start}:{cap_end}] -> {cap_end - cap_start}'
        )

        if args.dry_run:
            continue

        np.save(output_dir / f'{fold_prefix}_ims.npy', images[img_start:img_end])
        write_lines(output_dir / f'{fold_prefix}_caps.txt', caps_txt[cap_start:cap_end])
        if ids_txt is not None:
            write_lines(output_dir / f'{fold_prefix}_ids.txt', ids_txt[cap_start:cap_end])
        if caps_json is not None:
            with (output_dir / f'{fold_prefix}_caps.json').open('w', encoding='utf-8') as f:
                json.dump(split_json_records(caps_json, cap_start, cap_end), f, ensure_ascii=False)
        if precaps_txt is not None:
            write_lines(output_dir / f'{fold_prefix}_precaps_stan.txt', precaps_txt[cap_start:cap_end])

    print('Done.')


if __name__ == '__main__':
    main()
