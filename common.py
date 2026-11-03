"""Shared helpers: configuration loading and path resolution.

Every script takes ``--config config.yaml``. Relative paths in the file are
resolved against ``root``; absolute paths are used as they are.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

YEARS_DEFAULT = [2017, 2018, 2019, 2020, 2021]
STAGES = [10, 12, 15, 18, 19, 21, 24]
STAGE_NAMES = {10: 'Seeding', 12: 'Leaf development', 15: 'Stem elongation', 18: 'Heading',
               19: 'Milk ripeness', 21: 'Yellow ripeness', 24: 'Harvest'}


class Config(dict):
    """Dictionary with path helpers."""

    @property
    def root(self) -> Path:
        return Path(self['root']).expanduser()

    @property
    def years(self) -> list[int]:
        return list(self.get('years', YEARS_DEFAULT))

    def path(self, key: str, mkdir: bool = False) -> Path:
        p = Path(str(self['paths'][key])).expanduser()
        p = p if p.is_absolute() else self.root / p
        if mkdir:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def pattern(self, key: str) -> str:
        """Path pattern with {year} or {yy} placeholders, resolved against root."""
        p = str(self['paths'][key])
        return p if Path(p.split('{')[0]).is_absolute() else str(self.root / p)


def load_config(path: str | Path) -> Config:
    with open(path, encoding='utf-8') as f:
        return Config(yaml.safe_load(f))


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--config', default='config.yaml', help='YAML configuration file (default: config.yaml)')
    return ap
