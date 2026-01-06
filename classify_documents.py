#!/usr/bin/env python3
"""
Classify PDFs from data/carpetas into data/classified/<clase_documento>/...
based on the extracted metadata stored under output/*/*.json.

Expected layout:
- output/<carpeta>/<archivo>.pdf.json  (contains data.clase_documento)
- data/carpetas/<carpeta>/<archivo>.pdf  (source PDF to copy)

Destination layout:
- data/classified/<clase_documento>/<archivo>.pdf  (flattened, no carpeta)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).parent
OUTPUT_ROOT = ROOT / "output" / "classify"
SOURCE_ROOT = ROOT / "data" / "carpetas"
DEST_ROOT = ROOT / "data" / "classified"


def extract_class(json_path: Path) -> Optional[str]:
    """Return clase_documento from the extraction JSON, if present."""
    try:
        content = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - we only need a simple log here
        print(f"[WARN] No se pudo leer {json_path}: {exc}")
        return None

    def from_entry(entry: object) -> Optional[str]:
        if isinstance(entry, dict):
            data = entry.get("data") if isinstance(entry.get("data"), dict) else entry
            clase = data.get("clase_documento") if isinstance(data, dict) else None
            if isinstance(clase, str) and clase.strip():
                return clase.strip()
        return None

    if isinstance(content, list):
        for item in content:
            clase = from_entry(item)
            if clase:
                return clase
    else:
        clase = from_entry(content)
        if clase:
            return clase

    print(f"[WARN] No se encontró data.clase_documento en {json_path}")
    return None


def classify():
    if not OUTPUT_ROOT.exists():
        raise SystemExit(f"No existe el directorio {OUTPUT_ROOT}")

    total = 0
    copied = 0

    for json_file in OUTPUT_ROOT.glob("**/*.json"):
        total += 1
        # extract the part after output/
        folder = json_file.relative_to(OUTPUT_ROOT).parent
        pdf_name = json_file.with_suffix("").name  # remove .json, keep .pdf
        source_pdf = SOURCE_ROOT / folder / pdf_name

        if not source_pdf.exists():
            print(f"[WARN] PDF de origen no encontrado: {source_pdf}")
            continue

        clase = extract_class(json_file)
        if not clase:
            continue

        dest_dir = DEST_ROOT / clase
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / pdf_name

        if dest_path.exists():
            print(f"[WARN] Ya existe {dest_path}, se omite para evitar sobreescritura")
            continue

        shutil.copy2(source_pdf, dest_path)
        copied += 1
        print(f"[OK] {source_pdf} -> {dest_path}")

    print(f"Procesados: {total}, copiados: {copied}")


if __name__ == "__main__":
    classify()
