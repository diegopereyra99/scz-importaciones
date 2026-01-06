#!/usr/bin/env python3
"""
Convierte los resultados de clasificacion de importaciones a CSV o XLSX.

Recorre todas las carpetas dentro de output/clasificacion_importaciones y crea
una fila por archivo detectado. Las columnas generadas son:
- Archivo (ruta relativa en data/carpetas)
- tipo (clase del documento)
- identificador/numero
- fecha
- proveedor
- importe
- peso
- resumen
- meta_campos_clave (campos clave formateados en texto)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COLUMN_NAMES = [
    "Archivo",
    "tipo",
    "identificador/numero",
    "fecha",
    "proveedor",
    "importe",
    "peso",
    "resumen",
    "meta_campos_clave",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte los JSON de output/clasificacion_importaciones a CSV o XLSX."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Directorio base del proyecto (por defecto el cwd).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/clasificacion_importaciones.csv",
        help="Ruta del CSV de salida (por defecto output/clasificacion_importaciones.csv).",
    )
    parser.add_argument(
        "--output-excel",
        help="Ruta de salida opcional para XLSX. Requiere pandas instalado.",
    )
    return parser.parse_args()


def format_campos_clave(campos: Optional[Iterable[Dict[str, Any]]]) -> str:
    formatted: List[str] = []
    if not campos:
        return ""
    for campo in campos:
        nombre = str(campo.get("nombre", "")).strip()
        valor = campo.get("valor")
        if isinstance(valor, (dict, list)):
            valor_str = json.dumps(valor, ensure_ascii=False)
        elif valor is None:
            valor_str = ""
        else:
            valor_str = str(valor)
        notas = campo.get("notas")
        if notas:
            formatted.append(f"{nombre}: {valor_str} ({notas})")
        else:
            formatted.append(f"{nombre}: {valor_str}")
    return " | ".join(formatted)


def collect_rows(output_root: Path, data_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for json_path in sorted(output_root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text())
        except Exception as exc:  # pragma: no cover - defensivo
            raise RuntimeError(f"No se pudo leer {json_path}: {exc}") from exc

        entries = payload if isinstance(payload, list) else [payload]
        rel_output = json_path.relative_to(output_root)
        rel_data = rel_output.with_suffix("")  # quita solo .json -> deja .pdf
        data_path = data_root / rel_data

        for entry in entries:
            data = entry.get("data", {})
            row = {
                "Archivo": str(data_path),
                "tipo": data.get("clase_documento", ""),
                "identificador/numero": data.get("numero_documento", ""),
                "fecha": data.get("fecha_documento", ""),
                "proveedor": data.get("proveedor_importacion", ""),
                "importe": data.get("importe_total", ""),
                "peso": data.get("peso_total", ""),
                "resumen": data.get("resumen", ""),
                "meta_campos_clave": format_campos_clave(data.get("campos_clave")),
            }
            rows.append(row)
    return rows


def write_csv(destination: Path, rows: List[Dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMN_NAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(destination: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependencias externas
        raise RuntimeError(
            "La salida XLSX requiere pandas instalado. "
            "Instala pandas o usa solo CSV."
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=COLUMN_NAMES)
    frame.to_excel(destination, index=False)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.root).resolve()
    output_root = base_dir / "output" / "clasificacion_importaciones"
    data_root = base_dir / "data" / "carpetas"

    if not output_root.exists():
        raise SystemExit(f"No existe la carpeta de salida: {output_root}")
    if not data_root.exists():
        raise SystemExit(f"No existe la carpeta de datos: {data_root}")

    rows = collect_rows(output_root, data_root)
    write_csv(Path(args.output), rows)

    if args.output_excel:
        write_excel(Path(args.output_excel), rows)


if __name__ == "__main__":
    main()
