#!/usr/bin/env python3
"""
Une los CSV de DUA items y factura_packing items por factura/linea.
Las claves de cruce son: DUA.factura ↔ factura.numero_factura y
DUA.linea_factura ↔ factura.item_no.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge de dua_items.csv con factura_packing_items.csv."
    )
    parser.add_argument(
        "--dua",
        default="output/dua_items.csv",
        help="CSV de items de DUA (default: output/dua_items.csv).",
    )
    parser.add_argument(
        "--factura",
        default="output/factura_packing_items.csv",
        help="CSV de items de factura (default: output/factura_packing_items.csv).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/dua_factura_merge.csv",
        help="Ruta del CSV de salida (default: output/dua_factura_merge.csv).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo no tiene encabezado: {path}")
        rows = [dict(row) for row in reader]
    return reader.fieldnames, rows


def normalize_key(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Normaliza números como 1.0 -> 1 para mejorar el match
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        return text
    return text


def build_factura_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        key = (normalize_key(row.get("numero_factura")), normalize_key(row.get("linea_no")))
        index[key] = row
    return index


def rename_dua_columns(
    dua_header: List[str], dua_rows: List[Dict[str, str]], factura_header: List[str]
) -> Tuple[List[str], List[Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]]]:
    base_renames = {
        "numero": "dua_item_ix",
        "factura": "dua_factura",
        "linea_factura": "dua_linea_factura",
    }

    # Paso 1: renombra segun base_renames
    renamed_header: List[str] = []
    for col in dua_header:
        renamed_header.append(base_renames.get(col, col))

    # Paso 2: evita colisiones con columnas de factura
    factura_set = set(factura_header)
    final_header: List[str] = []
    for col in renamed_header:
        if col != "DUA" and col in factura_set and not col.startswith("dua_"):
            final_header.append(f"dua_{col}")
        else:
            final_header.append(col)

    # Mapa de columnas finales por indice
    col_map = dict(zip(dua_header, final_header))

    # Construir filas renombradas y un índice para merge
    index: Dict[Tuple[str, str], Dict[str, str]] = {}
    renamed_rows: List[Dict[str, str]] = []
    for row in dua_rows:
        factura_val = row.get("factura", "")
        linea_val = row.get("linea_factura", "")
        renamed_row: Dict[str, str] = {}
        for original, final in col_map.items():
            renamed_row[final] = row.get(original, "")
        key = (normalize_key(factura_val), normalize_key(linea_val))
        index[key] = renamed_row
        renamed_rows.append(renamed_row)

    return final_header, renamed_rows, index


def compute_output_columns(
    dua_header: List[str], factura_header: List[str]
) -> List[str]:
    dua_cols = [c for c in dua_header if c != "DUA"]
    return ["DUA"] + list(factura_header) + dua_cols


def merge_rows(
    factura_rows: List[Dict[str, str]],
    dua_index: Dict[Tuple[str, str], Dict[str, str]],
    output_columns: List[str],
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    matched_keys: Set[Tuple[str, str]] = set()

    for factura_row in factura_rows:
        key = (normalize_key(factura_row.get("numero_factura")), normalize_key(factura_row.get("linea_no")))
        dua_row = dua_index.get(key)

        combined = {col: "" for col in output_columns}
        combined["DUA"] = dua_row.get("DUA", "") if dua_row else ""
        for col, val in factura_row.items():
            combined[col] = val
        if dua_row:
            matched_keys.add(key)
            for col, val in dua_row.items():
                combined[col] = val
        merged.append(combined)

    # Agregar las filas de DUA que no matchearon con ninguna factura
    for key, dua_row in dua_index.items():
        if key in matched_keys:
            continue
        combined = {col: "" for col in output_columns}
        for col, val in dua_row.items():
            combined[col] = val
        merged.append(combined)

    return merged


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dua_header_raw, dua_rows_raw = read_csv(Path(args.dua))
    factura_header, factura_rows = read_csv(Path(args.factura))

    dua_header, dua_rows, dua_index = rename_dua_columns(dua_header_raw, dua_rows_raw, factura_header)
    output_columns = compute_output_columns(dua_header, factura_header)
    merged_rows = merge_rows(factura_rows, dua_index, output_columns)

    write_csv(Path(args.output), output_columns, merged_rows)


if __name__ == "__main__":
    main()
