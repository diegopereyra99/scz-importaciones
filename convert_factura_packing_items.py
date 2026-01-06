#!/usr/bin/env python3
"""
Convierte los JSON de output/factura_packing a un CSV/XLSX con una fila por item.
Las columnas iniciales son numero_factura, proveedor, fecha_factura y linea_no
(sin prefijo item_), para poder mergear con DUA por factura/linea. Los extras se
incluyen como items adicionales, usando su indice_item si está presente.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set

# Columnas fijas de cabecera
LEADING_COLUMNS = ["numero_factura", "proveedor", "fecha_factura", "linea_no"]

# Columnas globales (se repiten en cada item, después de las leading)
GLOBAL_COLUMNS = [
    "termino_pago",
    "incoterm",
    "moneda_importe",
    "importe_factura",
    "costo_seguro",
    "costo_flete",
]

# Campos de item en un orden legible; sin prefijo item_
ITEM_COLUMNS_ORDERED = [
    "codigo_producto_proveedor",
    "descripcion",
    "cantidad",
    "unidad_medida",
    "precio_unitario",
    "precio_total",
    "porcentaje_descuento_linea",
    "lote",
    "fecha_vencimiento",
    "fecha_fabricacion",
    "pais_origen",
    "via",
]

IGNORED_ITEM_KEYS = {"indice_item"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte todos los outputs de factura_packing a CSV/XLSX con filas por item."
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
        default="output/factura_packing_items.csv",
        help="Ruta del CSV de salida (por defecto output/factura_packing_items.csv).",
    )
    parser.add_argument(
        "--output-excel",
        help="Ruta de salida opcional para XLSX. Requiere pandas instalado.",
    )
    return parser.parse_args()


def collect_global_keys(output_root: Path) -> Set[str]:
    keys: Set[str] = set()
    for json_path in output_root.rglob("*.json"):
        payload = json.loads(json_path.read_text())
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            keys.update(entry.get("data", {}).keys())
    # lista_items no es columna global
    keys.discard("lista_items")
    keys.discard("extras")
    # Estas ya están en LEADING
    keys.discard("numero_factura")
    keys.discard("proveedor")
    keys.discard("fecha_factura")
    return keys


def collect_item_keys(output_root: Path) -> Set[str]:
    keys: Set[str] = set()
    for json_path in output_root.rglob("*.json"):
        payload = json.loads(json_path.read_text())
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            data = entry.get("data", {}) or {}
            items = data.get("lista_items") or []
            extras = data.get("extras") or []
            for item in items:
                keys.update(item.keys())
            for extra in extras:
                keys.update(extra.keys())
    return keys


def compute_columns(
    extra_global_keys: Set[str], extra_item_keys: Set[str]
) -> tuple[List[str], List[str], List[str]]:
    base_global_keys = set(GLOBAL_COLUMNS) | set(LEADING_COLUMNS)
    global_columns = GLOBAL_COLUMNS + sorted(k for k in extra_global_keys if k not in base_global_keys)

    base_item_keys = set(ITEM_COLUMNS_ORDERED)
    extra_keys_sorted = sorted(
        k for k in extra_item_keys if k not in base_item_keys and k not in IGNORED_ITEM_KEYS
    )
    item_columns = ITEM_COLUMNS_ORDERED + extra_keys_sorted
    columns = LEADING_COLUMNS + global_columns + item_columns
    return columns, global_columns, item_columns


def collect_rows(
    output_root: Path, columns: List[str], global_columns: List[str], item_columns: List[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for json_path in sorted(output_root.rglob("*.json")):
        payload = json.loads(json_path.read_text())
        entries = payload if isinstance(payload, list) else [payload]

        for entry in entries:
            data = entry.get("data", {})
            items = data.get("lista_items") or []
            extras = data.get("extras") or []
            all_lines = list(items) + list(extras)
            numero_factura = data.get("numero_factura", "")
            proveedor = data.get("proveedor", "")
            fecha_factura = data.get("fecha_factura", "")

            base_globals: Dict[str, Any] = {}
            for col in global_columns:
                base_globals[col] = data.get(col, "")

            for idx, item in enumerate(all_lines, start=1):
                item_no = item.get("indice_item")
                if item_no in (None, ""):
                    item_no = idx

                row = {col: "" for col in columns}
                row["numero_factura"] = numero_factura
                row["proveedor"] = proveedor
                row["fecha_factura"] = fecha_factura
                row["linea_no"] = item_no
                for col in global_columns:
                    row[col] = base_globals.get(col, "")

                for col in item_columns:
                    row[col] = item.get(col, "")

                rows.append(row)
    return rows


def write_csv(destination: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(destination: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - dependencias externas
        raise RuntimeError(
            "La salida XLSX requiere pandas instalado. "
            "Instala pandas o usa solo CSV."
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_excel(destination, index=False)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.root).resolve()
    output_root = base_dir / "output" / "factura_packing"

    if not output_root.exists():
        raise SystemExit(f"No existe la carpeta de salida: {output_root}")

    extra_global_keys = collect_global_keys(output_root)
    extra_item_keys = collect_item_keys(output_root)
    columns, global_columns, item_columns = compute_columns(extra_global_keys, extra_item_keys)
    rows = collect_rows(output_root, columns, global_columns, item_columns)

    write_csv(Path(args.output), columns, rows)
    if args.output_excel:
        write_excel(Path(args.output_excel), columns, rows)


if __name__ == "__main__":
    main()
