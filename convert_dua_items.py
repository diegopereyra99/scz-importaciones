#!/usr/bin/env python3
"""
Convierte los JSON de output/dua_items a un CSV/XLSX con una fila por item de
DUA. Incluye las columnas iniciales DUA, factura y linea_factura (expande una
fila por cada entrada en linea_factura_asociada) y genera columnas por tributo
de item: \"<Tributo> (%) - alicuota\", \"<Tributo> - B imp\" y \"<Tributo> - monto
total\" (base * alicuota).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# Columnas fijas al inicio
LEADING_COLUMNS = ["DUA", "factura", "linea_factura"]

# Orden base para campos de item (sin prefijo item_)
ITEM_COLUMNS_ORDERED = [
    "dua_item_ix",
    "descripcion_mercaderia",
    "nombre_producto",
    "pais_origen",
    "pais_procedencia",
    "cantidad_bultos",
    "peso_bruto_kg",
    "peso_neto_kg",
    "cantidad_unidades",
    "84_valor_cif_fob_usd",
    "85_valor_aduanas_pesos",
    "87_valor_aduanas_ajustado_pesos",
    "89_valor_aduanas_ajustado_tga",
]

IGNORED_ITEM_KEYS = {"factura_asociada", "linea_factura_asociada", "tributos_item", "numero"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte los outputs de DUA items a CSV/XLSX con filas por item."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Directorio base del proyecto (por defecto el cwd).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output/dua_items.csv",
        help="Ruta del CSV de salida (por defecto output/dua_items.csv).",
    )
    parser.add_argument(
        "--output-excel",
        help="Ruta de salida opcional para XLSX. Requiere pandas instalado.",
    )
    return parser.parse_args()


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.replace(",", "").strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None
    return None


def format_tributo_columns(names: Iterable[str]) -> List[str]:
    columns: List[str] = []
    for name in names:
        columns.append(f"{name} (%) - alicuota")
        columns.append(f"{name} - B imp")
        columns.append(f"{name} - monto total")
    return columns


def collect_tributo_names(output_root: Path) -> List[str]:
    names: Set[str] = set()
    for json_path in output_root.rglob("*.json"):
        payload = json.loads(json_path.read_text())
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            for item in entry.get("data", {}).get("lista_items") or []:
                for trib in item.get("tributos_item") or []:
                    nombre = str(trib.get("nombre", "")).strip()
                    if not nombre or nombre.lower() == "null":
                        continue
                    names.add(nombre)
    return sorted(names)


def collect_item_keys(output_root: Path) -> Set[str]:
    keys: Set[str] = set()
    for json_path in output_root.rglob("*.json"):
        payload = json.loads(json_path.read_text())
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            for item in entry.get("data", {}).get("lista_items") or []:
                keys.update(item.keys())
    return keys


def compute_columns(extra_item_keys: Set[str], tributo_names: List[str]) -> List[str]:
    base_item_keys = set(ITEM_COLUMNS_ORDERED)
    extras_sorted = sorted(
        k for k in extra_item_keys if k not in base_item_keys and k not in IGNORED_ITEM_KEYS
    )
    item_columns = ITEM_COLUMNS_ORDERED + extras_sorted
    tributo_columns = format_tributo_columns(tributo_names)
    return LEADING_COLUMNS + item_columns + tributo_columns


def collect_rows(output_root: Path, columns: List[str], tributo_names: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tributo_columns = set(format_tributo_columns(tributo_names))
    item_columns = [c for c in columns if c not in LEADING_COLUMNS and c not in tributo_columns]

    for json_path in sorted(output_root.rglob("*.json")):
        payload = json.loads(json_path.read_text())
        entries = payload if isinstance(payload, list) else [payload]

        for entry in entries:
            data = entry.get("data", {})
            items = data.get("lista_items") or [{}]
            numero_dua = data.get("numero_dua", "")

            for item in items:
                factura = item.get("factura_asociada", "")
                lineas_raw = item.get("linea_factura_asociada")
                if isinstance(lineas_raw, list) and lineas_raw:
                    lineas = lineas_raw
                elif lineas_raw is None:
                    lineas = [""]
                else:
                    lineas = [lineas]

                for linea in lineas:
                    row = {col: "" for col in columns}
                    row["DUA"] = numero_dua
                    row["factura"] = factura
                    row["linea_factura"] = linea

                    for col in item_columns:
                        if col == "dua_item_ix":
                            row[col] = item.get("numero", "")
                        else:
                            row[col] = item.get(col, "")

                    for trib in item.get("tributos_item") or []:
                        name = str(trib.get("nombre", "")).strip()
                        if not name or name.lower() == "null":
                            continue
                        alic = trib.get("alicuota", "")
                        base = trib.get("base_imponible", "")
                        base_num = to_float(base)
                        alic_num = to_float(alic)
                        monto = (
                            base_num * alic_num if base_num is not None and alic_num is not None else None
                        )

                        row[f"{name} (%) - alicuota"] = alic
                        row[f"{name} - B imp"] = base
                        row[f"{name} - monto total"] = f"{monto:.2f}" if monto is not None else ""

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
    output_root = base_dir / "output" / "dua_items"

    if not output_root.exists():
        raise SystemExit(f"No existe la carpeta de salida: {output_root}")

    tributo_names = collect_tributo_names(output_root)
    extra_item_keys = collect_item_keys(output_root)
    columns = compute_columns(extra_item_keys, tributo_names)
    rows = collect_rows(output_root, columns, tributo_names)

    write_csv(Path(args.output), columns, rows)
    if args.output_excel:
        write_excel(Path(args.output_excel), columns, rows)


if __name__ == "__main__":
    main()
