#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

REFERENCE_ORDER = [
    "dua",
    "mawb",
    "hawb",
    "bl",
    "factura",
    "producto",
    "oc_scienza",
    "peso_total_kg",
]
REFERENCE_ORDER_INDEX = {ref: idx for idx, ref in enumerate(REFERENCE_ORDER)}


def setup_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

# ---------------------------
# Utils: JSONL
# ---------------------------

def jsonl_load_ids(path: Path, id_field: str) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if id_field in obj and obj[id_field]:
                    ids.add(obj[id_field])
            except Exception:
                continue
    return ids

def jsonl_append(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------------------
# Body cleaning
# ---------------------------

RE_CUT_MARKERS = [
    r"^-{2,}\s*Original Message\s*-{2,}$",
    r"^-{2,}\s*Mensaje original\s*-{2,}$",
    r"^-{2,}\s*Forwarded message\s*-{2,}$",
    r"^----------\s*Forwarded message\s*----------$",
    r"^On .* wrote:$",
    r"^\s*De:\s.*$",
    r"^\s*From:\s.*$",
    r"^\s*Enviado:\s.*$",
    r"^\s*Sent:\s.*$",
    r"^\s*Para:\s.*$",
    r"^\s*To:\s.*$",
    r"^\s*Asunto:\s.*$",
    r"^\s*Subject:\s.*$",
]

CUT_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in RE_CUT_MARKERS]

def clean_body_text(text: str) -> str:
    """
    Heurística: quedarse con la parte "nueva" del mail.
    Si no encuentra markers, devuelve todo.
    """
    if not text:
        return ""
    lines = text.splitlines()
    cut_idx = None
    for i, line in enumerate(lines):
        for rx in CUT_REGEXES:
            if rx.match(line.strip()):
                cut_idx = i
                break
        if cut_idx is not None:
            break
    if cut_idx is not None and cut_idx > 0:
        lines = lines[:cut_idx]
    cleaned = "\n".join(lines).strip()
    # limpieza suave de espacios
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned

# ---------------------------
# Mail parsing from mail.json (Script 1 output)
# ---------------------------

def load_mail_json(mail_json_path: Path) -> dict:
    return json.loads(mail_json_path.read_text(encoding="utf-8"))

def get_email_id(mail_json: dict) -> str:
    # en tu script 1 ya lo guardaste
    return mail_json["mail_id"]

def get_body_text(mail_json: dict) -> str:
    body = mail_json.get("body") or {}
    # preferir text; si no hay, usar html (sin parsear HTML en 2A, por ahora)
    text = body.get("text") or ""
    if text.strip():
        return text
    html = body.get("html") or ""
    # fallback tosco: strip tags (suficiente para POC)
    if html.strip():
        html = re.sub("<[^>]+>", " ", html)
        html = re.sub(r"\s+", " ", html)
        return html
    return ""

def get_headers(mail_json: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    headers = mail_json.get("headers") or {}
    dt = headers.get("date")
    sender = headers.get("from")
    subject = headers.get("subject")
    message_id = mail_json.get("message_id")
    return dt, sender, subject, message_id

def iter_attachments(mail_json: dict, mail_dir: Path) -> List[dict]:
    """
    Devuelve lista con campos + path absoluto.
    """
    out = []
    for att in mail_json.get("attachments") or []:
        rel = att.get("relative_path")
        abs_path = (mail_dir / rel).resolve()
        out.append({
            "attachment_id_local": att.get("attachment_id"),  # "att_01"
            "filename": att.get("filename"),
            "mime_type": att.get("mime_type"),
            "relative_path": rel,
            "abs_path": str(abs_path),
        })
    return out

# ---------------------------
# Docflow classify runner
# ---------------------------

def run_docflow_classify(attachments_glob: str, classify_dir: Path, workers: int):
    classify_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in glob(attachments_glob) if Path(p).is_file()]
    if not files:
        logger.info("No attachments to classify for pattern %s", attachments_glob)
        return

    logger.info("Classifying %d attachment(s) into %s", len(files), classify_dir)

    def _run_one(file_path: Path) -> bool:
        out = classify_dir / f"{file_path.stem}.json"
        logger.debug("Running docflow classify for %s -> %s", file_path.name, out.name)
        cmd_args = [
            "docflow", "run", "classify",
            str(file_path),
            "--output-format", "json",
            "--output-path", str(out),
        ]
        try:
            subprocess.run(cmd_args, check=True)
            logger.info("docflow classify ok: %s -> %s", file_path.name, out.name)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("docflow classify failed for %s (exit %s)", file_path, exc.returncode)
            return False
        except Exception:
            logger.exception("Unexpected error running docflow classify for %s", file_path)
            return False

    if workers <= 1:
        results = [_run_one(file_path) for file_path in files]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, f) for f in files]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

    success = sum(1 for r in results if r)
    failures = len(results) - success
    logger.info("Docflow classify finished: %d ok, %d failed", success, failures)

def run_docflow_summarize(summary_input_fp: Union[Path, str]):
    logger.info("Running docflow summarize_email for %s", summary_input_fp)
    dirname = Path(summary_input_fp).parent
    out_fp = dirname / "mail_summary.json"
    cmd_args = [
        "docflow", "run", "summarize_email",
        str(summary_input_fp),
        "--output-format", "json",
        "--output-path", str(out_fp)
    ]
    
    try:
        subprocess.run(cmd_args, check=True)
        logger.info("docflow summarize_email completed successfully")
    except subprocess.CalledProcessError as exc:
        logger.error("docflow summarize_email failed (exit %s)", exc.returncode)
    except Exception:
        logger.exception("Unexpected error running docflow summarize_email")
    

def find_classify_output_for_attachment(classify_dir: Path, attachment_filename: str, attachment_id_local: str) -> Optional[Path]:
    """
    Docflow puede nombrar outputs de varias maneras.
    Estrategia robusta:
    - si existe att_01.json, usarlo
    - si existe <stem>.json, usarlo
    - si existe algo que contenga el stem, usar el primero
    """
    stem = Path(attachment_filename).stem
    candidates = [
        classify_dir / f"{attachment_id_local}.json",
        classify_dir / f"{stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: buscar por stem substring
    hits = list(classify_dir.glob(f"*{stem}*.json"))
    if hits:
        return hits[0]
    return None

def parse_docflow_classify_json(path: Path) -> dict:
    """
    Tu output parece ser una lista con 1 elemento:
    [{"data": {...}, "meta": {...}}]
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list) and obj:
        return obj[0]
    if isinstance(obj, dict):
        return obj
    return {}

def sort_referencias(referencias: Optional[list]) -> List[dict]:
    """
    Ordena referencias siguiendo el orden del esquema.
    Mantiene estabilidad para referencias del mismo tipo según el orden de entrada.
    """
    if not isinstance(referencias, list):
        return []
    filtered = [
        (idx, r)
        for idx, r in enumerate(referencias)
        if isinstance(r, dict) and r.get("tipo") and r.get("valor")
    ]
    filtered.sort(
        key=lambda item: (
            REFERENCE_ORDER_INDEX.get(item[1].get("tipo"), len(REFERENCE_ORDER_INDEX)),
            item[0],
        )
    )
    return [ref for _, ref in filtered]

def flatten_classify_record(rec: dict) -> dict:
    data = rec.get("data") or {}
    meta = rec.get("meta") or {}
    identificador = data.get("identificador") or data.get("numero_principal")
    out = {
        "clase_documento": data.get("clase_documento"),
        "descripcion": data.get("descripcion"),
        "identificador": identificador,
        # mantener numero_principal por compatibilidad con datos existentes
        "numero_principal": data.get("numero_principal") or identificador,
        "fecha_documento": data.get("fecha_documento"),
        "proveedor_importacion": data.get("proveedor_importacion"),
        "referencias": sort_referencias(data.get("referencias")),
        # opcionales, si vienen
        "moneda": data.get("moneda"),
        "importe_total": data.get("importe_total"),
        "peso_total": data.get("peso_total"),
        "oc_scienza": data.get("oc_scienza"),
        "profile": meta.get("profile"),
        "mode": meta.get("mode"),
        "model": meta.get("model"),
    }
    return out

def load_classify_bullets(classify_dir: Path) -> str:
    """
    Devuelve un string listo para el prompt:
    - clase_documento: ...
      descripcion: ...
      numero_principal: ...
      proveedor_importacion: ...
    """
    bullets = []

    for p in sorted(classify_dir.glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))

        if isinstance(obj, list) and obj:
            data = obj[0].get("data", {})
        elif isinstance(obj, dict):
            data = obj.get("data", {})
        else:
            continue
        
        
        lines = []
        if "clase_documento" in data:
            if data["clase_documento"] == "adjunto_irrelevante":
                continue  # skip irrelevantes
            lines.append(f"- clase_documento: {data['clase_documento']}")
        else:
            lines.append("- clase_documento: desconocido")

        identificador = data.get("identificador") or data.get("numero_principal")
        for key in [
            "descripcion",
            "identificador",
            "fecha_documento",
            "proveedor_importacion"
        ]:
            value = data.get(key) if key != "identificador" else identificador
            if value:
                lines.append(f"  {key}: {value}")

        referencias = sort_referencias(data.get("referencias"))
        if referencias:
            lines.append("  referencias:")
            for ref in referencias:
                lines.append(f"    - {ref['tipo']}: {ref.get('valor')}")

        bullets.append("\n".join(lines))

    return "\n".join(bullets)


def build_summary_input_txt(
    mail_dir: Path,
) -> Path:
    """
    Construye processed_mails/<mail_dir>/summary_input.txt
    y devuelve el path.
    """
    mail_json = json.loads((mail_dir / "mail.json").read_text(encoding="utf-8"))

    sender = mail_json.get("headers", {}).get("from", "")
    subject = mail_json.get("headers", {}).get("subject", "")

    body_clean_path = mail_dir / "mail_body_clean.txt"
    body_clean = body_clean_path.read_text(encoding="utf-8") if body_clean_path.exists() else ""

    classify_dir = mail_dir / "classified"
    attachments_bullets = load_classify_bullets(classify_dir) if classify_dir.exists() else ""

    txt = []

    txt.append("\nSender:")
    txt.append(sender)

    txt.append("\nSubject:")
    txt.append(subject)

    txt.append("\nBody (clean):")
    txt.append(body_clean)

    txt.append("\nAdjuntos clasificados:")
    txt.append(attachments_bullets if attachments_bullets else "- ninguno")

    out_path = mail_dir / "summary_input.txt"
    out_path.write_text("\n".join(txt), encoding="utf-8")

    return out_path


# ---------------------------
# Main 2A
# ---------------------------

def process_one_mail(mail_dir: Path, db_emails: Path, db_attachments: Path, workers: int, force: bool):
    mail_json_path = mail_dir / "mail.json"
    if not mail_json_path.exists():
        logger.warning("Skipping %s: mail.json not found", mail_dir)
        return

    mail_json = load_mail_json(mail_json_path)
    email_id = get_email_id(mail_json)
    logger.info("Processing email %s from %s", email_id, mail_dir.name)

    # Idempotencia por email_id
    existing_ids = jsonl_load_ids(db_emails, "email_id")
    if (not force) and (email_id in existing_ids):
        logger.info("Skipping %s: already present in %s", email_id, db_emails.name)
        return

    dt, sender, subject, message_id = get_headers(mail_json)

    # Body clean
    body_text = get_body_text(mail_json)
    body_clean = body_text #clean_body_text(body_text)
    logger.debug("Body extracted for %s (%d chars)", email_id, len(body_clean))
    body_clean_path = mail_dir / "mail_body_clean.txt"
    body_clean_path.write_text(body_clean, encoding="utf-8")

    atts = iter_attachments(mail_json, mail_dir)
    logger.info("Found %d attachment(s) for %s", len(atts), email_id)

    # Run classify for all attachments (glob)
    attachments_dir = mail_dir / "attachments"
    classify_dir = mail_dir / "classified"
    if False and attachments_dir.exists():
        # ojo: el glob lo resuelve el shell si lo pasás sin expandir.
        # docflow aparentemente acepta wildcard. Para estar seguros, usamos el patrón como string.
        attachments_glob = str(attachments_dir / "*")
        run_docflow_classify(attachments_glob, classify_dir, workers)
    else:
        logger.info("No attachments directory found for %s", mail_dir)

    summary_input_fp = build_summary_input_txt(mail_dir)
    logger.info("Built summary input txt for %s: %s", email_id, summary_input_fp)

    run_docflow_summarize(summary_input_fp)
    
    if (mail_dir / "mail_summary.json").exists():
        summ_fp = mail_dir / "mail_summary.json"
        try:
            summary_json = json.loads(summ_fp.read_text(encoding="utf-8"))
            if isinstance(summary_json, list) and summary_json:
                summary_obj = summary_json[0]
            elif isinstance(summary_json, dict):
                summary_obj = summary_json
            else:
                summary_obj = {}

            data = summary_obj.get("data", {}) if isinstance(summary_obj, dict) else {}
            proveedor = data.get("proveedor_importacion")
            referencias = sort_referencias(data.get("referencias"))
            notas = data.get("notas")

            lines = []
            lines.append(f"proveedor_importacion: {proveedor or 'null'}")
            lines.append("referencias:")
            if referencias:
                for ref in referencias:
                    lines.append(f"- {ref['tipo']}: {ref.get('valor')}")
            else:
                lines.append("- ninguna")
            if notas:
                lines.append(f"notas: {notas}")

            summary_md_path = mail_dir / "mail_summary.md"
            summary_md_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Wrote mail summary markdown for %s: %s", email_id, summary_md_path)
        except Exception:
            logger.exception("Error parsing mail_summary.json for %s", email_id)
        
    
    # Write email row
    email_row = {
        "email_id": email_id,
        "message_id": message_id,
        "datetime": dt,
        "sender": sender,
        "subject": subject,
        "mail_dir": str(mail_dir),
        "mail_json_path": str(mail_json_path),
        "body_clean_path": str(body_clean_path),
        # summary vendrá luego (perfil aparte)
        "summary_path": str(mail_dir / "mail_summary.md"),
        "created_at": datetime.utcnow().isoformat()
    }
    jsonl_append(db_emails, email_row)

    # Write attachments rows (+ classify fields)
    for att in atts:
        attachment_id = f"{email_id}:{att['attachment_id_local']}"
        classify_path = find_classify_output_for_attachment(
            classify_dir, att["filename"], att["attachment_id_local"]
        )

        flat = {}
        if classify_path and classify_path.exists():
            rec = parse_docflow_classify_json(classify_path)
            flat = flatten_classify_record(rec)
        else:
            logger.warning(
                "Classify output not found for attachment %s (%s)",
                att["filename"],
                att["attachment_id_local"],
            )

        row = {
            "attachment_id": attachment_id,
            "email_id": email_id,
            "attachment_id_local": att["attachment_id_local"],
            "filename": att["filename"],
            "mime_type": att["mime_type"],
            "relative_path": att["relative_path"],
            "abs_path": att["abs_path"],
            "classify_json_path": str(classify_path) if classify_path else None,
            **flat,
            "created_at": datetime.utcnow().isoformat()
        }
        jsonl_append(db_attachments, row)
    logger.info("Email %s processed: wrote %d attachment row(s)", email_id, len(atts))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-mails-dir", required=True, help="Ruta a processed_mails/")
    p.add_argument("--db-dir", default="db", help="Directorio para jsonl")
    p.add_argument("--workers", type=int, default=int(os.environ.get("DOCFLOW_WORKERS", "8")))
    p.add_argument("--force", action="store_true", help="Reprocesar aunque exista en emails.jsonl")
    p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = p.parse_args()

    setup_logging(args.log_level)

    processed_dir = Path(args.processed_mails_dir).resolve()
    db_dir = Path(args.db_dir).resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    db_emails = db_dir / "emails.jsonl"
    db_attachments = db_dir / "attachments.jsonl"

    mail_dirs = [p for p in processed_dir.iterdir() if p.is_dir()]
    mail_dirs.sort()

    logger.info("Starting processing of %d mail(s) from %s", len(mail_dirs), processed_dir)
    for mail_dir in mail_dirs:
        process_one_mail(mail_dir, db_emails, db_attachments, args.workers, args.force)

    logger.info("Done. Wrote:\n- %s\n- %s", db_emails, db_attachments)

if __name__ == "__main__":
    main()
