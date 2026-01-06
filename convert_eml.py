from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import json
import re

def sanitize_subject(s: str) -> str:
    s = (s or "no_subject").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\- ]+", "_", s, flags=re.UNICODE)  # reemplaza raro por _
    s = s.replace(" ", "_")
    return s[:80] or "no_subject"  # límite para no romper paths

def parse_eml(eml_path: Path):
    msg = BytesParser(policy=policy.default).parsebytes(eml_path.read_bytes())

    # Date
    date_iso = None
    if msg.get("Date"):
        try:
            date_iso = parsedate_to_datetime(msg["Date"]).isoformat()
        except Exception:
            pass

    text_body = None
    html_body = None

    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get_content_disposition() or "").lower()
            if disp == "attachment":
                continue
            if ctype == "text/plain" and text_body is None:
                text_body = part.get_content()
            elif ctype == "text/html" and html_body is None:
                html_body = part.get_content()
    else:
        ctype = (msg.get_content_type() or "").lower()
        if ctype == "text/html":
            html_body = msg.get_content()
        else:
            text_body = msg.get_content()

    base_for_snippet = text_body or re.sub("<[^>]+>", " ", html_body or "")
    snippet = re.sub(r"\s+", " ", (base_for_snippet or "")).strip()[:500]

    core = {
        "message_id": msg.get("Message-ID"),
        "headers": {
            "date": date_iso,
            "from": msg.get("From"),
            "to": msg.get_all("To", []),
            "subject": msg.get("Subject"),
        },
        "body": {"text": text_body, "html": html_body, "snippet": snippet},
    }
    return core, msg


def extract_attachments(msg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    attachments = []
    idx = 0

    for part in msg.walk():
        filename = part.get_filename()
        if not filename:
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        idx += 1
        path = out_dir / filename
        path.write_bytes(payload)

        attachments.append({
            "attachment_id": f"att_{idx:02d}",
            "filename": filename,
            "mime_type": part.get_content_type(),
            "relative_path": f"attachments/{filename}"
        })

    return attachments


def process_eml_folder(eml_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []

    for eml_path in sorted(eml_dir.glob("*.eml")):
        
        mail_core, msg = parse_eml(eml_path)
        
        mail_id = f"mail_{uuid4().hex[:8]}"
        subject_safe = sanitize_subject(mail_core["headers"]["subject"])
        mail_dir = out_dir / f"{subject_safe}__{mail_id}"

        attachments_dir = mail_dir / "attachments"
        mail_dir.mkdir(parents=True, exist_ok=True)

        attachments = extract_attachments(msg, attachments_dir)

        mail_json = {
            "mail_id": mail_id,
            "message_id": mail_core["message_id"],
            "source": {
                "type": "eml",
                "filename": eml_path.name
            },
            **mail_core,
            "attachments": attachments,
            "meta": {
                "parsed_at": datetime.utcnow().isoformat()
            }
        }

        (mail_dir / "mail.json").write_text(
            json.dumps(mail_json, indent=2, ensure_ascii=False)
        )

        index.append({
            "mail_id": mail_id,
            "date": mail_core["headers"]["date"],
            "subject": mail_core["headers"]["subject"],
            "path": mail_id
        })

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Procesa archivos .eml en una carpeta y extrae sus datos y adjuntos."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Carpeta de entrada con archivos .eml"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Carpeta de salida para los datos procesados"
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"La carpeta de entrada no existe o no es un directorio: {input_dir}")

    process_eml_folder(input_dir, output_dir)