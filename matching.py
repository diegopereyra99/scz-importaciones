#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import curses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# -----------------------------
# Assumptions about your processed mail layout
# -----------------------------
# Root structure like:
# processed_root/
#   <mail_folder_1>/
#      mail.json               (subject, date, ... optional)
#      mail_summary.json     (your summarize_email/v0 output, can be wrapped like [{data,meta}] or plain)
#      attachments/
#         file1.pdf
#         file2.pdf
#      classified/
#         file1.json
#         file2.json
#
# If filenames differ, pass with flags or adapt the constants.

DEFAULT_EMAIL_JSON = "mail.json"
DEFAULT_SUMMARY_JSON = "mail_summary.json"
DEFAULT_CLASSIFIED_DIR = "classified"
DEFAULT_ATTACH_DIR = "attachments"
DEFAULT_ATTACH_CLASSIFY_JSON = DEFAULT_CLASSIFIED_DIR  # can be a directory with per-attachment JSONs or a single JSON

# -----------------------------
# Normalization helpers (MVP)
# -----------------------------
def alnum_upper(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())

def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def dua6_from_value(v: str) -> Optional[str]:
    d = digits_only(v)
    if len(d) >= 6:
        return d[-6:]
    return None

def product_key(v: str) -> Optional[str]:
    # MVP: first strong token
    s = (v or "").upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks[0] if toks else None

def parse_date(s: Optional[str]) -> Optional[str]:
    # store ISO YYYY-MM-DD or ISO datetime; keep simple
    if not s:
        return None
    s = s.strip()
    # common formats: "2025-11-17", "2025-11-17T10:22:00", RFC2822 etc.
    # MVP: keep original if already ISO-like; else try parse loosely.
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s
    # try a couple of common patterns
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    return s  # last resort: keep raw

# -----------------------------
# Data extraction from your summarize_email output
# -----------------------------
def load_json_any(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def unwrap_docflow_output(obj):
    # Your output is usually: [ { "data": {...}, "meta": {...}} ]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "data" in obj[0]:
        return obj[0]["data"], obj[0].get("meta")
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"], obj.get("meta")
    # If it's already the data dict
    if isinstance(obj, dict) and ("referencias" in obj or "proveedor_importacion" in obj):
        return obj, None
    raise ValueError("Unrecognized mail_summary JSON structure")

def extract_keys_from_email_summary(summary_data: dict) -> Tuple[Optional[str], Set[str], Set[str], Set[str], Set[str]]:
    """
    Returns:
      proveedor, dua6_set, invoice_set, ship_set, product_key_set
    """
    prov = summary_data.get("proveedor_importacion")
    dua6 = set()
    invoices = set()
    ships = set()
    products = set()

    for r in summary_data.get("referencias", []) or []:
        t = (r.get("tipo") or "").lower().strip()
        val = (r.get("valor") or "").strip()
        if not t or not val:
            continue
        if t == "dua":
            s = dua6_from_value(val)
            if s:
                dua6.add(s)
        elif t in ("factura", "factura_comercial"):
            k = alnum_upper(val)
            if k:
                invoices.add(k)
        elif t in ("mawb", "hawb", "bl", "guia"):
            k = alnum_upper(val)
            if k:
                ships.add(k)
        elif t in ("producto", "producto_comercial"):
            k = product_key(val)
            if k:
                products.add(k)

    return prov, dua6, invoices, ships, products


def load_email_dates(emails_jsonl_path: Path) -> Dict[str, str]:
    """
    Lee db/emails.jsonl y construye un mapa email_id -> YYYY-MM-DD (solo fecha).
    """
    dates = {}
    if not emails_jsonl_path.exists():
        return dates
    for row in read_jsonl(emails_jsonl_path):
        eid = row.get("email_id")
        dt = row.get("datetime")
        mail_dir = row.get("mail_dir")
        if not eid or not dt:
            continue
        try:
            dates[eid] = dt.split("T", 1)[0]
        except Exception:
            dates[eid] = dt
        # también indexar por nombre de carpeta de mail_dir para lookup robusto
        if mail_dir:
            folder_name = Path(mail_dir).name
            if folder_name not in dates:
                dates[folder_name] = dates[eid]
    return dates

# -----------------------------
# Attachment classify loader (to detect invoice-starter)
# -----------------------------
def unwrap_attachment_classify(obj) -> List[dict]:
    """
    Supports:
      - docflow output list: [ {data:{...}, meta:{docs:[...]}} , ...]
      - single docflow output: [ {data:{...}, meta:{docs:[...]}} ]
      - or list of { "data":{...}} already
    """
    if isinstance(obj, dict) and "data" in obj:
        return [obj]
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and "data" in item:
                out.append(item)
        return out
    return []

def email_has_invoice_attachment(attach_classify_items: List[dict]) -> bool:
    for item in attach_classify_items:
        data = item.get("data") or {}
        cls = (data.get("clase_documento") or "").strip().lower()
        if cls in ("factura", "factura_packing"):
            return True
    return False

# -----------------------------
# Simple JSONL DB
# -----------------------------
def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Folder state (MVP)
# -----------------------------
@dataclass
class FolderState:
    folder_id: str
    proveedor: Optional[str]
    dua6: Set[str]
    invoices: Set[str]
    ships: Set[str]
    products: Set[str]
    emails: List[str]
    attachments: List[str]  # stored as relative paths from processed_root or absolute

    @staticmethod
    def from_dict(d: dict) -> "FolderState":
        return FolderState(
            folder_id=d["folder_id"],
            proveedor=d.get("proveedor"),
            dua6=set(d.get("dua6", [])),
            invoices=set(d.get("invoices", [])),
            ships=set(d.get("ships", [])),
            products=set(d.get("products", [])),
            emails=list(d.get("emails", [])),
            attachments=list(d.get("attachments", [])),
        )

    def to_dict(self) -> dict:
        return {
            "folder_id": self.folder_id,
            "proveedor": self.proveedor,
            "dua6": sorted(self.dua6),
            "invoices": sorted(self.invoices),
            "ships": sorted(self.ships),
            "products": sorted(self.products),
            "emails": self.emails,
            "attachments": self.attachments,
        }

# -----------------------------
# Matcher MVP with indexes
# -----------------------------
class MatcherDB:
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.folders_path = db_dir / "folders.jsonl"
        self.emails_path = db_dir / "emails.jsonl"
        self.assignments_path = db_dir / "assignments.jsonl"
        self.events_path = db_dir / "events.jsonl"

        self.folders: Dict[str, FolderState] = {}
        self.assigned_emails: Set[str] = set()

        # indexes: (prov, key) -> folder_ids
        self.idx_dua = defaultdict(set)
        self.idx_inv = defaultdict(set)
        self.idx_ship = defaultdict(set)

    def load(self):
        for row in read_jsonl(self.folders_path):
            f = FolderState.from_dict(row)
            self.folders[f.folder_id] = f
        for row in read_jsonl(self.assignments_path):
            self.assigned_emails.add(row["email_id"])
        self.rebuild_indexes()

    def rebuild_indexes(self):
        self.idx_dua.clear()
        self.idx_inv.clear()
        self.idx_ship.clear()
        for folder_id, f in self.folders.items():
            if not f.proveedor:
                continue
            prov = f.proveedor
            for d in f.dua6:
                self.idx_dua[(prov, d)].add(folder_id)
            for inv in f.invoices:
                self.idx_inv[(prov, inv)].add(folder_id)
            for sh in f.ships:
                self.idx_ship[(prov, sh)].add(folder_id)

    def new_folder_id(self, prefix="F") -> str:
        # stable short-ish id
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    def create_folder_from_email(
        self,
        email_id: str,
        prov: Optional[str],
        dua6: Set[str],
        inv: Set[str],
        ships: Set[str],
        products: Set[str],
        attachments: List[str],
    ) -> str:
        folder_id = self.new_folder_id()
        f = FolderState(
            folder_id=folder_id,
            proveedor=prov,
            dua6=set(dua6),
            invoices=set(inv),
            ships=set(ships),
            products=set(products),
            emails=[email_id],
            attachments=list(attachments),
        )
        self.folders[folder_id] = f
        append_jsonl(self.folders_path, f.to_dict())
        append_jsonl(self.assignments_path, {"email_id": email_id, "folder_id": folder_id})
        append_jsonl(self.events_path, {"ts": datetime.utcnow().isoformat(), "event": "folder_created", "email_id": email_id, "folder_id": folder_id})
        self.assigned_emails.add(email_id)
        self.rebuild_indexes()
        return folder_id

    def add_email_to_folder(
        self,
        email_id: str,
        folder_id: str,
        prov: Optional[str],
        dua6: Set[str],
        inv: Set[str],
        ships: Set[str],
        products: Set[str],
        attachments: List[str],
        matched_by: str,
        warnings: List[str],
    ):
        f = self.folders[folder_id]
        if not f.proveedor and prov:
            f.proveedor = prov
        f.dua6 |= dua6
        f.invoices |= inv
        f.ships |= ships
        f.products |= products
        f.emails.append(email_id)
        f.attachments.extend(attachments)

        # rewrite folders.jsonl naively: MVP approach
        # (for small dataset, easiest: rewrite whole file)
        write_json(self.db_dir / "folders_snapshot.json", {k: v.to_dict() for k, v in self.folders.items()})
        # also append an event and assignment
        append_jsonl(self.assignments_path, {"email_id": email_id, "folder_id": folder_id})
        append_jsonl(self.events_path, {"ts": datetime.utcnow().isoformat(), "event": "email_matched", "email_id": email_id, "folder_id": folder_id, "matched_by": matched_by, "warnings": warnings})
        self.assigned_emails.add(email_id)
        self.rebuild_indexes()

    def match_email(self, prov: Optional[str], dua6: Set[str], inv: Set[str], ships: Set[str]) -> Tuple[str, List[str]]:
        """
        Returns:
          status: "unassigned" | "ambiguous" | "matched"
          payload: if matched -> [folder_id, matched_by]
                   if ambiguous/unassigned -> list of candidate folder_ids (or empty)
        """
        if not prov:
            return "unassigned", []

        candidates: Set[str] = set()
        matched_by = ""

        # priority: dua6 -> invoice -> ship
        if dua6:
            for d in dua6:
                candidates |= self.idx_dua.get((prov, d), set())
            matched_by = "dua6"
        if not candidates and inv:
            for x in inv:
                candidates |= self.idx_inv.get((prov, x), set())
            matched_by = "invoice"
        if not candidates and ships:
            for s in ships:
                candidates |= self.idx_ship.get((prov, s), set())
            matched_by = "ship"

        if not candidates:
            return "unassigned", []
        if len(candidates) > 1:
            return "ambiguous", sorted(candidates)
        return "matched", [next(iter(candidates)), matched_by]

# -----------------------------
# Processed mail scanning
# -----------------------------
@dataclass
class ProcessedMail:
    email_id: str
    folder_path: Path
    subject: str
    date: Optional[str]
    summary_path: Path
    summary_data: dict
    attachments: List[Path]
    attach_classify_items: List[dict]

def detect_email_id(mail_folder: Path) -> str:
    # Use folder name as id (you said you use subject now; ok)
    return mail_folder.name

def load_processed_mail(
    mail_folder: Path,
    email_json_name: str,
    summary_json_name: str,
    attach_dir_name: str,
    attach_classify_name: str,
    email_dates: Optional[Dict[str, str]] = None,
) -> ProcessedMail:
    email_id = detect_email_id(mail_folder)

    subject = mail_folder.name
    date = (email_dates or {}).get(email_id)

    email_json_path = mail_folder / email_json_name
    if email_json_path.exists():
        ej = load_json_any(email_json_path)
        subject = ej.get("subject") or subject
        if not date:
            date = parse_date(ej.get("date") or ej.get("internal_date") or ej.get("sent_date") or ej.get("received_date"))

    summary_path = mail_folder / summary_json_name
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_path}")
    raw = load_json_any(summary_path)
    summary_data, _meta = unwrap_docflow_output(raw)

    attach_dir = mail_folder / attach_dir_name
    attachments = []
    if attach_dir.exists():
        for p in sorted(attach_dir.glob("*")):
            if p.is_file():
                attachments.append(p)

    attach_classify_items = []
    acp = mail_folder / attach_classify_name
    if acp.exists():
        if acp.is_dir():
            for fp in sorted(acp.glob("*.json")):
                try:
                    attach_classify_items.extend(unwrap_attachment_classify(load_json_any(fp)))
                except Exception as e:
                    print(f"[WARN] failed to load classify json {fp}: {e}", file=sys.stderr)
        else:
            try:
                attach_classify_items = unwrap_attachment_classify(load_json_any(acp))
            except Exception as e:
                print(f"[WARN] failed to load classify json {acp}: {e}", file=sys.stderr)

    return ProcessedMail(
        email_id=email_id,
        folder_path=mail_folder,
        subject=subject,
        date=date,
        summary_path=summary_path,
        summary_data=summary_data,
        attachments=attachments,
        attach_classify_items=attach_classify_items,
    )

def scan_processed_mails(processed_root: Path, **kwargs) -> List[ProcessedMail]:
    mails = []
    for child in sorted(processed_root.iterdir()):
        if child.is_dir():
            try:
                mails.append(load_processed_mail(child, **kwargs))
            except Exception as e:
                # keep going; you can inspect later
                print(f"[WARN] skipping {child}: {e}", file=sys.stderr)
    return mails

# -----------------------------
# Attachment copy into folder dir
# -----------------------------
def copy_attachments_to_folder(db_dir: Path, folder_id: str, mail: ProcessedMail, mode: str = "copy"):
    """
    mode: "copy" or "symlink" (symlink is great locally)
    """
    dest = db_dir / "folder_files" / folder_id / "attachments"
    dest.mkdir(parents=True, exist_ok=True)
    for att in mail.attachments:
        tgt = dest / att.name
        if tgt.exists():
            continue
        if mode == "symlink":
            os.symlink(att.resolve(), tgt)
        else:
            shutil.copy2(att, tgt)

# -----------------------------
# CLI commands
# -----------------------------
def cmd_init(args):
    processed_root = Path(args.processed_root)
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    matcher = MatcherDB(db_dir)
    matcher.load()

    email_dates = load_email_dates(db_dir / "emails.jsonl")

    mails = scan_processed_mails(
        processed_root,
        email_json_name=args.email_json,
        summary_json_name=args.summary_json,
        attach_dir_name=args.attach_dir,
        attach_classify_name=args.attach_classify_json,
        email_dates=email_dates,
    )

    created = 0
    for m in mails:
        if m.email_id in matcher.assigned_emails:
            continue

        prov, dua6, inv, ships, products = extract_keys_from_email_summary(m.summary_data)
        match_info = f"prov={prov or '-'} | dua6={sorted(dua6)} | inv={sorted(inv)} | ships={sorted(ships)}"

        # invoice-starter rule:
        # Solo crear carpeta si hay proveedor y un adjunto clasificado como factura/factura_packing
        has_classify = bool(m.attach_classify_items)
        has_invoice_att = email_has_invoice_attachment(m.attach_classify_items) if has_classify else False

        invoice_trigger = None
        if prov and has_invoice_att:
            invoice_trigger = "invoice attachment (classify) with proveedor"

        if invoice_trigger:
            folder_id = matcher.create_folder_from_email(
                email_id=m.email_id,
                prov=prov,
                dua6=dua6,
                inv=inv,
                ships=ships,
                products=products,
                attachments=[str(p) for p in m.attachments],
            )
            created += 1
            folder_path_hint = db_dir / "folder_files" / folder_id
            print(f"[INIT] created folder {folder_id} from email '{m.subject}' ({m.email_id}) -> {folder_path_hint}")
            print(f"       trigger: {invoice_trigger}")
            print(f"       keys: {match_info}")
            if args.copy_attachments:
                copy_attachments_to_folder(db_dir, folder_id, m, mode=args.copy_mode)

    print(f"[DONE] init created {created} folders.")

def summarize_mail_line(m: ProcessedMail) -> str:
    prov, dua6, inv, ships, products = extract_keys_from_email_summary(m.summary_data)
    return (
        f"date={m.date or '?'} | prov={prov or '-'} | "
        f"dua6={','.join(sorted(dua6)) or '-'} | "
        f"inv={','.join(sorted(inv)) or '-'} | "
        f"ship={','.join(sorted(ships)) or '-'} | "
        f"prod={','.join(sorted(list(products))[:3])}{'...' if len(products)>3 else ''} | "
        f"subject={m.subject}"
    )

def mail_detail_lines(m: ProcessedMail) -> List[str]:
    prov, dua6, inv, ships, products = extract_keys_from_email_summary(m.summary_data)
    lines = [
        f"subject: {m.subject}",
        f"date: {m.date or '-'}",
        f"summary: {m.summary_path}",
        f"proveedor: {prov or '-'}",
        f"dua6: {', '.join(sorted(dua6)) or '-'}",
        f"invoices: {', '.join(sorted(inv)) or '-'}",
        f"ships: {', '.join(sorted(ships)) or '-'}",
        f"products: {', '.join(sorted(products)) or '-'}",
        f"attachments: {len(m.attachments)} file(s)",
    ]
    return lines


def _date_sort_key(date_str: Optional[str]) -> datetime:
    """
    Devuelve un datetime utilizable como key de orden.
    Sin fecha -> datetime.max (queda al final).
    """
    if not date_str:
        return datetime.max
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return datetime.max


def mail_sort_key(m: ProcessedMail):
    prov, dua6, _, _, _ = extract_keys_from_email_summary(m.summary_data)
    dua_flag = 0 if not dua6 else 1  # 0 primero (sin dua)
    return (
        prov or "",
        dua_flag,
        sorted(dua6) if dua6 else [],
        _date_sort_key(m.date),
    )

def select_mail_curses(pending: List[ProcessedMail]) -> Tuple[Optional[int], bool]:
    """
    Returns (selected_index, quit_flag)
    """
    result = {"idx": None, "quit": False}

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)  # highlight
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # detail labels

        idx = 0
        top = 0
        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            visible_rows = max(h - 8, 3)

            # adjust window top to keep cursor visible
            if idx < top:
                top = idx
            elif idx >= top + visible_rows:
                top = idx - visible_rows + 1

            stdscr.addstr(0, 0, "↑/↓ navegar | Enter procesar | q salir")

            for i in range(top, min(len(pending), top + visible_rows)):
                line = summarize_mail_line(pending[i])
                line = (line[: w - 1]) if len(line) >= w else line
                y = 2 + (i - top)
                if i == idx:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(y, 0, line.ljust(w - 1))
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(y, 0, line)

            # detail pane
            detail_start = 2 + visible_rows + 1
            stdscr.hline(detail_start, 0, "-", w - 1)
            detail_lines = mail_detail_lines(pending[idx])
            for j, dl in enumerate(detail_lines[: h - detail_start - 1]):
                try:
                    stdscr.addstr(detail_start + 1 + j, 0, dl[: w - 1], curses.color_pair(2))
                except curses.error:
                    pass

            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                result["quit"] = True
                return
            if ch in (curses.KEY_UP, ord("k")):
                idx = (idx - 1) % len(pending)
            elif ch in (curses.KEY_DOWN, ord("j")):
                idx = (idx + 1) % len(pending)
            elif ch in (curses.KEY_NPAGE,):
                idx = min(len(pending) - 1, idx + visible_rows)
            elif ch in (curses.KEY_PPAGE,):
                idx = max(0, idx - visible_rows)
            elif ch in (curses.KEY_ENTER, 10, 13):
                result["idx"] = idx
                return

    curses.wrapper(_inner)
    return result["idx"], result["quit"]

def cmd_interactive(args):
    processed_root = Path(args.processed_root)
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    matcher = MatcherDB(db_dir)
    matcher.load()

    email_dates = load_email_dates(db_dir / "emails.jsonl")

    mails = scan_processed_mails(
        processed_root,
        email_json_name=args.email_json,
        summary_json_name=args.summary_json,
        attach_dir_name=args.attach_dir,
        attach_classify_name=args.attach_classify_json,
        email_dates=email_dates,
    )

    # filter unassigned
    pending = sorted(
        [m for m in mails if m.email_id not in matcher.assigned_emails],
        key=mail_sort_key,
    )

    if not pending:
        print("No pending emails (all assigned).")
        return

    while True:
        # refresh pending each loop
        pending = sorted(
            [m for m in mails if m.email_id not in matcher.assigned_emails],
            key=mail_sort_key,
        )
        if not pending:
            print("No pending emails remaining.")
            return

        sel_idx, quit_flag = select_mail_curses(pending)
        if quit_flag or sel_idx is None:
            return

        m = pending[sel_idx]
        prov, dua6, inv, ships, products = extract_keys_from_email_summary(m.summary_data)
        match_info = f"prov={prov or '-'} | dua6={sorted(dua6)} | inv={sorted(inv)} | ships={sorted(ships)}"

        # If it's invoice-starter, create folder immediately
        # Solo si hay proveedor y un adjunto clasificado como factura/factura_packing
        has_classify = bool(m.attach_classify_items)
        has_invoice_att = email_has_invoice_attachment(m.attach_classify_items) if has_classify else False

        invoice_trigger = None
        if prov and has_invoice_att:
            invoice_trigger = "invoice attachment (classify) with proveedor"

        if invoice_trigger:
            folder_id = matcher.create_folder_from_email(
                email_id=m.email_id,
                prov=prov,
                dua6=dua6,
                inv=inv,
                ships=ships,
                products=products,
                attachments=[str(p) for p in m.attachments],
            )
            folder_path_hint = db_dir / "folder_files" / folder_id
            print(f"[ACTION] Created new folder {folder_id} ({invoice_trigger}) -> {folder_path_hint}")
            print(f"         keys: {match_info}")
            if args.copy_attachments:
                copy_attachments_to_folder(db_dir, folder_id, m, mode=args.copy_mode)
            input("\n[Enter] para continuar...")
            continue

        status, payload = matcher.match_email(prov, dua6, inv, ships)

        if status == "unassigned":
            print("[RESULT] UNASSIGNED: no candidate folder found by dua6/invoice/ship.")
            print(f"         keys: {match_info}")
            print("         -> leaving as pending (you can create a folder manually later).")
            # optionally: ask to create folder anyway
            if args.allow_force_new and prov:
                yn = input("Force-create a new folder for this email? [y/N]: ").strip().lower()
                if yn == "y":
                    folder_id = matcher.create_folder_from_email(
                        email_id=m.email_id,
                        prov=prov,
                        dua6=dua6,
                        inv=inv,
                        ships=ships,
                        products=products,
                        attachments=[str(p) for p in m.attachments],
                    )
                    folder_path_hint = db_dir / "folder_files" / folder_id
                    print(f"[ACTION] Forced new folder {folder_id} -> {folder_path_hint}")
                    print(f"         keys: {match_info}")
                    if args.copy_attachments:
                        copy_attachments_to_folder(db_dir, folder_id, m, mode=args.copy_mode)
            input("\n[Enter] para continuar...")
            continue

        if status == "ambiguous":
            print("[RESULT] AMBIGUOUS candidates:", payload)
            print(f"         keys: {match_info}")
            print("         -> leaving as pending.")
            input("\n[Enter] para continuar...")
            continue

        folder_id, matched_by = payload[0], payload[1]

        # simple warnings: if mail has multiple strong keys and only one overlaps folder
        f = matcher.folders[folder_id]
        warnings = []
        if prov and f.proveedor and prov != f.proveedor:
            warnings.append("provider_mismatch (should not happen due to index)")

        # note: In MVP we won't compute overlap; just log what matched by
        matcher.add_email_to_folder(
            email_id=m.email_id,
            folder_id=folder_id,
            prov=prov,
            dua6=dua6,
            inv=inv,
            ships=ships,
            products=products,
            attachments=[str(p) for p in m.attachments],
            matched_by=matched_by,
            warnings=warnings,
        )
        print(f"[RESULT] MATCHED -> folder {folder_id} by {matched_by}")
        print(f"         keys: {match_info}")
        if warnings:
            print("         warnings:", warnings)

        if args.copy_attachments:
            copy_attachments_to_folder(db_dir, folder_id, m, mode=args.copy_mode)
        input("\n[Enter] para continuar...")

def cmd_clean(args):
    db_dir = Path(args.db_dir)
    targets = [
        db_dir / "folders.jsonl",
        db_dir / "assignments.jsonl",
        db_dir / "events.jsonl",
        db_dir / "folders_snapshot.json",
    ]
    removed = []
    for t in targets:
        if t.exists():
            t.unlink()
            removed.append(str(t))
    folder_files_dir = db_dir / "folder_files"
    if folder_files_dir.exists():
        shutil.rmtree(folder_files_dir)
        removed.append(str(folder_files_dir))

    if removed:
        print("[CLEAN] removed:", ", ".join(removed))
    else:
        print("[CLEAN] nothing to remove.")

def build_argparser():
    p = argparse.ArgumentParser(description="MVP mail->folder matcher (Scienza import)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--processed-root", default="data/emails/processed", help="Path to processed mails root folder")
        sp.add_argument("--db-dir", default="db", help="Path to fake db folder")
        sp.add_argument("--email-json", default=DEFAULT_EMAIL_JSON)
        sp.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
        sp.add_argument("--attach-classify-json", default=DEFAULT_ATTACH_CLASSIFY_JSON,
                        help="Path to classified outputs: directory with per-attachment JSONs or a single JSON file")
        sp.add_argument("--attach-dir", default=DEFAULT_ATTACH_DIR)
        sp.add_argument("--fallback-summary-invoice", action="store_true",
                        help="If attachments_classify.json missing, create folder if summary has proveedor+factura ref")
        sp.add_argument("--copy-attachments", action="store_true",
                        help="Copy or symlink attachments into db/folder_files/<folder_id>/attachments")
        sp.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")

    sp_init = sub.add_parser("init", help="Create initial folders from invoice-starter emails")
    add_common(sp_init)
    sp_init.set_defaults(func=cmd_init)

    sp_int = sub.add_parser("interactive", help="Process one email at a time, match into folders, update db")
    add_common(sp_int)
    sp_int.add_argument("--allow-force-new", action="store_true", help="If unassigned, prompt to force-create folder")
    sp_int.set_defaults(func=cmd_interactive)

    sp_clean = sub.add_parser("clean", help="Remove matcher-created db artifacts")
    sp_clean.add_argument("--db-dir", default="db", help="Path to fake db folder")
    sp_clean.set_defaults(func=cmd_clean)

    return p

def main():
    args = build_argparser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
