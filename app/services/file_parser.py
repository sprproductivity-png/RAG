import io
import os
import json
import csv
from typing import Any, Dict, Tuple
from fastapi import HTTPException, status
import pypdf

from app.core.logging import logger

def parse_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parses uploaded file bytes into (extracted_text, document_title, metadata).
    Supports: .txt, .md, .markdown, .pdf, .csv, .tsv, .json
    """
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty."
        )

    base_name, ext = os.path.splitext(filename)
    ext = ext.lower()
    inferred_title = base_name.replace("_", " ").replace("-", " ").strip()

    metadata: Dict[str, Any] = {
        "filename": filename,
        "file_size": len(file_bytes),
        "file_type": ext,
        "title": inferred_title,
    }

    try:
        if ext in [".txt", ".md", ".markdown", ".rst"]:
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1", errors="replace")

        elif ext == ".pdf":
            pdf_stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(pdf_stream)
            pages_text = []
            for page_idx, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                if page_content.strip():
                    pages_text.append(f"--- Page {page_idx + 1} ---\n{page_content.strip()}")
            extracted_text = "\n\n".join(pages_text)
            metadata["page_count"] = len(reader.pages)

        elif ext in [".csv", ".tsv"]:
            delimiter = "\t" if ext == ".tsv" else ","
            try:
                decoded_str = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                decoded_str = file_bytes.decode("latin-1", errors="replace")

            reader = csv.reader(io.StringIO(decoded_str), delimiter=delimiter)
            rows = list(reader)
            if not rows:
                extracted_text = ""
            else:
                header = rows[0]
                row_lines = []
                for row in rows[1:]:
                    row_desc = ", ".join(f"{h}: {v}" for h, v in zip(header, row) if v)
                    if row_desc:
                        row_lines.append(row_desc)
                extracted_text = "\n".join(row_lines)
            metadata["row_count"] = len(rows)

        elif ext == ".json":
            try:
                decoded_str = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                decoded_str = file_bytes.decode("latin-1", errors="replace")

            parsed = json.loads(decoded_str)
            if isinstance(parsed, list):
                extracted_text = "\n\n".join(
                    json.dumps(item, indent=2, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
                    for item in parsed
                )
            elif isinstance(parsed, dict):
                extracted_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            else:
                extracted_text = str(parsed)

        else:
            # Fallback text decoder
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1", errors="replace")

        cleaned_text = extracted_text.strip()
        if not cleaned_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not extract readable text from '{filename}'."
            )

        logger.info(f"Parsed uploaded file '{filename}' ({len(file_bytes)} bytes) -> {len(cleaned_text)} characters")
        return cleaned_text, inferred_title, metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing file '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process file '{filename}': {str(e)}"
        )
