"""Generated-only PDF text, provenance and historical-page precheck.

This is not visual acceptance. It never reads market sources, fits models,
changes publication files, or writes final_pdf_qa.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "research/runs/FROZEN_RISK_03_1_B5_v1"
PDF = ROOT / "output/pdf/Kronos_Frozen_Risk_03_1_Report_v1.pdf"
OLDPDF = ROOT / "output/pdf/Kronos_Frozen_Risk_03_Report_v2.pdf"
MD = RUN / "reports/Kronos_Frozen_Risk_03_1_Report_v1.md"
RENDERS = ROOT / "tmp/pdfs/frozen_risk03_1_v1"
FAMILIES = ["persistence", "ewma", "har", "R1", "R2", "B2",
            "random_s17", "random_s29", "random_s43", "B5", "B5_s17", "B5_s29", "B5_s43"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compact(text):
    return "".join(text.split())


def fmt(value, digits=7):
    value = float(value)
    if not math.isfinite(value):
        return "未定义"
    return f"{value:.{digits}f}"


def _bound(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Hash manifest path escapes intended directory: " + name)
    return path


def check_generated():
    destination = RUN / "reports/pdf_text_precheck.json"
    if destination.exists():
        raise FileExistsError("Refuse to overwrite precheck evidence: " + str(destination))
    record = {"status": "FAIL", "scope": "generated_pdf_text_and_byte_precheck_only",
              "visual_acceptance": False, "final_pdf_qa_written": False,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
              "limitations": ["Text geometry does not detect all overlap or visual defects; root must inspect rendered pages.",
                              "The historical appendix is tested by exact extracted text and RGB pixel samples at scale 1.3."]}
    checks = record["checks"]

    def check(name, condition, detail=None):
        item = {"check": name, "status": "PASS" if condition else "FAIL"}
        if detail is not None:
            item["detail"] = detail
        checks.append(item)

    try:
        sys.path.insert(0, str(ROOT / "tmp/pdfs/deps"))
        import pymupdf as fitz
        manifest = read(RUN / "reports/publication_manifest.json")
        summary = read(RUN / "metrics/summary.json")
        selected = read(RUN / "models/selected.json")
        with (RUN / "metrics/quarter_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        record["publication_manifest_sha256"] = sha(RUN / "reports/publication_manifest.json")
        record["pdf_sha256"] = sha(PDF)
        check("pdf_sha256", record["pdf_sha256"] == manifest["pdf_sha256"])
        check("markdown_sha256", sha(MD) == manifest["md_sha256"])
        check("original_pdf_sha256", sha(OLDPDF) == manifest["original_pdf_sha256"])
        check("original_markdown_sha256", sha(ROOT / "research/runs/FROZEN_RISK_03_v1/reports/Kronos_Frozen_Risk_03_Report_v2.md") == manifest["original_md_sha256"])
        check("publication_code_sha256", sha(ROOT / "research/delivery/frozen_risk_03_1_report.py") == manifest["publication_code_sha256"])
        for key, base in (("input_sha256", ROOT),
                          ("figure_sha256", RUN / "figures/publication"),
                          ("render_sha256", RENDERS)):
            check(key + "_nonempty", bool(manifest[key]))
            for name, digest in manifest[key].items():
                path = _bound(base, name)
                check(key + ":" + name, path.is_file() and sha(path) == digest)
        check("evidence_class", manifest["evidence_class"] == summary["evidence_class"] ==
              "POST_HOC_EXPLORATORY_NOT_NEW_INDEPENDENT_CONFIRMATION")
        check("original_primary_status", manifest["original_primary_status"] == summary["original_primary_status"])
        mdtext = MD.read_text(encoding="utf-8")
        check("markdown_no_replacement_character", "\ufffd" not in mdtext)
        with fitz.open(PDF) as document, fitz.open(OLDPDF) as old:
            check("fixed_page_counts", (len(document), len(old), manifest["supplement_pages"],
                  manifest["historical_3_0_pages"], manifest["total_pages"],
                  manifest["historical_appendix_start_page"]) == (39, 25, 14, 25, 39, 15))
            texts = []
            for index, page in enumerate(document):
                text = page.get_text("text")
                texts.append(text)
                check(f"page_{index+1}_nonempty_text", bool(text.strip()))
                check(f"page_{index+1}_no_replacement_character", "\ufffd" not in text)
                overflow = []
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if not span["text"].strip():
                                continue
                            x0, y0, x1, y1 = span["bbox"]
                            if x0 < -0.5 or y0 < -0.5 or x1 > page.rect.width + 0.5 or y1 > page.rect.height + 0.5:
                                overflow.append({"text": span["text"], "bbox": list(span["bbox"])})
                check(f"page_{index+1}_text_within_page", not overflow, overflow if overflow else None)
                if index < 14:
                    check(f"page_{index+1}_POST_HOC_marker", "POST-HOC" in text)
                render = RENDERS / f"page_{index+1:02d}.png"
                check(f"page_{index+1}_render_in_manifest", render.name in manifest["render_sha256"] and render.is_file())
                if render.is_file():
                    actual = page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False, colorspace=fitz.csRGB)
                    saved = fitz.Pixmap(str(render))
                    check(f"page_{index+1}_saved_render_matches_pdf",
                          (actual.width, actual.height, actual.n, actual.samples) ==
                          (saved.width, saved.height, saved.n, saved.samples))
            outline = manifest["supplement_outline"]
            check("outline_14_pages", [entry["page"] for entry in outline] == list(range(1, 15)))
            for entry in outline:
                index = entry["page"] - 1
                check(f"outline_page_{index+1}_title", compact(entry["title"]) in compact(texts[index]))
            summary_text = compact(texts[0])
            for name in ("R2_minus_B5", "B5_minus_R1", "B5_minus_B2"):
                result = summary["comparisons"]["7"][name]
                expected = compact(name + "".join(fmt(v) for v in [*result["quarter_deltas"],
                                   result["mean_delta"], result["ci_lower"], result["ci_upper"]]))
                check("summary_7day_values:" + name, expected in summary_text, expected)
            selection_text = compact(texts[2])
            check("selected_structure_and_VALID",
                  compact(str(selected["width"]) + fmt(selected["weight_decay"], 10) +
                          fmt(selected["validation_qlike"], 10)) in selection_text)
            for quarter, page_index in (("2026Q2", 3), ("2026Q3", 4)):
                rows = {row["family"]: row for row in metrics if row["quarter"] == quarter}
                check(quarter + "_all13_source_rows", set(rows) == set(FAMILIES))
                text = compact(texts[page_index])
                for family in FAMILIES:
                    row = rows[family]
                    expected = compact(family + "".join(fmt(row[key]) for key in
                        ("AUROC", "AP", "raw_QLIKE", "QLIKE_Regret", "logRV_MSE")))
                    check(quarter + "_metrics:" + family, expected in text, expected)
                first = rows[FAMILIES[0]]
                check(quarter + "_sample_counts", all(token in text for token in
                      ("N=" + first["N"], "positive=" + first["positive"], "negative=" + first["negative"])))
            historical = []
            for index in range(min(len(old), max(0, len(document) - 14))):
                original, appendix = old[index], document[index + 14]
                text_equal = original.get_text("text") == appendix.get_text("text")
                pix1 = original.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False, colorspace=fitz.csRGB)
                pix2 = appendix.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False, colorspace=fitz.csRGB)
                pixel_equal = (pix1.width, pix1.height, pix1.n, pix1.samples) == (pix2.width, pix2.height, pix2.n, pix2.samples)
                check(f"historical_page_{index+1}_text_exact", text_equal)
                check(f"historical_page_{index+1}_RGB_1_3_exact", pixel_equal)
                historical.append({"original_page": index + 1, "appendix_page": index + 15,
                                   "text_exact": text_equal, "RGB_1_3_exact": pixel_equal,
                                   "original_pixels_sha256": hashlib.sha256(pix1.samples).hexdigest(),
                                   "appendix_pixels_sha256": hashlib.sha256(pix2.samples).hexdigest()})
            record["historical_pages"] = historical
            check("all25_historical_pages_compared", len(historical) == 25)
        record["status"] = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    except Exception as error:
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
    # Keep failed evidence too. This file deliberately cannot become visual QA.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "checks": len(checks),
                      "failed_checks": sum(item["status"] == "FAIL" for item in checks),
                      "visual_acceptance": False, "evidence": str(destination)}, ensure_ascii=False))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-generated-only", action="store_true", required=True)
    parser.parse_args()
    raise SystemExit(check_generated())
