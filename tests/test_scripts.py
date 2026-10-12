import csv
import importlib.util

from conftest import SAMPLES, FakeExtractor
from ocrmod.config import REPO_ROOT, Settings


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_on_samples_reports_no_values(tmp_path):
    ev = load("evaluate")
    settings = Settings.load()
    settings.data["ingest"]["use_paddle_orientation"] = False
    with open(SAMPLES / "ground_truth.csv", newline="") as f:
        truth = {row.pop("file"): row for row in csv.DictReader(f)}
    docs = sorted(SAMPLES.glob("inv-*.pdf"))
    summary, records = ev.evaluate("pdftext", docs, truth, tmp_path, settings, "pdftext",
                                   extractor=FakeExtractor())
    assert summary["field_accuracy"] == 1.0
    assert set(records[0]) == {"engine", "field", "correct", "level", "score"}
    assert "Syarikat" not in str(summary) + str(records)


def test_evaluate_normalise():
    ev = load("evaluate")
    assert ev.normalise("total", "1,733.1") == ev.normalise("total", 1733.10) == "1733.10"
    assert ev.normalise("invoice_number", "ti/09/2026/0088") == "TI0920260088"
    assert ev.normalise("vendor_name", "Kedai Maju Sdn. Bhd.") == ev.normalise("vendor_name", "KEDAI MAJU")
    assert ev.normalise("po_number", "") is None


def test_calibrate_fit():
    cal = load("calibrate")
    recs = ([{"score": 0.99, "correct": True}] * 95 + [{"score": 0.95, "correct": True}] * 10
            + [{"score": 0.8, "correct": False}] * 3 + [{"score": 0.8, "correct": True}] * 7
            + [{"score": 0.3, "correct": False}] * 5)
    yellow, red, stats = cal.fit(recs, green_precision=0.99, red_accuracy=0.5)
    assert yellow == 0.95  # everything >= 0.95 is correct
    assert red == 0.35     # the 0.30 band is all wrong; the 0.80 band is 70% right
    assert stats["fields"] == 120
