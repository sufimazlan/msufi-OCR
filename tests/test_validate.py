from datetime import date

from ocrmod.config import Settings
from ocrmod.validate import Registry, VendorMaster, VendorMatch, run_checks

CFG = Settings.load()["validation"]
GOOD_VENDOR = VendorMatch("V0001", "X", 100.0)


def base(**kw):
    d = {"vendor_name": "X", "invoice_number": "INV-1", "invoice_date": "2026-09-14",
         "due_date": "2026-10-14", "subtotal": 100.0, "tax_rate": 8.0, "tax_amount": 8.0,
         "total": 108.0, "line_items": [{"quantity": 2, "unit_price": 50, "amount": 100}]}
    d.update(kw)
    return d


def failed(checks):
    return {c.name for c in checks if not c.passed}


def run(data, vendor=GOOD_VENDOR, dups=()):
    return run_checks(data, CFG, vendor, 88, list(dups), today=date(2026, 10, 12))


def test_clean_invoice_passes():
    assert failed(run(base())) == set()


def test_sum_and_total_mismatch():
    names = failed(run(base(subtotal=110.0, tax_amount=8.8, total=118.8)))
    assert "sum mismatch" in names
    assert "total mismatch" not in names
    assert "total mismatch" in failed(run(base(total=109.0)))


def test_tax_maths_and_unusual_rate():
    assert "tax maths" in failed(run(base(tax_amount=6.0, total=106.0)))
    assert "unusual tax rate" in failed(run(base(tax_rate=None, tax_amount=7.0, total=107.0)))


def test_dates():
    assert "date in future" in failed(run(base(invoice_date="2027-01-01", due_date="2027-02-01")))
    assert "invalid date" in failed(run(base(invoice_date="2026-02-30")))
    assert "due before invoice date" in failed(run(base(due_date="2026-09-01")))


def test_missing_and_duplicate():
    names = failed(run(base(invoice_number=None, total=None), dups=["other.pdf"]))
    assert {"missing invoice_number", "missing total", "duplicate invoice"} <= names


def test_vendor_master_fuzzy_and_tax_id():
    vm = VendorMaster([{"vendor_code": "V1", "vendor_name": "Syarikat Contoh Bekalan Sdn. Bhd.", "tax_id": "W10-1"},
                       {"vendor_code": "V2", "vendor_name": "Kedai Alat Tulis Maju", "tax_id": ""}])
    assert vm.match("SYARIKAT CONTOH BEKALAN SDN BHD").code == "V1"
    assert vm.match("Kedai Alat Tulis Maju Enterprise").ratio >= 88
    assert vm.match("anything", tax_id="w10 1").code == "V1"
    assert vm.match("Perniagaan Baharu Cahaya").ratio < 88


def test_registry_ignores_rerun_of_same_file(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    reg.register("V1", "INV1", "sha-a", "a.pdf")
    assert reg.duplicates("V1", "INV1", "sha-a") == []
    assert reg.duplicates("V1", "INV1", "sha-b") == ["a.pdf"]
    reg.close()


def test_apply_gpu_limits_sets_env_but_respects_existing(monkeypatch):
    from ocrmod.config import apply_gpu_limits

    monkeypatch.delenv("FLAGS_gpu_memory_limit_mb", raising=False)
    s = Settings.load()
    apply_gpu_limits(s)
    import os
    assert os.environ["FLAGS_gpu_memory_limit_mb"] == str(s["gpu"]["paddle_memory_limit_mb"])
    monkeypatch.setenv("FLAGS_gpu_memory_limit_mb", "3000")
    apply_gpu_limits(s)
    assert os.environ["FLAGS_gpu_memory_limit_mb"] == "3000"
