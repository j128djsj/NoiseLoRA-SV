from utils.verification import scan_runtime_files


def test_scanner_does_not_report_its_own_patterns():
    hits = [hit for hit in scan_runtime_files(".") if hit["file"].endswith("utils/verification.py")]
    assert hits == []


def test_scanner_detects_windows_absolute_path(tmp_path):
    path = tmp_path / "bad.py"
    bad_path = "D:" + "\\private\\file.wav"
    path.write_text(f"DATA = {bad_path!r}\n", encoding="utf-8")
    hits = scan_runtime_files(tmp_path)
    assert any(hit["kind"] == "windows_absolute_path" for hit in hits)


def test_scanner_detects_public_home_path(tmp_path):
    path = tmp_path / "bad.py"
    bad_path = "/public" + "/ho" + "me/user/data.wav"
    path.write_text(f"DATA = {bad_path!r}\n", encoding="utf-8")
    hits = scan_runtime_files(tmp_path)
    assert any(hit["kind"] == "public_home_path" for hit in hits)


def test_scanner_allows_generic_checkpoint_placeholder(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("Use /path/to/checkpoint.pth for examples.\n", encoding="utf-8")
    assert scan_runtime_files(tmp_path) == []
