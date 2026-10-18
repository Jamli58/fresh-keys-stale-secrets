import hashlib
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "model_provenance", Path(__file__).resolve().parents[1] / "scripts" / "model_provenance.py"
)
mp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mp)


def _remote(path, data):
    return {"Path": path, "Sha256": hashlib.sha256(data).hexdigest(), "Size": len(data), "Revision": "abc"}


def test_sha256_of_file_streams_the_whole_file(tmp_path):
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"x" * 3_000_000)
    assert mp.sha256_of(f) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_compare_passes_when_every_weight_file_matches(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"aaa")
    (tmp_path / "config.json").write_bytes(b"{}")
    remote = [_remote("a.safetensors", b"aaa"), _remote("config.json", b"{}"), _remote("README.md", b"hi")]
    report = mp.compare(remote, tmp_path)
    assert report["ok"] and report["mismatched"] == [] and report["missing"] == []
    assert set(report["checked"]) == {"a.safetensors", "config.json"}  # README is not a checked file


def test_compare_flags_mismatch_and_missing_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"corrupt")
    remote = [_remote("a.safetensors", b"aaa"), _remote("b.safetensors", b"bbb")]
    report = mp.compare(remote, tmp_path)
    assert not report["ok"]
    assert report["mismatched"] == ["a.safetensors"] and report["missing"] == ["b.safetensors"]


def test_excluded_remote_paths_are_not_required(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"aaa")
    remote = [_remote("a.safetensors", b"aaa"), _remote("original/consolidated.00.pth", b"big")]
    assert mp.compare(remote, tmp_path)["ok"]
