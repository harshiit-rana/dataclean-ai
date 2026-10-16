"""Uploads are bounded by their shape -- columns, rows, cell length -- not only their bytes."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.errors import FileShapeTooLargeError
from app.main import app
from app.services.encoding import read_csv_bytes

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[3] / "samples"


def upload(content: bytes):
    return client.post("/api/upload", files={"file": ("f.csv", io.BytesIO(content), "text/csv")})


@pytest.fixture
def tight(monkeypatch):
    monkeypatch.setenv("MAX_COLUMNS", "5")
    monkeypatch.setenv("MAX_ROWS", "10")
    monkeypatch.setenv("MAX_CELL_CHARS", "50")
    get_settings.cache_clear()


class TestEachLimit:
    def test_too_many_columns_is_refused_from_the_header(self, tight):
        content = (",".join(f"c{i}" for i in range(6)) + "\n" + ",".join("1" * 6) + "\n").encode()
        with pytest.raises(FileShapeTooLargeError) as caught:
            read_csv_bytes(content)
        assert "6 columns; the limit is 5" in caught.value.detail and "MAX_COLUMNS" in caught.value.detail

    def test_too_many_rows_is_refused(self, tight):
        content = ("a,b\n" + "1,2\n" * 11).encode()
        with pytest.raises(FileShapeTooLargeError) as caught:
            read_csv_bytes(content)
        assert "11 rows; the limit is 10" in caught.value.detail

    def test_an_enormous_cell_is_refused_with_where_it_is(self, tight):
        content = ("a,b\nx,1\n" + "y" * 51 + ",2\n").encode()
        with pytest.raises(FileShapeTooLargeError) as caught:
            read_csv_bytes(content)
        assert "column 'a' (row 2) is 51 characters long; the limit is 50" in caught.value.detail

    def test_a_file_at_every_limit_is_accepted(self, tight):
        content = ("a,b,c,d,e\n" + ("x" * 50 + ",1,2,3,4\n") * 10).encode()
        assert read_csv_bytes(content).shape == (10, 5)

    def test_the_upload_route_answers_413_with_the_reason(self, tight):
        response = upload(("a,b\n" + "1,2\n" * 11).encode())
        assert response.status_code == 413 and "rows" in response.json()["detail"]


class TestTheDefaults:
    def test_the_defaults_are_conservative_but_roomy(self):
        settings = get_settings()
        assert settings.max_columns == 1000 and settings.max_rows == 2_000_000 and settings.max_cell_chars == 32_767

    @pytest.mark.parametrize("sample", sorted(p.name for p in SAMPLES.glob("*.csv")))
    def test_every_shipped_sample_is_within_them(self, sample):
        assert len(read_csv_bytes((SAMPLES / sample).read_bytes())) > 0
