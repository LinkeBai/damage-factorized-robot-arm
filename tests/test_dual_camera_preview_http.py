from __future__ import annotations

from io import BytesIO
import json
import time

import numpy as np
import pytest

from scripts import serve_dual_camera_preview as preview


def request_without_network(path: str):
    """Exercise the HTTP handler without opening a socket or any hardware."""
    handler = preview.Handler.__new__(preview.Handler)
    handler.path = path
    handler.wfile = BytesIO()
    result = {"status": None, "headers": {}}

    def send_response(status, _message=None):
        result["status"] = status

    def send_header(name, value):
        result["headers"][name] = value

    def send_error(status, _message=None, _explain=None):
        result["status"] = status

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = lambda: None
    handler.send_error = send_error
    handler.do_GET()
    result["body"] = handler.wfile.getvalue()
    return result


def test_query_string_and_index_alias_serve_uncached_resilient_page() -> None:
    for path in ("/?session=target-marking", "/index.html?v=3"):
        response = request_without_network(path)
        page = response["body"].decode("utf-8")

        assert response["status"] == 200
        assert "no-store" in response["headers"]["Cache-Control"]
        assert "AbortController" in page
        assert "cache:'no-store'" in page
        assert "setTimeout(refreshFrame,delayMs)" in page
        assert "Camera preview: reconnecting" in page
        assert "initializeControls" in page
        assert "e.value=String(settings[names[i]])" in page
        assert "value=3000" not in page


def test_snapshot_reports_sequence_and_rejects_stale_frame(monkeypatch) -> None:
    monkeypatch.setattr(preview, "FRAME", np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(preview, "FRAME_NUMBER", 42)
    monkeypatch.setattr(preview, "FRAME_UPDATED_MONOTONIC", time.monotonic())

    fresh = request_without_network("/snapshot.jpg?t=123")
    assert fresh["status"] == 200
    assert fresh["headers"]["X-Frame-Number"] == "42"
    assert fresh["headers"]["Content-Type"] == "image/jpeg"
    assert fresh["body"].startswith(b"\xff\xd8")

    monkeypatch.setattr(preview, "FRAME_UPDATED_MONOTONIC", time.monotonic() - 3.0)
    stale = request_without_network("/snapshot.jpg?t=124")
    assert stale["status"] == 503


FROZEN = {
    "daheng_exposure": 7400.0,
    "daheng_gain": 14.0,
    "second_exposure": -5.0,
    "second_brightness": -38.0,
}


def test_repository_frozen_settings_are_the_selected_values() -> None:
    assert preview.read_settings_file(preview.FROZEN_SETTINGS_PATH) == FROZEN


def test_startup_validates_frozen_baseline_and_prefers_preview_session(tmp_path) -> None:
    frozen_path = tmp_path / "camera_settings_selected.json"
    session_path = tmp_path / "camera_settings_preview_session.json"
    frozen_path.write_text(json.dumps(FROZEN), encoding="utf-8")

    loaded, source = preview.load_startup_settings(frozen_path, session_path)
    assert loaded == FROZEN
    assert source == "formal_frozen"

    adjusted = dict(FROZEN, second_brightness=-32.0)
    session_path.write_text(json.dumps(adjusted), encoding="utf-8")
    loaded, source = preview.load_startup_settings(frozen_path, session_path)
    assert loaded == adjusted
    assert source == "preview_session"

    session_path.write_text(json.dumps({"daheng_exposure": 7400.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        preview.load_startup_settings(frozen_path, session_path)


def test_http_update_persists_only_preview_session(monkeypatch, tmp_path) -> None:
    frozen_path = tmp_path / "camera_settings_selected.json"
    session_path = tmp_path / "camera_settings_preview_session.json"
    frozen_bytes = json.dumps(FROZEN).encode("utf-8")
    frozen_path.write_bytes(frozen_bytes)
    monkeypatch.setattr(preview, "FROZEN_SETTINGS_PATH", frozen_path)
    monkeypatch.setattr(preview, "PREVIEW_SESSION_SETTINGS_PATH", session_path)
    monkeypatch.setattr(preview, "PENDING", dict(FROZEN))

    response = request_without_network("/settings?second_brightness=-31")

    assert response["status"] == 200
    assert json.loads(response["body"])["second_brightness"] == -31.0
    assert json.loads(session_path.read_text(encoding="utf-8"))["second_brightness"] == -31.0
    assert frozen_path.read_bytes() == frozen_bytes
    with pytest.raises(ValueError, match="must not overwrite"):
        preview.persist_preview_settings(frozen_path)


def test_first_camera_apply_uses_one_validated_settings_snapshot() -> None:
    class FloatFeature:
        def __init__(self, name, applied):
            self.name = name
            self.applied = applied

        def set(self, value):
            self.applied[self.name] = value

    class Features:
        def __init__(self):
            self.applied = {}

        def get_float_feature(self, name):
            return FloatFeature(name, self.applied)

    class SecondCamera:
        def __init__(self):
            self.applied = {}

        def set(self, prop, value):
            self.applied[prop] = value

    features = Features()
    second = SecondCamera()
    preview.apply_camera_settings(features, second, FROZEN)

    assert features.applied == {"ExposureTime": 7400.0, "Gain": 14.0}
    assert second.applied[preview.cv2.CAP_PROP_EXPOSURE] == -5.0
    assert second.applied[preview.cv2.CAP_PROP_BRIGHTNESS] == -38.0
