from orion.vision.stream_ids import is_url_like, safe_camera_name, strip_userinfo


def test_strip_userinfo_removes_password() -> None:
    assert strip_userinfo("rtsp://admin:Pandas!@192.168.1.21:554/Preview_01_sub") == "rtsp://192.168.1.21:554/Preview_01_sub"
    assert strip_userinfo("rtsp://host/x") == "rtsp://host/x"


def test_strip_userinfo_password_with_at_sign() -> None:
    out = strip_userinfo("rtsp://u:p@ss@10.0.0.2/x")
    assert "p@ss" not in out and out.endswith("10.0.0.2/x")


def test_safe_camera_name_prefers_plain_name() -> None:
    assert safe_camera_name("rtsp://u:p@h/x", "cam0") == "cam0"
    assert safe_camera_name("carbon-webcam", "carbon") == "carbon-webcam"
    assert safe_camera_name(None, "", default="unknown") == "unknown"
    only_url = safe_camera_name("rtsp://u:p@h/x")
    assert only_url == "rtsp://h/x" and "p@" not in only_url


def test_is_url_like() -> None:
    assert is_url_like("rtsp://h") and not is_url_like("walkway") and not is_url_like(None)
