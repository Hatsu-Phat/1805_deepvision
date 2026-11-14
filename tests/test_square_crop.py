
def test_square_crop_hides_aspect_ratio():
    import cv2
    import numpy as np

    from deepvision.data.clips import crop_center_square, crop_for, shrink_jpeg

    wide = np.zeros((256, 456, 3), np.uint8)
    wide[:, 100:356] = 200  # centered 256 x 256 block
    assert crop_center_square(wide).shape == (256, 256, 3)
    assert crop_center_square(wide).min() == 200
    assert crop_for({"crop": "square"}, "vif_test", "x", 1) == "square"
    ok, buf = cv2.imencode(".png", wide)
    out = cv2.imdecode(np.frombuffer(shrink_jpeg(buf.tobytes(), "square", 224, 95), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (224, 224, 3) and out.mean() > 190


def test_c2_prescale_then_square_crop_gives_one_downscale():
    """C2 (2026-11-14): short side -> 224 (INTER_AREA) at decode, then the square crop is already 224 x 224."""
    import cv2
    import numpy as np

    from deepvision.data.clips import shrink_jpeg
    from deepvision.data.extract import _resize_short_side

    for h, w in [(1080, 1920), (320, 568), (256, 456), (224, 224)]:
        img = _resize_short_side(np.full((h, w, 3), 128, np.uint8), 224)
        assert min(img.shape[:2]) == 224
        ok, buf = cv2.imencode(".jpg", img)
        out = cv2.imdecode(np.frombuffer(shrink_jpeg(buf.tobytes(), "square", 224, 95), np.uint8), cv2.IMREAD_COLOR)
        assert out.shape == (224, 224, 3)
