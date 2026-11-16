
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


def test_c2v2_bottleneck_removes_detail_above_b_and_keeps_size():
    """C2 v2 (2026-11-15): 224 -> B -> 224 caps the detail; an image already band-limited to B changes much less."""
    import cv2
    import numpy as np

    from deepvision.data.clips import shrink_jpeg

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (224, 224, 3), np.uint8)  # detail up to Nyquist
    smooth = cv2.resize(cv2.resize(noise, (128, 128), interpolation=cv2.INTER_AREA), (224, 224),
                        interpolation=cv2.INTER_LINEAR)

    def lap(b):
        g = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        assert g.shape == (224, 224)
        return cv2.Laplacian(g, cv2.CV_32F).var()

    png = lambda a: cv2.imencode(".png", a)[1].tobytes()  # noqa: E731
    assert lap(shrink_jpeg(png(noise), "square", 224, 95, 128)) < 0.2 * lap(shrink_jpeg(png(noise), "square", 224, 95))
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)  # noqa: E731
    change = lambda a: np.abs(dec(shrink_jpeg(png(a), "square", 224, 95, 128))  # noqa: E731
                              - dec(shrink_jpeg(png(a), "square", 224, 95))).mean()
    assert change(smooth) < 0.25 * change(noise)  # measured 5.1 vs 35.7 gray levels
    assert shrink_jpeg(png(noise), "square", 224, 95, 0) == shrink_jpeg(png(noise), "square", 224, 95, 224)


def test_c2v3_reencode_keeps_frames_is_deterministic_and_crf_controls_loss():
    """C2 v3 (2026-11-16): one shared H.264 generation at the bottleneck scale. libx264 (ffmpeg 9.0.2 gyan build) is not
    bit-exact run to run even single-threaded (about 1 run in 15 differs, by <= 1.2 gray levels mean per frame vs ~9.6
    for the compression itself), so repeat runs are compared with a tolerance."""
    import cv2
    import numpy as np

    from deepvision.data.clips import reencode_frames

    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(0, 256, (56, 56, 3), np.uint8), (224, 224), interpolation=cv2.INTER_LINEAR)
    frames = [cv2.imencode(".jpg", np.roll(base, 2 * k, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
              for k in range(16)]
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).astype(np.float32)  # noqa: E731
    lo = reencode_frames(frames, 224, 192, 23, 95)
    assert len(lo) == 16 and all(dec(b).shape == (224, 224, 3) for b in lo)
    again = reencode_frames(frames, 224, 192, 23, 95)
    assert max(np.abs(dec(a) - dec(b)).mean() for a, b in zip(lo, again)) < 2.0
    err = lambda out: np.mean([np.abs(dec(a) - dec(b)).mean() for a, b in zip(frames, out)])  # noqa: E731
    assert err(lo) < err(reencode_frames(frames, 224, 192, 40, 95))
    assert len(reencode_frames(frames[:8], 224, 192, 23, 95)) == 8
