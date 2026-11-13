
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
