import re

_reader = None


def read_plate(img):
    global _reader
    import easyocr, cv2
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 2, h * 2))
    best, score = "NOT READ", 0
    for _, text, conf in _reader.readtext(img):
        t = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(t) >= 4 and conf > score:
            best, score = t, conf
    return best
