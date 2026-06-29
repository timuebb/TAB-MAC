from pathlib import Path

from PIL import Image


folder = Path(__file__).parent


def stack(folder_name):
    images = [Image.open(path).convert("RGB") for path in sorted((folder / folder_name).glob("*.png"))]

    result = Image.new(
        "RGB",
        (max(image.width for image in images), sum(image.height for image in images)),
        "white",
    )

    y = 0
    for image in images:
        result.paste(image, (0, y))
        y += image.height

    return result


raw = stack("raw")
log1p = stack("log1p")

result = Image.new("RGB", (raw.width + log1p.width, max(raw.height, log1p.height)), "white")
result.paste(raw, (0, 0))
result.paste(log1p, (raw.width, 0))
result.save(folder / "raw_log1p_gesamt.png")
