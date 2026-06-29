from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).parent
SEARCH_WORD = "bsp"
OUTPUT_NAME = "zusammengefuegt.png"


def combine_images(folder):
    image_paths = sorted(folder.glob(f"*{SEARCH_WORD}*.png"))
    if not image_paths:
        return

    images = [Image.open(path).convert("RGB") for path in image_paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)

    result = Image.new("RGB", (width, height), "white")

    y = 0
    for image in images:
        x = (width - image.width) // 2
        result.paste(image, (x, y))
        y += image.height

    output_dir = BASE_DIR
    for part in folder.relative_to(BASE_DIR).parts:
        output_dir = output_dir / f"{part}_"

    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / OUTPUT_NAME)

    for image in images:
        image.close()


def main():
    for folder in sorted(BASE_DIR.rglob("*")):
        if folder.is_dir() and not any(part.endswith("_") for part in folder.relative_to(BASE_DIR).parts):
            combine_images(folder)


if __name__ == "__main__":
    main()
