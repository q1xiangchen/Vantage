import os

from PIL import Image, ImageDraw, ImageFont


def add_label_to_image(image: Image.Image, label: str) -> Image.Image:
    font_size = max(20, int(image.height / 27))
    font = ImageFont.truetype(
        os.path.join(os.path.dirname(__file__), 'OpenSans-Bold.ttf'),
        font_size
    )

    padding = int(font_size * 0.4)

    try:
        text_bbox = font.getbbox(label)
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        _, text_height = font.getsize(label)

    title_height = text_height + padding * 2

    new_width = image.width
    new_height = image.height + title_height
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    draw = ImageDraw.Draw(new_image)
    draw.rectangle([(0, 0), (new_width, title_height)], fill=(220, 220, 220))

    text_position = (padding, padding)
    draw.text(text_position, label, fill=(0, 0, 0), font=font)
    new_image.paste(image, (0, title_height))

    return new_image
