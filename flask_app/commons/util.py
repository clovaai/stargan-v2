import base64
from io import BytesIO

from PIL import Image


def base64_to_image(base64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes))
    return image


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    image_bytes = base64.b64encode(buffered.getvalue())
    return image_bytes.decode("utf-8")
