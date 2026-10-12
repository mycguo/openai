"""Bounded image decoding, orientation correction, and metadata-free model images."""

from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from media_rag.config import RagError
from .config import MAX_IMAGE_BYTES, MAX_PIXELS


register_heif_opener()
FORMATS = {"JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png"),
           "WEBP": (".webp", "image/webp"), "HEIF": (".heic", "image/heic")}


def jpeg_bytes(image, limit):
    image = image.copy()
    image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    # New images have no EXIF/GPS, embedded text, or other source metadata.
    while True:
        output = BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
        if output.tell() <= MAX_IMAGE_BYTES:
            return output.getvalue()
        image.thumbnail((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.LANCZOS)


def prepare_image(source, directory):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as opened:
                if opened.format not in FORMATS:
                    raise RagError("Use a JPEG, PNG, WebP, HEIC, or HEIF photo.")
                if opened.width * opened.height > MAX_PIXELS:
                    raise RagError("Photos are limited to 50 megapixels. Resize this image first.")
                if getattr(opened, "n_frames", 1) != 1:
                    raise RagError("Use a single still photo, rather than an animated or multi-frame image.")
                extension, mime_type = FORMATS[opened.format]
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                # Resize before allocating RGBA/RGB copies of large phone photos.
                oriented.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                rgba = oriented.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, "white")
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                image_data, thumbnail_data = jpeg_bytes(rgb, 2048), jpeg_bytes(rgb, 400)
    except RagError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise RagError("This photo could not be decoded safely. Try exporting it as JPEG or PNG.") from exc
    image_path, thumbnail_path = Path(directory) / "image.jpg", Path(directory) / "thumbnail.jpg"
    image_path.write_bytes(image_data)
    thumbnail_path.write_bytes(thumbnail_data)
    return {"extension": extension, "mime_type": mime_type, "width": width, "height": height,
            "image_path": image_path, "thumbnail_path": thumbnail_path}
