"""
OCR Pipeline for SoloLLM.

Extracts text from images and scanned PDFs using EasyOCR or Tesseract.
Integrates with the document ingestion pipeline for image-heavy documents.
"""

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_ocr_engine = None
_ocr_backend = None


@dataclass
class OCRResult:
    """Result of OCR processing on an image or page."""
    text: str
    confidence: float = 0.0
    language: str = "en"
    page_number: int | None = None
    bounding_boxes: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _init_ocr():
    """Initialize the best available OCR engine."""
    global _ocr_engine, _ocr_backend

    if _ocr_engine is not None:
        return _ocr_engine, _ocr_backend

    # Try EasyOCR first (better accuracy, GPU support)
    try:
        import easyocr
        _ocr_engine = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ocr_backend = "easyocr"
        logger.info("OCR initialized: EasyOCR")
        return _ocr_engine, _ocr_backend
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"EasyOCR init failed: {e}")

    # Try Tesseract
    try:
        import pytesseract
        # Quick test
        pytesseract.get_tesseract_version()
        _ocr_engine = pytesseract
        _ocr_backend = "tesseract"
        logger.info("OCR initialized: Tesseract")
        return _ocr_engine, _ocr_backend
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Tesseract init failed: {e}")

    logger.warning(
        "No OCR engine available. Install easyocr or pytesseract. "
        "Run: pip install easyocr  OR  pip install pytesseract"
    )
    return None, None


class OCRPipeline:
    """
    OCR pipeline that extracts text from images and scanned PDFs.

    Supports:
    - EasyOCR (preferred, better accuracy)
    - Tesseract (fallback)
    """

    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en"]

    def ocr_image(self, image_path: str) -> OCRResult:
        """
        Extract text from a single image file.

        Supports: PNG, JPG, JPEG, TIFF, BMP, WebP
        """
        engine, backend = _init_ocr()

        if engine is None:
            return OCRResult(
                text="",
                errors=["No OCR engine available. Install easyocr or pytesseract."],
            )

        if not os.path.exists(image_path):
            return OCRResult(text="", errors=[f"Image not found: {image_path}"])

        try:
            if backend == "easyocr":
                return self._ocr_easyocr(engine, image_path)
            else:
                return self._ocr_tesseract(engine, image_path)
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return OCRResult(text="", errors=[f"OCR failed: {str(e)}"])

    def ocr_pdf_pages(self, pdf_path: str) -> list[OCRResult]:
        """
        OCR all pages of a scanned PDF.

        Converts each page to an image, then runs OCR.
        Requires PyMuPDF for PDF-to-image conversion.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return [OCRResult(
                text="",
                errors=["PyMuPDF required for PDF OCR. Run: pip install PyMuPDF"],
            )]

        results = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc, 1):
                # Render page to image at 300 DPI
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=mat)

                # Save temp image
                temp_path = str(Path(pdf_path).parent / f"_ocr_temp_p{page_num}.png")
                pix.save(temp_path)

                # OCR the image
                result = self.ocr_image(temp_path)
                result.page_number = page_num
                results.append(result)

                # Clean up temp file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            doc.close()
        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            results.append(OCRResult(text="", errors=[f"PDF OCR failed: {str(e)}"]))

        return results

    def _ocr_easyocr(self, reader, image_path: str) -> OCRResult:
        """Run OCR using EasyOCR."""
        raw_results = reader.readtext(image_path)

        text_parts = []
        bboxes = []
        total_conf = 0.0

        for bbox, text, conf in raw_results:
            text_parts.append(text)
            total_conf += conf
            bboxes.append({
                "text": text,
                "confidence": round(conf, 3),
                "bbox": bbox,
            })

        avg_conf = total_conf / max(len(raw_results), 1)

        return OCRResult(
            text="\n".join(text_parts),
            confidence=round(avg_conf, 3),
            bounding_boxes=bboxes,
        )

    def _ocr_tesseract(self, pytesseract, image_path: str) -> OCRResult:
        """Run OCR using Tesseract."""
        try:
            from PIL import Image
            img = Image.open(image_path)
        except ImportError:
            # Try without PIL
            text = pytesseract.image_to_string(image_path)
            return OCRResult(text=text.strip(), confidence=0.0)

        text = pytesseract.image_to_string(img)

        # Get detailed data for confidence
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if int(c) > 0]
            avg_conf = sum(confidences) / max(len(confidences), 1) / 100
        except Exception:
            avg_conf = 0.0

        return OCRResult(
            text=text.strip(),
            confidence=round(avg_conf, 3),
        )

    def is_available(self) -> bool:
        """Check if any OCR engine is available."""
        engine, _ = _init_ocr()
        return engine is not None

    def get_backend_info(self) -> dict:
        """Get info about the active OCR backend."""
        engine, backend = _init_ocr()
        return {
            "available": engine is not None,
            "backend": backend or "none",
            "languages": self.languages,
        }


# Singleton
ocr_pipeline = OCRPipeline()
