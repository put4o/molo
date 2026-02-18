import os
import base64
import fitz  # PyMuPDF

def convert_pdf_pages_to_base64_images(pdf_path: str, page_numbers: list[int], image_dpi: int, extract_text: bool = False) -> tuple[list[str], list[str]]:
    """
    Opens a PDF, extracts specified pages, renders them as PNG images,
    and returns them as a list of base64 encoded strings.
    Optionally extracts text from the pages as well.

    Args:
        pdf_path (str): The full path to the PDF file.
        page_numbers (list[int]): A list of 0-indexed page numbers to extract.
        image_dpi (int): DPI for rendering PDF pages
        extract_text (bool): Whether to extract text from the pages as well.

    Returns:
        tuple[list[str], list[str]]: A tuple of (base64_images, page_texts)
                   Returns empty lists if errors occur or PDF not found.
    """
    base64_images = []
    page_texts = []
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return base64_images, page_texts

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return base64_images, page_texts

    for page_num in sorted(list(set(page_numbers))):  # Process unique pages in order
        if 0 <= page_num - 1 < doc.page_count:
            try:
                page = doc.load_page(page_num - 1)  # Page numbers are 0-indexed in PyMuPDF
                
                # Render page to a pixmap (raster image)
                pix = page.get_pixmap(dpi=image_dpi)
                img_bytes = pix.tobytes("png")  # Get image bytes in PNG format
                base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(base64_encoded_image)
                
                # Extract text if requested
                if extract_text:
                    text = page.get_text()
                    page_texts.append(text)
                else:
                    page_texts.append("")
                    
            except Exception as e:
                print(f"Error processing page {page_num} from '{pdf_path}': {e}")
        else:
            print(
                f"Warning: Page number {page_num} is out of range for '{pdf_path}' (Total pages: {doc.page_count}). Skipping this page.")

    doc.close()
    return base64_images, page_texts