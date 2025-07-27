import fitz  # PyMuPDF: pip install PyMuPDF
import os
import argparse

def pdf_to_images(pdf_path: str, out_folder: str, img_format: str = 'png', dpi: int = 200):
    """
    Convert each page of a PDF into an image file.

    Args:
        pdf_path (str): Path to the input PDF file.
        out_folder (str): Directory to save output images.
        img_format (str): 'png' or 'jpg' (or 'jpeg').
        dpi (int): Resolution in dots per inch.
    """
    # Ensure output directory exists
    os.makedirs(out_folder, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)

    # Iterate over pages
    for page_number in range(len(doc)):
        page = doc[page_number]
        # Calculate transformation matrix for desired DPI
        zoom = dpi / 72  # 72 dpi is the default resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Construct output file path
        out_path = os.path.join(out_folder, f'page_{page_number + 1}.{img_format.lower()}')
        # Save image
        pix.save(out_path)
        print(f"Saved: {out_path}")

    doc.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF pages to images (PNG or JPG)")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--out_folder", default="output_images",
                        help="Output directory for images (default: output_images)")
    parser.add_argument("-f", "--format", choices=["png", "jpg", "jpeg"], default="png",
                        help="Image format (png, jpg, jpeg)")
    parser.add_argument("-d", "--dpi", type=int, default=200,
                        help="Resolution in DPI (default: 200)")
    args = parser.parse_args()

    pdf_to_images(args.pdf_path, args.out_folder, args.format, args.dpi)
