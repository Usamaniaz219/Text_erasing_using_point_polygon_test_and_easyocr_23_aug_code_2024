

# Text Erasure from Zoning Map Images using Bounding Box Masks

This project is designed to **retain text pixel information** within specific regions of a zoning map and erase unnecessary text based on intersection with bounding box masks. It uses OpenCV, EasyOCR, and geometric operations for accurate processing of zoning data.

---

## 📂 Project Structure

The script performs the following key tasks:

1. **Detects text** from a map image using EasyOCR bounding boxes.
2. **Retains only contours** that intersect with text zones.
3. **Erases unwanted text pixels** outside zoning areas.
4. **Saves the processed masks** for downstream use.

---

## 🔧 Requirements

* Python 3.7+
* OpenCV
* NumPy
* EasyOCR (and `EasyOcr_Bounding_box_into_mask.py`)
* Shapely

Install dependencies via:

```bash
pip install numpy opencv-python shapely easyocr
```

---

## 🚀 How It Works

### 🔹 Step-by-Step Flow:

1. **Input Directory**: Contains original map images.
2. **Bounding Box Directory**: Contains masks generated from bounding boxes.
3. **Processing**:

   * Extracts bounding boxes using `EasyOcr_Bounding_box_into_mask.main()`.
   * Retains contours within bounding boxes using bitwise operations.
   * Identifies intersecting text regions within zoning areas.
4. **Output Directory**:

   * `*_output_mask.jpg`: Final combined zoning + retained text mask.
   * `*_text_mask.jpg`: Text mask retained inside zoning regions.
   * `*_text_pixel_mask.jpg`: Raw text pixel mask from OCR-based bounding boxes.

---

## 🖼️ Example Output

Each image processed produces:

* `image_output_mask.jpg`
* `image_text_mask.jpg`
* `image_text_pixel_mask.jpg`

Organized inside subdirectories named after the source image name.

---

## 🧠 Key Functions

| Function                                 | Description                                                                              |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| `retain_contours()`                      | Fills polygons from OCR bounding boxes and intersects them with adaptive threshold mask. |
| `text_erasing_using_pointpolygon_text()` | Uses `cv2.pointPolygonTest()` to retain only inner text pixels within zones.             |
| `process_image()`                        | Handles single image and mask pair processing.                                           |
| `process_images()`                       | Handles directory-wise bulk image processing.                                            |

---

## ⚙️ Usage

Update the paths in the `__main__` section before running:

```python
if __name__ == "__main__":
    tile_width = 1024
    tile_height = 1024

    input__image_directory = '/path/to/input/images/'
    mask_image_dir = '/path/to/bounding_box_masks/'
    output_directory = '/path/to/output/'

    process_images(input__image_directory, output_directory, mask_image_dir)
```

Run the script:

```bash
python text_erasing_from_mask_image_23_updated_aug_2024.py
```

---

## 📝 Notes

* Bounding boxes are expected in `[x1, y1], [x2, y2], [x3, y3], [x4, y4]` format.
* Intersection is checked using pixel-level containment for improved precision.
* Handles overlapping regions gracefully using `cv2.bitwise_or`.








