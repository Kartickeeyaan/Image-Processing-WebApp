import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from utils.image_io import allowed_file, unique_name, save_image, read_bgr
from cvops.point_ops import Negative, Gamma, Log, Threshold
from cvops.neighborhood_ops import MeanFilter, MedianFilter, GaussianFilter, SobelFilter
from cvops.contrast_ops import ContrastStretch

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret"
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["RESULT_FOLDER"] = os.path.join("static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# Registry mapping (Category, Operation) -> Implementation
OP_REGISTRY = {
    ("Point Operations", "Negative transformation"): Negative(),
    ("Point Operations", "Gamma transformation"): Gamma(),
    ("Point Operations", "Log transformation"): Log(),
    ("Point Operations", "Threshold transformation"): Threshold(),
    ("Neighbourhood Operations", "Mean filter"): MeanFilter(),
    ("Neighbourhood Operations", "Median Filter"): MedianFilter(),
    ("Neighbourhood Operations", "Gaussian Filter"): GaussianFilter(),
    ("Neighbourhood Operations", "Sobel Filter"): SobelFilter(),
    ("Contrast Stretching", "Contrast Stretching"): ContrastStretch(),
}

# Categories dictionary used in index.html
CATEGORIES = {
    "Point Operations": [
        "Negative transformation",
        "Gamma transformation",
        "Log transformation",
        "Threshold transformation",
    ],
    "Neighbourhood Operations": [
        "Mean filter",
        "Median Filter",
        "Gaussian Filter",
        "Sobel Filter",
    ],
    "Contrast Stretching": [
        "Contrast Stretching",
    ],
}


@app.route("/", methods=["GET"])
def index():
    """Render the input page with categories & operations."""
    return render_template("index.html", categories=CATEGORIES)


@app.route("/process", methods=["POST"])
def process():
    """Handle file upload, apply selected operation, and render result."""
    if "image" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Allowed file types: jpg, jpeg, png")
        return redirect(url_for("index"))

    # Save uploaded file
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    in_name = unique_name(ext)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], in_name)
    file.save(save_path)

    # Get category + operation
    category = request.form.get("category")
    operation = request.form.get("operation")
    if not category or not operation:
        flash("Please select category and operation.")
        return redirect(url_for("index"))

    op_key = (category, operation)
    if op_key not in OP_REGISTRY:
        flash("Selected operation is not available.")
        return redirect(url_for("index"))

    op = OP_REGISTRY[op_key]

    # Read input image
    img = read_bgr(save_path)
    if img is None:
        flash("Failed to read uploaded image.")
        return redirect(url_for("index"))

    # Collect extra kwargs (dynamic parameters)
    kwargs = {}
    for key, value in request.form.items():
        if key in ("category", "operation"):
            continue
        if not value:
            continue
        try:
            kwargs[key] = int(value)
        except ValueError:
            try:
                kwargs[key] = float(value)
            except ValueError:
                kwargs[key] = value

    # Apply the selected operation
    try:
        out_img = op.apply(img, **kwargs)
    except Exception as e:
        flash(f"Error applying operation: {e}")
        return redirect(url_for("index"))

    # Save result
    out_name = unique_name("png")
    out_path = os.path.join(app.config["RESULT_FOLDER"], out_name)
    save_image(out_path, out_img)

    # URLs for templates
    in_url = url_for("static", filename=f"uploads/{in_name}")
    out_url = url_for("static", filename=f"results/{out_name}")

    return render_template(
        "result.html",
        input_image_url=in_url,
        output_image_url=out_url,
        category=category,
        operation=operation,
        params=kwargs,
    )


@app.route("/docs", methods=["GET"])
def docs():
    """Render the Docs page."""
    return render_template("docs.html")


@app.route("/about", methods=["GET"])
def about():
    """Render the About page."""
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
