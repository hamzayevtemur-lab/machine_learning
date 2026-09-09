import tkinter as tk
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

# Load trained model
model = joblib.load("model/digit_model.pkl")
columns = joblib.load("model/columns.pkl")

canvas_size = 280

# Create blank image
image = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(image)

last_x, last_y = None, None

def paint(event):
    global last_x, last_y
    r = 10  # brush radius — slightly bigger for smoother strokes
    if last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, event.x, event.y,
                           fill="white", width=r * 2, capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, event.x, event.y], fill=255, width=r * 2)
    last_x, last_y = event.x, event.y

def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

def predict():
    img = np.array(image)

    # Find where the digit is (non-zero pixels)
    coords = np.column_stack(np.where(img > 0))

    if coords.size == 0:
        label.config(text="Draw something!")
        return

    # Bounding box with a small padding
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad = 10
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(img.shape[0], y_max + pad)
    x_max = min(img.shape[1], x_max + pad)

    # Crop digit
    img_cropped = img[y_min:y_max, x_min:x_max]

    # Resize to 20x20 using LANCZOS (high quality, matches MNIST style)
    img_resized = Image.fromarray(img_cropped).resize((20, 20), Image.LANCZOS) #type:ignore

    # Apply slight Gaussian blur to smooth strokes like MNIST
    img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=1))

    # Create 28x28 black image and paste centered
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img_resized, (4, 4))

    img_array = np.array(new_img, dtype=np.float32)

    # Normalize to [0, 1] — DO NOT invert!
    # MNIST is stored as white digit on black background (pixel=255 → feature=1.0)
    # Your canvas is also white-on-black, so no inversion needed.
    img_array = img_array / 255.0

    # Flatten to 1x784
    img_flat = img_array.reshape(1, -1)

    # Build DataFrame with correct column names
    img_df = pd.DataFrame(img_flat, columns=columns)

    # Predict
    prediction = model.predict(img_df)
    proba = model.predict_proba(img_df)
    confidence = np.max(proba)

    # Show top-3 candidates for transparency
    top3_idx = np.argsort(proba[0])[::-1][:3]
    top3 = [(model.classes_[i], proba[0][i]) for i in top3_idx]
    top3_str = "  |  ".join([f"{d}: {p:.2f}" for d, p in top3])

    label.config(text=f"Prediction: {prediction[0]}  ({confidence:.0%} confident)\n{top3_str}")

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
    label.config(text="Draw a digit (0–9)")

# GUI setup
root = tk.Tk()
root.title("Digit Recognizer")
root.resizable(False, False)

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black",
                   cursor="crosshair")
canvas.pack(pady=8)

canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=4)

btn_predict = tk.Button(btn_frame, text="Predict", command=predict,
                        width=10, font=("Arial", 12))
btn_predict.pack(side=tk.LEFT, padx=6)

btn_clear = tk.Button(btn_frame, text="Clear", command=clear,
                      width=10, font=("Arial", 12))
btn_clear.pack(side=tk.LEFT, padx=6)

label = tk.Label(root, text="Draw a digit (0–9)", font=("Arial", 14), justify=tk.CENTER)
label.pack(pady=6)

root.mainloop()