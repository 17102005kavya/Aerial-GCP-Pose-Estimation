import requests
from PIL import Image, ImageDraw
import io

def test_prediction():
    url = "http://127.0.0.1:8000/predict"
    
    # 1. Create a synthetic image (e.g. 1000x1000 white background)
    img = Image.new("RGB", (1000, 1000), color="white")
    draw = ImageDraw.Draw(img)
    
    # Draw a black cross in the center (to resemble a GCP)
    # Center is at (500, 500)
    draw.line((500, 450, 500, 550), fill="black", width=10)
    draw.line((450, 500, 550, 500), fill="black", width=10)
    
    # Save image to bytes buffer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    # 2. Upload to FastAPI predict endpoint
    files = {"file": ("test_gcp.jpg", img_byte_arr, "image/jpeg")}
    
    print("Sending request to backend /predict endpoint...")
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print("\nPrediction successful!")
        print(f"File: {data['filename']}")
        print(f"Dimensions: {data['width']}x{data['height']}")
        print(f"Predicted Shape: {data['shape']}")
        print(f"Coordinates: X={data['x']}, Y={data['y']}")
        print(f"Confidence: {data['confidence']}")
        print(f"Inference Latency: {data['inference_time_ms']} ms")
        print("\nCascade Stages:")
        for stage in data['stages']:
            cw = stage['crop_window']
            print(f"  Stage {stage['stage']} ({stage['label']}): crop at ({cw['left']}, {cw['top']}), size {cw['width']}x{cw['height']}, predicted abs coord: ({stage['predicted_absolute']['x']}, {stage['predicted_absolute']['y']})")
    else:
        print(f"Prediction failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_prediction()
