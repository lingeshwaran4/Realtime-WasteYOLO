import cv2
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="O5gSX0xWhnQ19NRtutlR"
)

# Change this index based on your DroidCam test (usually 1 or 2)
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Unable to open DroidCam video source.")
    exit()

print("✅ DroidCam connected successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame from camera.")
        break

    # Save the current frame temporarily
    cv2.imwrite("temp.jpg", frame)

    # Run Roboflow workflow
    result = client.run_workflow(
        workspace_name="lingeshwaran",
        workflow_id="find-cardboards-papers-and-plastics-3",
        images={"image": "temp.jpg"},
        use_cache=True
    )

    # Extract predictions safely
    predictions = result[0].get('predictions', {}).get('predictions', [])

    for det in predictions:
        x = int(det['x'] - det['width']/2)
        y = int(det['y'] - det['height']/2)
        w = int(det['width'])
        h = int(det['height'])
        cls = det['class']
        conf = det['confidence']

        if conf > 0.5:  # Only show confident detections
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("♻️ Live Waste Detection (DroidCam USB)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
