## Object Detection and Text Recognition Using Gradio and MediaPipe

### Abstract

This project demonstrates an application that performs object detection and text recognition on uploaded images using the Gradio interface. The application integrates several technologies and libraries including Gradio for the user interface, MediaPipe for object detection, OpenCV for image processing, PyTesseract for text recognition, and gTTS for text-to-speech conversion. The project is developed by Vignesh T D, Suraj N S, and Md Sadiq Ali, under the guidance of Dr. Vinod Kumar.

### Introduction

The objective of this project is to create an easy-to-use web interface for detecting objects and recognizing text in images. The application allows users to upload an image, processes the image to detect objects and recognize text, and provides an option to convert the recognized text to speech.

### Technologies and Libraries Used

1. **Gradio**: Gradio is an open-source Python library that allows you to quickly create user-friendly web interfaces for machine learning models. In this project, Gradio is used to create an interface where users can upload images and receive the processed results.

2. **MediaPipe**: MediaPipe is a framework for building multimodal (e.g., video, audio, etc.) machine learning pipelines. The MediaPipe Object Detection module is used in this project to detect objects within the uploaded images.

3. **OpenCV**: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It is used in this project for image processing tasks, such as reading and displaying images.

4. **PyTesseract**: PyTesseract is an## Project Paper: Object Detection and Text Recognition Using Gradio and MediaPipe

### Abstract

This project demonstrates an application that performs object detection and text recognition on uploaded images using the Gradio interface. The application integrates several technologies and libraries including Gradio for the user interface, MediaPipe for object detection, OpenCV for image processing, PyTesseract for text recognition, and gTTS for text-to-speech conversion. The project is developed by Vignesh T D, Suraj N S, and Md Sadiq Ali, under the guidance of Dr. Vinod Kumar.

### Introduction

The objective of this project is to create an easy-to-use web interface for detecting objects and recognizing text in images. The application allows users to upload an image, processes the image to detect objects and recognize text, and provides an option to convert the recognized text to speech.

### Technologies and Libraries Used

1. **Gradio**: Gradio is an open-source Python library that allows you to quickly create user-friendly web interfaces for machine learning models. In this project, Gradio is used to create an interface where users can upload images and receive the processed results.

2. **MediaPipe**: MediaPipe is a framework for building multimodal (e.g., video, audio, etc.) machine learning pipelines. The MediaPipe Object Detection module is used in this project to detect objects within the uploaded images.

3. **OpenCV**: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It is used in this project for image processing tasks, such as reading and displaying images.

4. **PyTesseract**: PyTesseract is an OCR (Optical Character Recognition) tool for Python. It is a wrapper for Google’s Tesseract-OCR Engine. In this project, PyTesseract is used to extract text from the uploaded images.

5. **gTTS (Google Text-to-Speech)**: gTTS is a Python library and CLI tool to interface with Google Translate’s text-to-speech API. It is used in this project to convert recognized text into speech.

### Project Setup and Execution

#### Cloning the Repository

To set up the project, clone the repository from GitHub:

```sh
git clone https://github.com/your-username/object_detection_project.git
cd object_detection_project
```

#### Creating a Virtual Environment

Create and activate a virtual environment (optional but recommended):

For Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

For macOS and Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

#### Installing Dependencies

Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

Ensure you have the `efficientdet.tflite` model file in the same directory as your `app.py`.

#### Running the Project

Execute the `app.py` script to start the Gradio interface:

```sh
python app.py
```

### Detailed Explanation of the Code

#### Imports and Constants

```python
import gradio as gr
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gtts import gTTS
import os
import pytesseract

MARGIN = 10  
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)
```

The above code imports all the necessary libraries and defines constants used for drawing bounding boxes and text on images.

#### Visualization Function

```python
def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
        
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image
```

The `visualize` function draws bounding boxes and labels on the detected objects in the image.

#### Object Detection Function

```python
def detect_objects(image_path):
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    detection_texts = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        detection_texts.append(f"{category_name} ({probability})")
    
    return rgb_annotated_image, "\n".join(detection_texts)
```

The `detect_objects` function uses MediaPipe to detect objects in the image and then visualizes the results.

#### Text Detection Function

```python
def detect_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    if text.strip() == "":
        return "No text element found"
    else:
        return text
```

The `detect_text` function uses PyTes## Project Paper: Object Detection and Text Recognition Using Gradio and MediaPipe

### Abstract

This project demonstrates an application that performs object detection and text recognition on uploaded images using the Gradio interface. The application integrates several technologies and libraries including Gradio for the user interface, MediaPipe for object detection, OpenCV for image processing, PyTesseract for text recognition, and gTTS for text-to-speech conversion. The project is developed by Vignesh T D, Suraj N S, and Md Sadiq Ali, under the guidance of Dr. Vinod Kumar.

### Introduction

The objective of this project is to create an easy-to-use web interface for detecting objects and recognizing text in images. The application allows users to upload an image, processes the image to detect objects and recognize text, and provides an option to convert the recognized text to speech.

### Technologies and Libraries Used

1. **Gradio**: Gradio is an open-source Python library that allows you to quickly create user-friendly web interfaces for machine learning models. In this project, Gradio is used to create an interface where users can upload images and receive the processed results.

2. **MediaPipe**: MediaPipe is a framework for building multimodal (e.g., video, audio, etc.) machine learning pipelines. The MediaPipe Object Detection module is used in this project to detect objects within the uploaded images.

3. **OpenCV**: OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It is used in this project for image processing tasks, such as reading and displaying images.

4. **PyTesseract**: PyTesseract is an OCR (Optical Character Recognition) tool for Python. It is a wrapper for Google’s Tesseract-OCR Engine. In this project, PyTesseract is used to extract text from the uploaded images.

5. **gTTS (Google Text-to-Speech)**: gTTS is a Python library and CLI tool to interface with Google Translate’s text-to-speech API. It is used in this project to convert recognized text into speech.

### Project Setup and Execution

#### Cloning the Repository

To set up the project, clone the repository from GitHub:

```sh
git clone https://github.com/your-username/object_detection_project.git
cd object_detection_project
```

#### Creating a Virtual Environment

Create and activate a virtual environment (optional but recommended):

For Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

For macOS and Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

#### Installing Dependencies

Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

Ensure you have the `efficientdet.tflite` model file in the same directory as your `app.py`.

#### Running the Project

Execute the `app.py` script to start the Gradio interface:

```sh
python app.py
```

### Detailed Explanation of the Code

#### Imports and Constants

```python
import gradio as gr
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gtts import gTTS
import os
import pytesseract

MARGIN = 10  
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)
```

The above code imports all the necessary libraries and defines constants used for drawing bounding boxes and text on images.

#### Visualization Function

```python
def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
        
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image
```

The `visualize` function draws bounding boxes and labels on the detected objects in the image.

#### Object Detection Function

```python
def detect_objects(image_path):
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    detection_texts = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        detection_texts.append(f"{category_name} ({probability})")
    
    return rgb_annotated_image, "\n".join(detection_texts)
```

The `detect_objects` function uses MediaPipe to detect objects in the image and then visualizes the results.

#### Text Detection Function

```python
def detect_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    if text.strip() == "":
        return "No text element found"
    else:
        return text
```

The `detect_text` function uses PyTesseract to extract text from the image.

#### Text-to-Speech Function

```python
def text_to_speech(text):
    tts = gTTS(text)
    tts.save("output.mp3")
    os.system("mpg123 output.mp3")
```

The `text_to_speech` function uses gTTS to convert text to speech and plays the audio.

#### Main Function

```python
def main(image):
    annotated_image, detection_texts = detect_objects(image)
    detected_text = detect_text(image)
    return annotated_image, detection_texts, detected_text
```

The `main` function integrates object detection and text detection.

#### Speak Function

```python
def speak(text):
    text_to_speech(text)
    return "Speaking now..."
```

The `speak` function converts the detection results text to speech.

#### Reset Function

```python
def reset():
    return None, None, None
```

The `reset` function clears the output.

### Gradio Interface

```python
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Upload Image")
            output_image = gr.Image(type="numpy", label="Annotated Image")
            output_text = gr.Textbox(label="Detection Results")
            output_text_detected = gr.Textbox(label="Detected Text")
        
        with gr.Column():
            speak_button = gr.Button("Speak Now")
            reset_button = gr.Button("New Image")
            
            speak_button.click(speak, inputs=output_text, outputs=gr.Textbox())
            reset_button.click(reset, outputs=[output_image, output_text, output_text_detected])
            
    img_input.change(main, inputs=img_input, outputs=[output_image, output_text, output_text_detected])

app.launch()
```

The above code defines the Gradio interface with image upload, display of annotated image, detection results, and text recognition. It also includes buttons for text-to-speech and resetting the interface.

### Conclusion

This project demonstrates a comprehensive approach to object detection and text recognition using several powerful Python libraries and frameworks. The Gradio interface provides an intuitive way for users to interact with the application, making it accessible and easy to use. The integration of MediaPipe, OpenCV, PyTesseract, and gTTS showcases the potential of combining different technologies to build a robust application.

### Credits

- **Developed by**: Vignesh T D, Suraj N S, Md Sadiq Ali
- **Faculty Guide**: Dr. Vinod Kumarseract to extract text from the image.

#### Text-to-Speech Function

```python
def text_to_speech(text):
    tts = gTTS(text)
    tts.save("output.mp3")
    os.system("mpg123 output.mp3")
```

The `text_to_speech` function uses gTTS to convert text to speech and plays the audio.

#### Main Function

```python
def main(image):
    annotated_image, detection_texts = detect_objects(image)
    detected_text = detect_text(image)
    return annotated_image, detection_texts, detected_text
```

The `main` function integrates object detection and text detection.

#### Speak Function

```python
def speak(text):
    text_to_speech(text)
    return "Speaking now..."
```

The `speak` function converts the detection results text to speech.

#### Reset Function

```python
def reset():
    return None, None, None
```

The `reset` function clears the output.

### Gradio Interface

```python
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Upload Image")
            output_image = gr.Image(type="numpy", label="Annotated Image")
            output_text = gr.Textbox(label="Detection Results")
            output_text_detected = gr.Textbox(label="Detected Text")
        
        with gr.Column():
            speak_button = gr.Button("Speak Now")
            reset_button = gr.Button("New Image")
            
            speak_button.click(speak, inputs=output_text, outputs=gr.Textbox())
            reset_button.click(reset, outputs=[output_image, output_text, output_text_detected])
            
    img_input.change(main, inputs=img_input, outputs=[output_image, output_text, output_text_detected])

app.launch()
```

The above code defines the Gradio interface with image upload, display of annotated image, detection results, and text recognition. It also includes buttons for text-to-speech and resetting the interface.

### Conclusion

This project demonstrates a comprehensive approach to object detection and text recognition using several powerful Python libraries and frameworks. The Gradio interface provides an intuitive way for users to interact with the application, making it accessible and easy to use. The integration of MediaPipe, OpenCV, PyTesseract, and gTTS showcases the potential of combining different technologies to build a robust application.

### Credits

- **Developed by**: Vignesh T D, Suraj N S, Md Sadiq Ali
- **Faculty Guide**: Dr. Vinod Kumar OCR (Optical Character Recognition) tool for Python. It is a wrapper for Google’s Tesseract-OCR Engine. In this project, PyTesseract is used to extract text from the uploaded images.

5. **gTTS (Google Text-to-Speech)**: gTTS is a Python library and CLI tool to interface with Google Translate’s text-to-speech API. It is used in this project to convert recognized text into speech.

### Project Setup and Execution

#### Cloning the Repository

To set up the project, clone the repository from GitHub:

```sh
git clone https://github.com/your-username/object_detection_project.git
cd object_detection_project
```

#### Creating a Virtual Environment

Create and activate a virtual environment (optional but recommended):

For Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

For macOS and Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

#### Installing Dependencies

Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

Ensure you have the `efficientdet.tflite` model file in the same directory as your `app.py`.

#### Running the Project

Execute the `app.py` script to start the Gradio interface:

```sh
python app.py
```

### Detailed Explanation of the Code

#### Imports and Constants

```python
import gradio as gr
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gtts import gTTS
import os
import pytesseract

MARGIN = 10  
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)
```

The above code imports all the necessary libraries and defines constants used for drawing bounding boxes and text on images.

#### Visualization Function

```python
def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
        
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image
```

The `visualize` function draws bounding boxes and labels on the detected objects in the image.

#### Object Detection Function

```python
def detect_objects(image_path):
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)
    
    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    detection_texts = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        detection_texts.append(f"{category_name} ({probability})")
    
    return rgb_annotated_image, "\n".join(detection_texts)
```

The `detect_objects` function uses MediaPipe to detect objects in the image and then visualizes the results.

#### Text Detection Function

```python
def detect_text(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    if text.strip() == "":
        return "No text element found"
    else:
        return text
```

The `detect_text` function uses PyTesseract to extract text from the image.

#### Text-to-Speech Function

```python
def text_to_speech(text):
    tts = gTTS(text)
    tts.save("output.mp3")
    os.system("mpg123 output.mp3")
```

The `text_to_speech` function uses gTTS to convert text to speech and plays the audio.

#### Main Function

```python
def main(image):
    annotated_image, detection_texts = detect_objects(image)
    detected_text = detect_text(image)
    return annotated_image, detection_texts, detected_text
```

The `main` function integrates object detection and text detection.

#### Speak Function

```python
def speak(text):
    text_to_speech(text)
    return "Speaking now..."
```

The `speak` function converts the detection results text to speech.

#### Reset Function

```python
def reset():
    return None, None, None
```

The `reset` function clears the output.

### Gradio Interface

```python
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Upload Image")
            output_image = gr.Image(type="numpy", label="Annotated Image")
            output_text = gr.Textbox(label="Detection Results")
            output_text_detected = gr.Textbox(label="Detected Text")
        
        with gr.Column():
            speak_button = gr.Button("Speak Now")
            reset_button = gr.Button("New Image")
            
            speak_button.click(speak, inputs=output_text, outputs=gr.Textbox())
            reset_button.click(reset, outputs=[output_image, output_text, output_text_detected])
            
    img_input.change(main, inputs=img_input, outputs=[output_image, output_text, output_text_detected])

app.launch()
```

The above code defines the Gradio interface with image upload, display of annotated image, detection results, and text recognition. It also includes buttons for text-to-speech and resetting the interface.

### Conclusion

This project demonstrates a comprehensive approach to object detection and text recognition using several powerful Python libraries and frameworks. The Gradio interface provides an intuitive way for users to interact with the application, making it accessible and easy to use. The integration of MediaPipe, OpenCV, PyTesseract, and gTTS showcases the potential of combining different technologies to build a robust application.

### Credits

- **Developed by**: Vignesh T D, Suraj N S, Md Sadiq Ali
- **Faculty Guide**: Dr. Vinod Kumar
