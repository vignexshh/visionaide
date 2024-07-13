# Installation Guide For Visionade

### Step 1: Clone the Repository

1. Open your terminal (or Command Prompt on Windows).
2. Navigate to the directory where you want to clone the repository.
3. Use the `git clone` command followed by the URL of your GitHub repository. For example:

```sh
git clone https://github.com/vignexshh/visionaide.git
```

### Step 2: Navigate to the Project Directory

Once the repository is cloned, navigate into the project directory:

```sh
cd visionaide 
```

### Step 3: Create a Virtual Environment (Optional but Recommended)

It's a good practice to create a virtual environment to manage your project dependencies. Here are the commands to create and activate a virtual environment:

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

### Step 4: Install the Necessary Packages

Install the required packages using `pip`. Ensure you have a `requirements.txt` file in your project directory. If not, you can create it and list all the required packages:

Create `requirements.txt` (file already exists):
```txt
gradio
numpy
opencv-python
mediapipe
gtts
pytesseract
```

Install the packages:
```sh
pip install -r requirements.txt
```

### Step 5: Download the Model File

Ensure you have the `efficientdet.tflite` model file in the same directory as your `app.py`. If not, download it and place it in the directory.

### Step 6: Execute the Project

Run the `app.py` script to start the Gradio interface:

```sh
python app.py
```

This will launch the Gradio app, and you should see the interface in your web browser.

### Conclusion

Your Gradio interface should now be up and running in your local-host address. Star this repo if you found usefull & follow me!

### Project Submission (Vignesh T D, Suraj NS, Md Sadiq Ali (cobros))
