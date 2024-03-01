from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

from logistic import LogisticRegression
from SVM import SVM
from LRclassificationMetrics import ClassificationMetrics
from standarize import standardize_data
from scale import scale


app = Flask(__name__, static_url_path="/static")


# Load the trained model from the .pkl file
with open("D:\BreastCancerPredictionFinal\LRmodel.pkl", "rb") as file:
    model_LR = LogisticRegression.load_model(
        "D:/BreastCancerPredictionFinal/LRmodel.pkl"
    )


with open("D:/BreastCancerPredictionFinal/SVM_model.pkl", "rb") as file:
    model_SVM = SVM.load_model("D:/BreastCancerPredictionFinal/SVM_model.pkl")

# print(model)


@app.route("/")
def index():
    return render_template("index.html")


def predict_logistic(features):
    final_features = [np.array(features)]
    final_features = standardize_data(final_features)
    prediction_proba = model_LR.predict_proba_lr(final_features)
    benign_prob = prediction_proba[0]
    output = model_LR.predict(final_features)

    if np.any(output == 1):
        return f"The patient is predicted as Malignant with a probability of {benign_prob:.2%}."
    else:
        return f"The patient is predicted as Benign with a probability of {1 - benign_prob:.2%}."


# SVM Prediction
def predict_svm(features):
    final_features = [np.array(features)]
    final_features = standardize_data(final_features)
    prediction_proba = model_SVM.predict_proba_svm(final_features)
    benign_prob = prediction_proba[0]
    output = model_SVM.predict(final_features)

    if np.any(output == 1):
        return f"The patient is predicted as Malignant with a probability of {benign_prob:.2%}."
    else:
        return f"The patient is predicted as Benign with a probability of {1 - benign_prob:.2%}."


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        selected_model = request.form["model"]
        features = [
            float(request.form[key]) for key in request.form.keys() if key != "model"
        ]
        print(request)
        print(features)
        prediction = ""

        if selected_model == "logistic":
            prediction = predict_logistic(features)
        elif selected_model == "svm":
            prediction = predict_svm(features)

        return render_template("predict.html", prediction=prediction)
    return render_template("predict.html")


@app.route("/data")
def data():
    return render_template("data.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


import matplotlib

matplotlib.use("agg")

from flask import Flask, render_template, request, flash, redirect
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a new model with the correct architecture
num_classes = 2  # Adjust this based on your specific model
model = models.resnet101(pretrained=False)
model.conv1 = torch.nn.Conv2d(
    3, 64, kernel_size=7, stride=2, padding=3, bias=False
)  # Adjust input channels
model.fc = torch.nn.Linear(2048, num_classes)

# Load the checkpoint weights into the new model
checkpoint = torch.load("final_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Define the class labels
class_labels = ["Benign", "Malignant"]

# Variables for storing accuracy data
epochs = []
train_accuracies = []
val_accuracies = []


def plot_probability_graph(class_labels, prob_values, prediction):
    plt.bar(
        class_labels,
        prob_values,
        color=["pink" if label == prediction else "gray" for label in class_labels],
    )
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Prediction Probability")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()


def save_probability_plot(class_labels, prob_values, prediction, image_path):
    plot_probability_graph(class_labels, prob_values, prediction)
    plt.savefig(image_path, format="png")
    plt.close()


@app.route("/image", methods=["GET", "POST"])
def image():
    prediction = ""
    image_path = ""
    img_base64 = ""
    prob_dict = {}
    prob_values = []  # Initialize prob_values outside the try block

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        uploaded_file = request.files["file"]

        # Check if the file is present and has an allowed extension
        if uploaded_file.filename == "" or not allowed_file(uploaded_file.filename):
            flash("Invalid file")
            return redirect(request.url)

        try:
            # Save the image to a temporary location
            image_path = os.path.join("static", "temp_image.jpg")
            uploaded_file.save(image_path)

            # Preprocess the image
            image = Image.open(image_path)
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move to GPU if available

            # Make the prediction
            predicted_label, probabilities = predict_image(input_tensor)

            # Get the predicted class label
            prediction = class_labels[predicted_label]

            # Plotting probability values
            prob_values = probabilities
            prob_dict = {label: prob_values[i] for i, label in enumerate(class_labels)}

            # Save the plot to a BytesIO object
            img_bytes = io.BytesIO()
            save_probability_plot(class_labels, prob_values, prediction, img_bytes)

            # Convert the BytesIO object to base64 for rendering in HTML
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        except Exception as e:
            flash(f"Error processing image: {e}")
            return redirect(request.url)

    return render_template(
        "image.html",
        prediction=prediction,
        image_path=image_path,
        img_base64=img_base64,
        prob_dict=prob_dict,
        prob_values=prob_values,
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "jpg",
        "jpeg",
        "png",
    }


def predict_image(image_tensor):
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_label = torch.max(output, 1)
    return predicted_label.item(), probabilities.numpy()


if __name__ == "__main__":
    app.run(debug=True)


# @app.route("/predict", methods=["POST", "GET"])
# def predict():
#     if request.method == "POST":
#         features = [float(x) for x in request.form.values()]
#         print(request)
#         print(features)
#         final_features = [np.array(features)]
#         final_features = standardize_data(final_features)

#         print(final_features)
#         prediction_proba = model.predict_proba(final_features)
#         benign_prob = prediction_proba[0]
#         print([benign_prob])

#         output = model.predict(final_features)
#         print(output)

#         if np.any(output == 1):
#             res_val = f"The patient is predicted as Malignant with a probability of {benign_prob:.2%}."
#         else:
#             res_val = f"The patient is predicted as Benign with a probability of {1 - benign_prob:.2%}."

#         return render_template("predict.html", prediction="{}".format(res_val))

#     elif request.method == "GET":
#         return render_template("predict.html")


# @app.route("/data")
# def data():
#     return render_template("data.html")


# @app.route("/faq")
# def faq():
#     return render_template("faq.html")


# @app.route("/contact")
# def contact():
#     return render_template("contact.html")


# if __name__ == "__main__":
#     app.run(debug=True)
