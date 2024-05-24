from fastapi import File, UploadFile, Request, FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from torchvision import transforms
from PIL import Image
from torchvision import transforms
import mlflow.pytorch


# Set the MLflow tracking URI to the local server: 
# The MLflow server is useful for managing models when using URIs in the runs:/<mlflow_run_id>/path/to/model format.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Choose the model you want to run for production
MODEL_PATH = "./mlruns/0/04f225cb815e4cfa9401a822e57485d3/artifacts/model"

# Initialize the FastAPI app
app = FastAPI()

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Function to perform inference on an image using the specified model
def infer_image(model_path, image_path):
 
    # Load the model from the specified path (mlflow load)
    model_vgg = mlflow.pytorch.load_model(model_path)
    model_vgg.eval() # Set the model to evaluation mode
    
    # Define the class names
    class_names = ['LeafBlast', 'Healthy', 'Hispa', 'BrownSpot']

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)

    # Move input tensor to GPU if available
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Perform the prediction
    prediction = model_vgg(img_tensor)
    prediction = prediction.cpu().data.numpy().argmax()
    
    # Print and return the detected class
    print('Detected: {}'.format(class_names[prediction]))
    return class_names[prediction]


# Endpoint to upload an image
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = file.file.read()
        file_location = "./static/uploads/"+ file.filename

        # Save the file to the specified location (static)
        with open("./static/uploads/"+ file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        # Raise an HTTP exception if there's an error during upload
        raise HTTPException(status_code=500, detail="Could not upload image")

    finally:
        # Close the file
        file.file.close()

    # Print and return the file location and upload message
    print(file_location)
    return {"image" : file.filename , "message" : "image Uploaded" }




# Endpoint to classify an uploaded image
@app.post("/classify")
async def classify(request: Request):
    # Parse the request body to get the image file name
    body = await request.json()
    print('body', body)
    file_location = "./static/uploads/"+  body["image"]
    print('aaaa', file_location)
    image_class =  'Duck'

    # Perform image classification
    image_class = infer_image(MODEL_PATH, file_location)

    # Return the classification result
    return {"message" : "image classified successfully", "imageClass" : image_class }

# Endpoint to render the main page
@app.get("/")
def main(request: Request):
    # Render the index.html template
    context = {"request": request}
    return templates.TemplateResponse("index.html", context)