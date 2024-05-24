# Rice Leaf Classification Project

## Overview

This project aims to classify images of rice leaves into different categories, such as Leaf Blast, Healthy, Hispa, and Brown Spot. It consists of two main components: model development and an API for image classification.

## Model Development

The `model_dev` directory contains scripts for model development:

### Installation and Setup

### Prerequisites

- It is advisable to have a virtual machine running such as digital ocean droplet with at lesat RAM 8GB.


- Ensure your dataset is placed in the specified folder.
 > **Note**: The foldername `LabelledRice` has been changed to `DataLabelledRice`.
<!-- - Create an `mlruns` folder for MLflow to store its runs. -->

### Steps

1. **Clone the repository (from the any virtual environment e.g. digital ocean droplet)**:

    ```sh
    git clone https://github.com/atopatalos/mlflow-project.git
    ```

2. **Navigate to the cloned directorythen to 'model/dev'** :

    ```sh
    cd RiceLeafClassification/model_dev
    ```

3. **Build your Docker image**:

    ```sh
    docker build -t model_development .
    ```

4. **Modify `start.sh`**:

    Ensure the `start.sh` file has the following content, replacing `{your_local_path}` with the actual path to your directories on your local machine:


    ```sh
    #!/bin/bash
    docker run -it -p 5000:5000 \
    -v {your_local_path}/RiceLeafClassification/mlruns:/model_dev/mlruns \
    -v {your_local_path}/DataLabelledRice:/app/DataLabelledRice \
    model_development
    ```
    your `mlruns` directory should be located in `RiceLeafClassification` because the saved models are used for the api.
    For example, if your local dataset is at `/home/user/Documents/DataLabelledRice`, your `start.sh` should look like this:

    ```sh
    #!/bin/bash
    docker run -it -p 5000:5000 \
    -v /home/user/RiceLeafClassification/mlruns:/model_dev/mlruns \
    -v /home/user/Documents/DataLabelledRice:/app/DataLabelledRice \
    model_development
    ```

5. **Give execution permission to `start.sh`**:

    ```sh
    chmod +x start.sh
    ```

7. **Run `start.sh`**:

    ```sh
    ./start.sh
    ```

> **Note**: Your `mlruns` folder can become very large over time. If you have experiments you don't need, you can delete them from the folder. Alternatively, you can choose not to link an external volume for your MLflow runs, but be aware that your runs will not be saved. You can also modify your script to include a flag for whether or not to use MLflow.

### Re-running Training

Your Docker container will keep running after training, allowing you to see your experiment logs on MLflow. You can re-run your training with:

```sh
python src/train.py
```

This setup ensures your Docker environment is properly configured to run your training script and log experiments with MLflow, while also providing the flexibility to manage your MLflow runs as needed.

> **Note**: It's important to regularly clean up any unused Docker containers or images to free up disk space and maintain system performance.


## API Development

1. **Navigate to `RiceLeafClassification/app`**:

    ```sh
    cd RiceLeafClassification/app
    ```


The `api` directory hosts the API for image classification:

- `main.py`: FastAPI server that serves as the API for image classification.
- `templates/`: HTML templates for the web interface.
- `static/`: Static files for the web interface.


2. **Modify `start.sh`**:

    Ensure the `start.sh` file has the following content, replacing `{your_local_path}` with the actual path to your directories on your local machine:


    ```sh
    #!/bin/bash
    docker run -it -p 80:80 \
    -v {your_local_path}/RiceLeafClassification/mlruns:/app/mlruns \
    app
    ```
    - Your mlruns directory should be located in the RiceLeafClassification folder, as this is where the saved models required by the API are stored.
    - your `start.sh` should look something like this:

    ```sh
    #!/bin/bash
    docker run -it -p 80:80 \
    -v /home/user/Documents/RiceLeafClassification/mlruns:/app/mlruns \
    app
    ```

3. **Modify your MODEL_PATH in `main.py`**:

    ```
    MODEL_PATH = "./mlruns/0/{RUN_ID}/artifacts/model"
    ```

    If your ID for instance is `f743517c95254d0e93d2a32187d690a7` then your path should look something like this:

    ```
    MODEL_PATH = "./mlruns/0/f743517c95254d0e93d2a32187d690a7/artifacts/model"
    ```

    Your ID is the name of the directory in `mlruns/0/{IDs}` ou can also get the ID of your best experiment from the MLflow UI.


3. **Build your Docker image**:

    ```sh
    docker build -t app .
    ```

4. **Give execution permission to `start.sh`**:

    ```sh
    chmod +x start.sh
    ```

5. **Run `start.sh`**:

    ```sh
    ./start.sh
    ```

6. **Go to `http://0.0.0.0:80`**:

    You can then select an image from your drive to classify. Simply upload the image and proceed to classification.