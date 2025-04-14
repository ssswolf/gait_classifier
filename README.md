# Gait Classification using Machine Learning

This project focuses on classifying human gait activities using sensor data. It supports loading and preprocessing data, extraxting useful features, and training and evaluation using Deep Neural Networks (DNN), Support Vector Machines (SVM), and Random Forest (RF).

## Project Structure

root.
│   main.py
│   README.md
│   requirements.txt
│
├───assets
│       dnn_model_20250411-111756.h5
│       normalizer_20250411-111730.pkl
│
├───data
│       data_loader.py
│   
├───dataset
│   ├───processed
│   │       dataset_ft.csv
│   │       dataset_mag.npy
│   │       dataset_ts.npy
│   │       labels.npy
│   │
│   └───raw
│       ├───Brush_teeth
│       ├───Climb_stairs
│       ├───Comb_hair
│       ├───Descend_stairs
│       ├───Drink_glass
│       ├───Eat_meat
│       ├───Eat_soup
│       ├───Getup_bed
│       ├───Liedown_bed
│       ├───Pour_water
│       ├───Sitdown_chair
│       ├───Standup_chair
│       ├───Use_telephone
│       └───Walk_mini
│
├───models
│       model_trainer.py
│
├───training
│       training_session.py
│
└───utils
        gait.py


## Models Supported

- `dnn` – Deep Neural Network (TensorFlow/Keras)
- `svm` – Support Vector Machine (Scikit-learn)
- `rf` – Random Forest Classifier (Scikit-learn)

## Activities Classified

The default target activities are based on the current dataset, but can be replaced with any activities present in your own dataset:

- `Climb_stairs`
- `Descend_stairs`
- `Walk_mini`

## Installation

1. Create and activate a conda environment:

```bash
conda create -n delve_env python=3.10
conda activate delve_env
```

2. Clone the repository or manually place the project folder:
    - Make sure your project folder contains files like main.py, data/, models/, training/, etc.
    - Open a terminal in the project root directory

```bash
cd <project-folder>
```

3. Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Run the training pipeline with default settings:

```bash
python main.py
```

2. Run the training pipeline with your selected model and activities:

```bash
python main.py --model dnn --activities Use_telephone,Standup_chair,Sitdown_chair,Eat_meat,Eat_soup --data_path dataset/raw/
```