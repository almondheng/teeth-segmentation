# Teeth-Segmentation

## Problem
Dental X-rays provide critical insights for diagnosis and treatment planning in dentistry. However, manual segmentation of teeth from X-ray images is labor-intensive, time-consuming, and prone to human error. Accurate and automated teeth segmentation is essential for tasks such as detecting dental anomalies, planning orthodontic treatments, and performing forensic analysis. This project aims to develop a robust machine learning model for teeth segmentation on dental X-ray images, addressing challenges like varying image quality, overlapping teeth structures, and occlusal orientation differences. The goal is to improve efficiency and precision in dental image analysis through automated segmentation.

## Dataset
1.Download dataset from [Teeth Segmentation on dental X-ray images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images/data).
2.Unzip and copy the Teeth Segmentation PNG folder into the repo.
3.Rename the folder as "data".

# Getting started

## Prerequisite
- Python 3.12.x
- Pipenv
- Docker

## Installation
```bash
pipenv install
pipenv shell
```

## Run Training
```bash
python train.py
```

## Start application
### Local
```bash
waitress-serve --host 127.0.0.1 predict:app
```

### Docker
```bash
docker build -t teeth-segmentation .
docker run -p 8080:8080 teeth-segmentation
```

## Run inference
```bash
curl -F "image=@examples/124.jpg" http://localhost:8080/predict \
```
