FROM python:3.12.2-slim

RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --deploy --system

COPY ["teeth_segmentation_model.pkl", "./"]
COPY [ "predict.py", "./" ]

EXPOSE 8080

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:8080", "predict:app" ]