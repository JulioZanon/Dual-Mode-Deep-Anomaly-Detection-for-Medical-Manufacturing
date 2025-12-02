FROM tensorflow/tensorflow:latest-gpu-jupyter
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install graphviz ffmpeg libsm6 libxext6 nano python3-tk libx11-dev curl -y
RUN python -m pip install -U pip
RUN curl https://getmic.ro | bash

RUN uv pip install scikit-image scikit-learn matplotlib \
opencv-python  mlflow rich tqdm loguru xonsh jupyter tensorflow[and-cuda] \
imgaug albumentations seaborn isort graphviz tensorflow-datasets nevergrad \
optuna hyperopt bayesian-optimization pycocotools ax-platform hpbandster \
ConfigSpace "ray[data,train,tune,serve]" gradio pydot tabulate GPUtil plotly \
pandas dash xgboost catboost hdbscan streamlit
