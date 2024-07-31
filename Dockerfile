FROM --platform=linux/amd64 docker.io/library/python:3.10-slim

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt /opt/app/
# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user create_model.py /opt/app/

RUN  ["python", "create_model.py"]
ENTRYPOINT ["python", "inference.py"]