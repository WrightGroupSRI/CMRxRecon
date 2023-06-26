FROM pytorch/pytorch
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

LABEL maintainer="calder.sheagren@mail.utoronto.ca"
LABEL version="0.01"
LABEL description="DSL-PILLAR Reconstruction: CMRxRecon Sunnybrook Submission"

WORKDIR /workspace

VOLUME /input
VOLUME /output

COPY . /workspace

RUN pip install -r requirements.txt

CMD ["python", "main.py", "--input_dir=/input", "--predict_dir=/output"]
