# litr_app/Dockerfile

FROM python:3.11-buster as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && apt-get install -y git

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
    
COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8080

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.11-slim-buster as runtime

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

COPY ./litr_app ./litr_app

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

CMD ["streamlit", "run", "--server.port", "8080", "--server.enableCORS", "false" "litr_app/sum_eval.py"]