FROM python:3.7.4
ENV SERVER_PORT 3000
ENV SERVER_HOST 0.0.0.0
EXPOSE 3000
WORKDIR /mlflow.dir
RUN pip install pandas==0.25.1 \
&& pip install scikit-learn==0.21.3 \
&& pip install protobuf==3.20.1 \
&& pip install mlflow==1.2.0 \
&& apt-get update \
&& apt-get install -y git
# Copy over artifact and code
COPY run.sh /mlflow.dir
COPY src/models/predict_model.py /mlflow.dir
CMD ["python", "/mlflow.dir/predict_model.py"]