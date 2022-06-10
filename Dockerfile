FROM python:3.7.4
ENV SERVER_PORT 3000
ENV SERVER_HOST 0.0.0.0
EXPOSE 3000
WORKDIR /src/models
RUN pip install pandas==0.25.1 \
&& pip install scikit-learn==0.21.3 \
&& pip install matplotlib==3.5.2\
&& pip install protobuf==3.20.1 \
&& pip install mlflow==1.2.0 \
&& apt-get update \
&& apt-get install -y git
# Copy over artifact and code
COPY run.sh /src/models
RUN mkdir -p /src/models/src
RUN mkdir -p /src/models/src/models
COPY src/models/dataframe_test.csv /src/models/src/models
RUN file="$(ls -l /src/models/src/models)" && echo $file
RUN echo $(ls -l /src/models)
COPY src/models/predict_model.py /src/models
CMD ["python", "predict_model.py"]