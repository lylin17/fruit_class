FROM python:3.6.9-slim

#Set working directory
RUN mkdir data
WORKDIR /data

#set volume mounting point
VOLUME /data

#pip install to setup python dependencies
COPY requirements.txt /data/requirements.txt
RUN pip install -r requirements.txt

#Copy required files
COPY . .

#Run data pre_processing script
CMD [ "python", "data_preprocess.py" ]
