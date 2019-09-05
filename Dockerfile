FROM python:3.6.9-slim

#Set working directory
RUN mkdir src
WORKDIR /src

#set volume mounting point
VOLUME /src

#pip install to setup python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

#Copy required files
COPY . .

#Expose port 5000
EXPOSE 5000

#Run data pre_processing script
CMD [ "python", "app.py" ]
