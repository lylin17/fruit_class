FROM python:3.6.9-slim

#Set working directory
RUN mkdir src
WORKDIR /src

#set volume mounting point
VOLUME /src

#pip install to setup python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter

#Copy required files
COPY . .

#Expose port 80
EXPOSE 80

#Run data pre_processing script
CMD [ "jupyter", "notebook", "--ip='*'", "--port=8000", "--no-browser", "--allow-root" ]
