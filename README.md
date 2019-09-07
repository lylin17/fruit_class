# Fruit Classifer  

- Built a Flask app to classify fruit images using Docker
- Based on a simple CNN classification model trained using transfer learning and dockerize for reproduciblility
- We will be building 3 containers for data preprocessing, model training and deployment respectively. 
- For data consistency, model training used prepared processed data that are made available in dropbox.
- For each container, test suite for unit testing using pytest is available.

## Solution

#### Prerequisites:

1. Install docker engine as shown [here](https://docs.docker.com/install/)

2. Pre-processed data used can be reproduce by executing the docker container in *data_code folder*:

	a. Checking your chrome version [here](https://www.whatismybrowser.com/detect/what-version-of-chrome-do-i-have) and download the correct chromedriver [here](http://chromedriver.chromium.org/downloads)   
	b. Unzip and move chromedriver.exe to the data_code folder   
	c. Run the following blocks of python code to use a chrome browser to scrap images from google (google-images-download==2.5.0):

	<pre><code> 	
	response = google_images_download.googleimagesdownload()   

	arguments = {"keywords": "apples fruit,oranges fruit,pears fruit",
				 "limit": 400,
				 "metadata": False,
				 "chromedriver": "chromedriver.exe"}   

	paths = response.download(arguments) 
	</code></pre>	
	
	d. Build docker image from Dockerfile as follows:   

	<pre><code>docker build -t preprocess_data .</code></pre>  

	e. Create and start the container to run data preprocessing with data volume mounted (to access scapped images in downloads/ and to extract the generated pickle files) as follows:   
	
	<pre><code>docker run -v $(pwd):/data preprocess_data</code></pre>   

3. If required, unit testing of the data preprocessing code can be performed with pytest as follows:

	a. Enter container bash as follows(CONTAINER_ID can be found using docker ps -a):   
	
	<pre><code>docker exec -it <CONTAINER_ID> bash</code></pre>   

	b. Run pytest with the appriopriate flags(e.g. -v for verbose) as follows:    

	<pre><code>pytest -v</code></pre>   	

#### Solution Details 	

###### Part A: Train Model

1. In *model folder*, build docker image from Dockerfile as follows:

<pre><code>docker build -t model .</code></pre>

2. Create and start container with data volume mounted (to extract the generated model.h5 file) as follows:

<pre><code> docker run -v $(pwd):/src -p 8000:8000 model </code></pre>

3. Launch dockerized jupyter notebook by copying the URL given. For data consistency, we will used prepared data available on dropbox.

4. Train and evaluate fruit classification model in notebook and extract model.h5 file 

###### Part B: Deployment

1. In repository folder, build docker image from Dockerfile as follows:

<pre><code>docker build -t deploy .</code></pre>

2. Create and start container with data volume mounted (to read generated model.h5 file) as follows:

<pre><code> docker run -v $(pwd)/model/model.h5:/src/model.h5 -p 5000:5000 deploy </code></pre>

## Built With

Code tested with Docker Engine - Community Edition (version 19.03.1) on Google Cloud Platform.

1. To launch dockerized jupyter notebook in Google Cloud Platform, create SSH tunnel using:

<pre><code> gcloud compute ssh <instance_name> -- -L 8000:127.0.0.1:8000 </code></pre>

2. To access Flask app through Google Cloud Platform,   
	
	a. create SSH tunnel as before using:   
	
	<pre><code> gcloud compute ssh <instance_name> -- -L 5000:127.0.0.1:5000 </code></pre>   

	b. Set ip to 0.0.0.0 in app.py as follows:    

	<pre><code> app.run(host='0.0.0.0', port=5000) </code></pre>   

	c. Use the following address after running the app:    

	<pre><code> localhost:5000 </code></pre>   

## Author

<p>Lin Laiyi, Senior AI Apprentice at AI Singapore, NUS MSBA 2017/2018</p>
<p>LinkedIn: https://www.linkedin.com/in/laiyilin/</p>
<p>Portfolio of selected analytics project: https://drive.google.com/file/d/1fVntFEvj6us_6ERzRmbU85EOeZymFxEm/view</p>