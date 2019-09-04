# Fruit Classifer  

- Build a reproducible fruit classification model using transfer learning   
- Dockerize the model for reproduciblility
- Unit test using pytest

## Solution

#### Prerequisites:

1. Install docker engine as shown [here](https://docs.docker.com/install/)

2. Pre-processed data used can be reproduce by executing the docker container in data_code:

	a. Checking your chrome version [here](https://www.whatismybrowser.com/detect/what-version-of-chrome-do-i-have) and download the correct chromedriver [here](http://chromedriver.chromium.org/downloads)
	b. Build docker image from Dockerfile as follows:

<pre><code>docker build -t preprocess_data .</code></pre>

	c. Creating and start the container with data volume mounted to extract the generated pickle files as follows:

<pre><code>docker run -v $(pwd):/data preprocess_data </code></pre>

#### Solution Details 	

1. Build docker image from Dockerfile as follows:

<pre><code>docker build -t solution .</code></pre>

2. Launch dockerize jupyter notebook as follows 

3. Train and evaluate fruit classification model in notebook and extract model.h5 file 

## Built With

Code tested with Docker version ? on Google Cloud Platform 

## Author

<p>Lin Laiyi, Senior AI Apprentice at AI Singapore, NUS MSBA 2017/2018</p>
<p>LinkedIn: https://www.linkedin.com/in/laiyilin/</p>
<p>Portfolio of selected analytics project: https://drive.google.com/file/d/1fVntFEvj6us_6ERzRmbU85EOeZymFxEm/view</p>