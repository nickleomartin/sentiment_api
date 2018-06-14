# sentiment_api
Implements a Bidirectional LSTM that utilises word (and character embeddings later) to classify sentiment as positive or negative. Exposes a trained model via a REST API using Django. 

Getting Setup
-------------
Clone the repo:
```
git clone git@github.com:NickLeoMartin/sentiment_api.git
cd sentiment_api/
```

Head into the repo and create and activate a virtual environment:
```
virtualenv --no-site-packages -p python3 venv
source venv/bin/activate
```

Install the packages used:
```
pip install -r requirements.txt
```

Running The Repo Locally:
-------------------------
To see the demo run Django lightweight development sever which is available on http://127.0.0.1:8000/ :
```
./manage.py runserver 
```

The demo was trained on single-sentence book reviews, avaliable [here](https://www.kaggle.com/c/si650winter11/data). 

![alt text](https://raw.githubusercontent.com/NickLeoMartin/sentiment_api/master/sentiment_demo.png)

To-Do
-----
V1:
- [x] Simple Boostrap interface
- [x] Basic Class-based API endpoint
- [x] Ajax call to API endpoint
- [x] BaseModel class
- [x] Bidirectional LSTM with word
- [x] Vocabulary builder
- [x] DocIdTransformer
- [x] Model training
- [x] Model storage
- [x] Model wrapper
- [x] Model prediction 
- [x] Local loading and prediction through API endpoint
- [x] Downloading of weights from git-lfs
- [x] Run demo from scratch
- [ ] Input validation, error handling, documentation etc.
- [ ] Dockerize project
- [ ] Deploy demo on server

V2:
- [ ] Hyperparameter tuning, training summary statistic, more comprehensive dataset 
- [ ] Asynchronous prediction with Celery
- [ ] Downloading script for short-text sentiment data
- [ ] Character embeddings
- [ ] Multi-task learning for entities and sentiment
