import json
import redis
import uuid
import tensorflow as tf

from sentiment_api import settings
from models.wrapper import SentimentAnalysisModel

## Connect to Redis
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

## Auxillary functions. TODO: move to another file
def load_model():
	""" Load pre-trained sentiment classifier """
	## TODO: read in from config
	weights_file = "trained_models/2018_06_15_10_weights.h5"
	params_file = "trained_models/2018_06_15_10_params.json"
	preprocessor_file = "trained_models/2018_06_15_10_preprocessor.pkl" 
	sentiment_model = SentimentAnalysisModel.load(weights_file, params_file, preprocessor_file)
	return sentiment_model


def run_model_server():
	""" Continuously poll for request objects in Redis queue """
	
	## Load trained model
	MODEL = load_model()

	## Continously poll for request objects
	while True:

		## Poll for requests	
		queue = db.lrange(settings.REDIS_REQUEST_QUEUE, 1, settings.REDIS_BATCH_SIZE)

		image_ids = []
		batch = []
		for q in queue:
			## Deserialize object
			q = json.loads(q.decode("utf-8"))

			if "text" in q:
				batch.append(q["text"])
				image_ids.append(q["id"])

		if len(image_ids) > 0: 
			sentiment_scores = MODEL.predict(batch)

			for (img_id, text, sent_score) in zip(image_ids, batch, sentiment_scores):
				## Create response object
				output = {"id": img_id, "text": text, "sentiment_score": sent_score}
				
				## Add to Redis queue
				db.set(img_id, json.dumps(output))

			## Remove requests from the queue
			## TODO: edit this to make it accurate 
			db.ltrim(settings.REDIS_REQUEST_QUEUE, 1, len(image_ids))



if __name__ == "__main__":
	run_model_server()





















































