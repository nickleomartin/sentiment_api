from django.shortcuts import render
from django.views.generic.base import View 
from django.http.response import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import tensorflow as tf

from models.wrapper import SentimentAnalysisModel


## Auxillary function
def load_model():
	## TODO: read in from config
	weights_file = "trained_models/2018_06_13_23_weights"
	params_file = "trained_models/2018_06_13_23_params"
	preprocessor_file = "trained_models/2018_06_13_23_preprocessor" 
	sentiment_model = SentimentAnalysisModel.load(weights_file, params_file, preprocessor_file)
	graph = tf.get_default_graph()
	return sentiment_model, graph


MODEL, graph = load_model()

class PredictSentimentView(View):
	""" API endpoint to predict sentiment of text """

	def get(self, request, *args, **kwargs):
		""" GET requests returns a warning message"""
		return JsonResponse({"response":"Use the post method","status": 400})

	def post(self, request, *args, **kwargs):
		""" POST requests take in text and return sentiment"""
		try:
			json_object = json.loads(request.body.decode("utf-8"))
			text = json_object['input']

			## Inference on loaded model
			global graph
			with graph.as_default():
				sent_class = MODEL.predict([text])

			## TODO: Asynchronous Celery call to process model prediction? Not needed now. 

			return JsonResponse({"text": text, "sentiment_score": sent_class, "response": "Successful", "status": 200})
		except:
			return JsonResponse({"text": text, "sentiment_score": "Error", "response": "Successful", "status": 400})

	@method_decorator(csrf_exempt)
	def dispatch(self, request, *args, **kwargs):
		""" Handle Cross-Site Request Forgery """
		return View.dispatch(self, request, *args, **kwargs)
