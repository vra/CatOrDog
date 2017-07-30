import StringIO

from PIL import Image
from keras.models import load_model
import tensorflow as tf
import numpy as np
from django.shortcuts import render

graph = tf.get_default_graph()

def index(request):
	return render(request, 'index.html')

def demo(request):
	if request.method == 'POST' and request.FILES['img']:
		img = request.FILES['img']
		cls = classifer(img)
		print 'classify result: ', 'dog' if cls else 'cat'
		render_dict = {'done': 1, 'cls': cls}
		return render(request, 'demo.html', render_dict)
	else:
		return render(request, 'demo.html')


def classifer(img):
	model_path = 'models/vgg16_finetune.h5'
	
	img_str = ''
	for chunk in img.chunks():
		img_str += chunk
	
	# chunks to PIL Image, see here: https://stackoverflow.com/questions/24996518/what-size-to-specify-to-pil-image-frombytes
	img = Image.open(StringIO.StringIO(img_str))
	img = img.resize((150,150))
	img_np = np.expand_dims(np.asarray(img, dtype='float32'), axis=0)
	
	#NOTE: must set graph, see here: https://github.com/fchollet/keras/issues/2397
	global graph	
	with graph.as_default():
		model = load_model(model_path)
		res = model.predict(img_np, batch_size=1)
	if res[0,0] < 0.5:
		return 0
	else:
		return 1
