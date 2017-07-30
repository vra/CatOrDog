# CatOrDog
use deep learning to classify the animal in a picture is a cat or a dog

# How to deploy
## 1. clone this repo 
```bash
git clone https://github.com/vra/CatOrDog.git
```

## 2. Download models
```bash
cd CatOrDog
mkdir models
cd models
wget https://github.com/vra/CatOrDog/releases/download/v1.0/vgg16_finetune.h5
```

## 3. Create medias directory
```bash
mkdir -p static/medias
```

## 4. Run the django project
```bash
python manage.py runserver 0.0.0.0:8000
```
