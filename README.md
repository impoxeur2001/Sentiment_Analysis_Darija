# Sentiment_Analysis_Darija
This project implements a sentiment analysis model for the Darija dialect using the BERT model. The model was trained on a training dataset and evaluated on a testing dataset, achieving an accuracy of 92%. The trained model was then deployed as a REST API using Flask.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.6 or higher
- pip package manager
- virtualenv (optional)

### Installation

1. Clone the repository:

```
git clone https://github.com/impoxeur2001/sentiment_analysis_darija.git
cd sentiment-analysis-darija
```


3. Install the required packages:

```
pip install -r requirements.txt
```

### Usage

1. Start the Flask server:

```
python app.py
```

2. Make a POST request to the `/predict` endpoint with a JSON payload containing the text to analyze:

```python
import requests

url = 'http://localhost:5000/predict?phrase='
data = 'film mzian hada 3jboni chakhsiat'
response = requests.post(url+data, json=data)

print(response.json())
```

The response will be a JSON object containing the predicted sentiment :

```json
{
  "sentiment": "positive"
}
```

## Training and Evaluation

The sentiment analysis model was trained using the BERT model and tokenizer from the `transformers` library. The training dataset consisted of 1900 labeled examples, and the testing dataset consisted of 150 labeled examples.

The training and evaluation were done in a python notebook that you can find in this repository

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Malainine Mohamed Limame
- Drhorhi omar
- Meryem Boukdimi
- Bnou Mohamed
