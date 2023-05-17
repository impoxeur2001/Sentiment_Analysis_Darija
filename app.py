# requirement
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch
from aaransia import transliterate
from flask import Flask, request, jsonify

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the targeted values
class_values = [0, 1]

# set the max lenght
MAX_LEN = 180

# initialize the encoder
DarijaBERT_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")


# define the sentiment classifier class
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = DarijaBert_model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT", return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)


# load the trained model
model = SentimentClassifier(len(class_values))
model.load_state_dict(torch.load('best_model_state.bin'))
model = model.to(device)


# the encoding function
def encode(phrase):
    phrase_ar = transliterate(phrase, source='ma', target='ar', universal=True)
    encoded_phrase = DarijaBERT_tokenizer.encode_plus(
        phrase_ar,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt', )
    return encoded_phrase


# the prediction function

def predict(phrase):
    encoded_phrase = encode(phrase)
    input_ids = encoded_phrase['input_ids'].to(device)
    attention_mask = encoded_phrase['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    if class_values[prediction] == 1:
        return 'positif'
    else:
        return 'négatif'


# creat flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def post_prediction():
    phrase = request.args.get('phrase')
    prediction = predict(phrase)
    return prediction

if __name__== "__main__" :
    app.run(debug=True)

