import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BertPredictor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict_batch(self, texts, max_length=128):
        predictions = []
        probabilities = []

        for text in texts:
            inputs = self.tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            pred = int(torch.argmax(probs).item())
            phishing_prob = float(probs[1].item())

            predictions.append(pred)
            probabilities.append(phishing_prob)

        return predictions, probabilities