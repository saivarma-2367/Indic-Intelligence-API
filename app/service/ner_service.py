from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any

class NERService:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        results = self.pipeline(text)

        cleaned_results = []
        for item in results:
            cleaned_item = {
                "entity": item.get("entity_group"),
                "word": item.get("word"),
                "score": float(item.get("score")),
                "start": int(item.get("start")),
                "end": int(item.get("end"))
            }
            cleaned_results.append(cleaned_item)

        return cleaned_results