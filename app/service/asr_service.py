from transformers import pipeline

class ASRService:
    def __init__(self, model_name: str):
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_name
        )

    def transcribe(self, file_path: str) -> str:
        result = self.asr(file_path)
        return result["text"]