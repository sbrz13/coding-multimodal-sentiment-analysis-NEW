import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.training.models import CrossAttentionModel, TextOnlyModel, ImageOnlyModel
from src.config.config import Config
from src.data_processing.preprocessor import TextPreprocessor, ImagePreprocessor


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
REVERSE_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


class MultimodalPredictor:
    def __init__(self, config_path, model_dir, device=None):
        self.config = Config(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor(image_size=224)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_config["text_model"])
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config.model_config["vision_model"])
        
        self.models = {}
        self.meta_learner = None
        self._load_models()
        
    def _load_models(self):
        model_files = {
            "cross_attention": "v5_large_cross_attn_best.pt",
            "text_only": "v5_large_text_only_best.pt",
            "image_only": "v5_large_image_only_best.pt",
        }
        
        for model_name, model_file in model_files.items():
            model_path = os.path.join(self.model_dir, model_file)
            if os.path.exists(model_path):
                print(f"Loading {model_name} from {model_path}")
                model = self._create_model(model_name)
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(self.device)
                model.eval()
                self.models[model_name] = model
                print(f"  ✓ {model_name} loaded successfully")
            else:
                print(f"  ✗ {model_name} not found at {model_path}")
        
        meta_learner_path = os.path.join(self.model_dir, "meta_learner.pkl")
        if os.path.exists(meta_learner_path):
            import joblib
            self.meta_learner = joblib.load(meta_learner_path)
            print(f"  ✓ Meta-learner loaded successfully")
    
    def _create_model(self, model_name):
        if model_name == "cross_attention":
            return CrossAttentionModel(self.config)
        elif model_name == "text_only":
            return TextOnlyModel(self.config)
        elif model_name == "image_only":
            return ImageOnlyModel(self.config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def preprocess_text(self, text):
        processed_text = self.text_preprocessor.preprocess(text)
        inputs = self.tokenizer(
            processed_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device)
        }
    
    def preprocess_image(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        image = self.image_preprocessor.preprocess(image)
        inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        )
        return inputs["pixel_values"].to(self.device)
    
    def predict_single_model(self, text=None, image_path=None, model_name="cross_attention"):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        if model_name == "cross_attention":
            if text is None or image_path is None:
                raise ValueError("Cross-Attention model requires both text and image")
            
            text_inputs = self.preprocess_text(text)
            image_inputs = self.preprocess_image(image_path)
            
            with torch.no_grad():
                logits, text_feat, image_feat, z_head = model(
                    text_inputs["input_ids"],
                    text_inputs["attention_mask"],
                    image_inputs
                )
                probs = F.softmax(logits, dim=-1)
        
        elif model_name == "text_only":
            if text is None:
                raise ValueError("Text-Only model requires text input")
            
            text_inputs = self.preprocess_text(text)
            
            with torch.no_grad():
                logits, text_feat = model(
                    text_inputs["input_ids"],
                    text_inputs["attention_mask"]
                )
                probs = F.softmax(logits, dim=-1)
        
        elif model_name == "image_only":
            if image_path is None:
                raise ValueError("Image-Only model requires image input")
            
            image_inputs = self.preprocess_image(image_path)
            
            with torch.no_grad():
                logits, image_feat = model(image_inputs)
                probs = F.softmax(logits, dim=-1)
        
        probs = probs.cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        predicted_label = LABEL_MAP[predicted_class]
        confidence = probs[predicted_class]
        
        return {
            "predicted_label": predicted_label,
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "probabilities": {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2])
            }
        }
    
    def predict_ensemble(self, text=None, image_path=None):
        if not self.meta_learner:
            raise ValueError("Meta-learner not loaded. Train meta-learner first.")
        
        all_probs = []
        
        for model_name in ["cross_attention", "text_only", "image_only"]:
            if model_name in self.models:
                try:
                    result = self.predict_single_model(text, image_path, model_name)
                    probs = [
                        result["probabilities"]["negative"],
                        result["probabilities"]["neutral"],
                        result["probabilities"]["positive"]
                    ]
                    all_probs.extend(probs)
                except Exception as e:
                    print(f"Warning: {model_name} prediction failed: {e}")
                    all_probs.extend([0.33, 0.33, 0.34])
        
        if len(all_probs) != 9:
            raise ValueError(f"Expected 9 probabilities, got {len(all_probs)}")
        
        ensemble_probs = self.meta_learner.predict_proba([all_probs])[0]
        predicted_class = np.argmax(ensemble_probs)
        predicted_label = LABEL_MAP[predicted_class]
        confidence = ensemble_probs[predicted_class]
        
        return {
            "predicted_label": predicted_label,
            "predicted_class": int(predicted_class),
            "confidence": float(confidence),
            "probabilities": {
                "negative": float(ensemble_probs[0]),
                "neutral": float(ensemble_probs[1]),
                "positive": float(ensemble_probs[2])
            }
        }
    
    def predict(self, text=None, image_path=None, use_ensemble=True):
        if use_ensemble and self.meta_learner:
            return self.predict_ensemble(text, image_path)
        else:
            available_models = list(self.models.keys())
            if "cross_attention" in available_models:
                return self.predict_single_model(text, image_path, "cross_attention")
            elif "text_only" in available_models:
                return self.predict_single_model(text=text, model_name="text_only")
            elif "image_only" in available_models:
                return self.predict_single_model(image_path=image_path, model_name="image_only")
            else:
                raise ValueError("No models available for prediction")


def predict_from_json(config_path, model_dir, input_json, output_json=None, use_ensemble=True):
    predictor = MultimodalPredictor(config_path, model_dir)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    results = []
    for i, sample in enumerate(data):
        text = sample.get("text", "")
        image_path = sample.get("image_path", "")
        
        try:
            result = predictor.predict(text=text, image_path=image_path, use_ensemble=use_ensemble)
            result["sample_id"] = sample.get("id", i)
            if "label" in sample:
                result["true_label"] = sample["label"]
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)} samples")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({
                "sample_id": sample.get("id", i),
                "error": str(e)
            })
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_json}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model weights directory")
    parser.add_argument("--text", type=str, help="Input text for prediction")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--input_json", type=str, help="Path to input JSON file with samples")
    parser.add_argument("--output_json", type=str, help="Path to output JSON file for results")
    parser.add_argument("--no_ensemble", action="store_true", help="Disable ensemble prediction")
    
    args = parser.parse_args()
    
    predictor = MultimodalPredictor(args.config, args.model_dir)
    
    if args.input_json:
        results = predict_from_json(
            args.config,
            args.model_dir,
            args.input_json,
            args.output_json,
            use_ensemble=not args.no_ensemble
        )
    elif args.text or args.image:
        result = predictor.predict(
            text=args.text,
            image_path=args.image,
            use_ensemble=not args.no_ensemble
        )
        print("\n" + "="*50)
        print("Prediction Result:")
        print("="*50)
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
        print("="*50)
    else:
        print("Please provide either --text/--image or --input_json")
