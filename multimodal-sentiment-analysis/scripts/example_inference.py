"""
Example script demonstrating how to use the inference module
for multimodal sentiment analysis prediction.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.inference import MultimodalPredictor


def main():
    config_path = "config_gpu.json"
    model_dir = "results/models"
    
    print("Initializing Multimodal Predictor...")
    predictor = MultimodalPredictor(config_path, model_dir)
    
    print("\n" + "="*60)
    print("Example 1: Cross-Attention Model Prediction")
    print("="*60)
    
    text = "I absolutely love this product! The quality is amazing and the design is beautiful."
    image_path = "path/to/your/image.jpg"
    
    if os.path.exists(image_path):
        result = predictor.predict_single_model(
            text=text,
            image_path=image_path,
            model_name="cross_attention"
        )
        
        print(f"\nInput Text: {text}")
        print(f"Input Image: {image_path}")
        print(f"\nPredicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:10s}: {prob:.4f} {bar}")
    else:
        print(f"\nImage not found at {image_path}")
        print("Please provide a valid image path to test Cross-Attention model")
    
    print("\n" + "="*60)
    print("Example 2: Text-Only Model Prediction")
    print("="*60)
    
    if "text_only" in predictor.models:
        result = predictor.predict_single_model(
            text=text,
            model_name="text_only"
        )
        
        print(f"\nInput Text: {text}")
        print(f"\nPredicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:10s}: {prob:.4f} {bar}")
    
    print("\n" + "="*60)
    print("Example 3: Ensemble Prediction (if meta-learner available)")
    print("="*60)
    
    if predictor.meta_learner and os.path.exists(image_path):
        result = predictor.predict_ensemble(
            text=text,
            image_path=image_path
        )
        
        print(f"\nInput Text: {text}")
        print(f"Input Image: {image_path}")
        print(f"\nPredicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"  {label:10s}: {prob:.4f} {bar}")
    else:
        print("\nMeta-learner not available. Train meta-learner first to use ensemble prediction.")
    
    print("\n" + "="*60)
    print("Example 4: Batch Prediction from JSON")
    print("="*60)
    
    print("""
To run batch prediction from a JSON file:

python src/inference/inference.py \\
    --config config_gpu.json \\
    --model_dir results/models \\
    --input_json data/test_samples.json \\
    --output_json results/predictions.json

Input JSON format:
[
    {
        "id": "sample_001",
        "text": "Sample text content",
        "image_path": "/path/to/image.jpg",
        "label": "positive"
    }
]
    """)


if __name__ == "__main__":
    main()
