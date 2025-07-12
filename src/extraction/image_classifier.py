import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np


class ImageClassifier:
    """
    Simple image classifier using pre-trained ResNet50 for entity extraction.

    This classifier identifies objects, scenes, and concepts in images to extract
    meaningful entities for knowledge graph construction.
    """

    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config.get('image_classification', {}).get('confidence_threshold', 0.1)
        self.max_entities = config.get('image_classification', {}).get('max_entities', 10)

        # Load pre-trained ResNet50 model
        print("Loading pre-trained ResNet50 model...")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()

        # Load ImageNet class labels
        self.class_labels = self._load_imagenet_labels()

        # Define image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Image classifier initialized with confidence threshold: {self.confidence_threshold}")

    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        # Using torchvision's built-in ImageNet labels
        weights = ResNet50_Weights.IMAGENET1K_V2
        return weights.meta["categories"]

    def classify_image(self, image_path):
        """
        Classify an image and extract meaningful entities.

        Args:
            image_path (str): Path to the image file

        Returns:
            list: List of extracted entity names
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Extract top predictions above confidence threshold
            entities = []
            top_indices = torch.argsort(probabilities, descending=True)

            for idx in top_indices[:self.max_entities]:
                confidence = probabilities[idx].item()
                if confidence >= self.confidence_threshold:
                    label = self.class_labels[idx]
                    # Clean up the label to make it more entity-like
                    entity = self._clean_label(label)
                    if entity and entity not in entities:
                        entities.append(entity)
                else:
                    break  # Stop when confidence drops below threshold

            return entities

        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            return []

    def _clean_label(self, label):
        """
        Clean ImageNet labels to create meaningful entity names.

        Args:
            label (str): Raw ImageNet label

        Returns:
            str: Cleaned entity name
        """
        # Remove common ImageNet artifacts
        if not label:
            return None

        # Split on comma and take first part (ImageNet often has multiple synonyms)
        entity = label.split(',')[0].strip()

        # Remove unwanted patterns
        unwanted_patterns = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
        for pattern in unwanted_patterns:
            if pattern in entity:
                return None

        # Filter out very generic or unhelpful labels
        generic_labels = {
            'web site', 'website', 'screen', 'monitor', 'display',
            'background', 'texture', 'pattern', 'color', 'shape'
        }

        if entity.lower() in generic_labels:
            return None

        # Ensure minimum length
        if len(entity) < 3:
            return None

        return entity

    def extract_entities(self, image_paths):
        """
        Extract entities from multiple images.

        Args:
            image_paths (list): List of image file paths

        Returns:
            dict: Dictionary mapping image paths to extracted entities
        """
        results = {}

        print(f"Classifying {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images")

            entities = self.classify_image(image_path)
            results[image_path] = entities

        return results

    def get_supported_formats(self):
        """Return list of supported image formats"""
        return ['.jpg', '.jpeg', '.png']