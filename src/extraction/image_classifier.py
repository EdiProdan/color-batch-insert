import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


class ImageClassifier:

    def __init__(self, config):
        self.config = config
        self.confidence_threshold = 0.1
        self.max_entities = 10

        print("Loading pre-trained ResNet50 model...")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()

        self.class_labels = self._load_imagenet_labels()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Image classifier initialized with confidence threshold: {self.confidence_threshold}")

    def _load_imagenet_labels(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        return weights.meta["categories"]

    def classify_image(self, image_path):
        try:

            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            entities = []
            top_indices = torch.argsort(probabilities, descending=True)

            for idx in top_indices[:self.max_entities]:
                confidence = probabilities[idx].item()
                if confidence >= self.confidence_threshold:
                    label = self.class_labels[idx]
                    entity = self._clean_label(label)
                    if entity and entity not in entities:
                        entities.append(entity)
                else:
                    break

            return entities

        except Exception as e:
            print(f"Error classifying image {image_path}: {e}")
            return []

    def _clean_label(self, label):
        if not label:
            return None

        entity = label.split(',')[0].strip()

        unwanted_patterns = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
        for pattern in unwanted_patterns:
            if pattern in entity:
                return None

        generic_labels = {
            'web site', 'website', 'screen', 'monitor', 'display',
            'background', 'texture', 'pattern', 'color', 'shape'
        }

        if entity.lower() in generic_labels:
            return None

        if len(entity) < 3:
            return None

        return entity

    def extract_entities(self, image_paths):

        results = {}

        print(f"Classifying {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images")

            entities = self.classify_image(image_path)
            results[image_path] = entities

        return results

    def build_relationships(self, image_data):

        relationships = []
        for item in image_data:
            title = item['title']
            for entity in item['entities']:
                relationships.append({"from": title, "to": entity})

        return {"LINKS_TO": relationships}
