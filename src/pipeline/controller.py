from src.pipeline.text_pipeline import TextPipeline


class PipelineController:

    def __init__(self, config):
        self.config = config
        self.text_pipeline = TextPipeline(self.config)
        # TODO: self.image_pipeline = ImagePipeline(config)

    def run_pipeline(self):
        # Phase 1: Content Extraction
        self.text_pipeline.process()

