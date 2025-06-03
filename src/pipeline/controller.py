from src.pipeline.text_pipeline import TextPipeline


class PipelineController:

    def __init__(self, config):
        self.config = config
        self.text_pipeline = TextPipeline(self.config)
        # self.image_pipeline = ImagePipeline(self.config)

    def run_pipeline(self):

        print("Starting pipeline execution...")

        self.text_pipeline.process()
        # self.image_pipeline.process()

        # self.graph_loader

        print("\nPipeline execution completed successfully.")
