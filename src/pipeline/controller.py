from src.pipeline.text_pipeline import TextPipeline
from src.evaluation import EvaluationFramework, BaselineAlgorithm


class PipelineController:

    def __init__(self, config):
        self.config = config
        self.text_pipeline = TextPipeline(self.config)
        # self.image_pipeline = ImagePipeline(self.config)
        # evaluation: enabled: true
        self.evaluation_enabled = self.config["evaluation"]["enabled"]

    def run_pipeline(self):
        if self.evaluation_enabled:
            print("Evaluation is enabled")
            self.process_evaluation()
        else:
            print("Running full text processing pipeline")
            self.text_pipeline.process()
            # self.image_pipeline.process()
            # self.process_evaluation()

        print("\nPipeline execution completed successfully.")

    def process_evaluation(self):
        """
        Evaluation framework execution

        This mode runs your algorithm comparison experiments.
        Requires relationship data to already exist from data processing mode.
        """
        print("Initializing evaluation framework...")

        evaluation_framework = None

        try:
            # Initialize evaluation framework
            evaluation_framework = EvaluationFramework(self.config)

            # Register all algorithms for comparison
            evaluation_framework.register_algorithm(
                BaselineAlgorithm,
                self.config['algorithms']['baseline']
            )

            # Execute comprehensive evaluation
            print("\nStarting algorithm evaluation experiments...")
            evaluation_framework.run_evaluation()

        except Exception as e:
            print(f"Evaluation failed: {e}")
            print("Ensure relationship data exists from data processing mode")
            raise
        finally:
            if evaluation_framework:
                evaluation_framework.close()
