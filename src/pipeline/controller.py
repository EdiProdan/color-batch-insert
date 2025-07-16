from src.pipeline.image_pipeline import ImagePipeline
from src.pipeline.text_pipeline import TextPipeline
from src.pipeline.data_integration_layer import DataIntegrationLayer
from src.evaluation import EvaluationFramework, ColorBatchInsert, MixAndBatchInsert, NaiveParallelInsert, \
    SequentialInsert, ApocInsert


class PipelineController:

    def __init__(self, config):
        self.image_pipeline = None
        self.config = config
        self.text_pipeline = TextPipeline(self.config)
        self.image_pipeline = ImagePipeline(self.config)
        self.data_integration_layer = DataIntegrationLayer(self.config)
        self.evaluation_enabled = self.config["evaluation"]["enabled"]

    def run_pipeline(self):
        if self.evaluation_enabled:
            self.process_evaluation()
        else:
            image_relationships = self.image_pipeline.process()
            text_relationships = self.text_pipeline.process()
            integrated_relationships = self.data_integration_layer.merge_relationships(
                text_relationships, image_relationships
            )
            self.data_integration_layer.save_results(integrated_relationships)

        print("\nPipeline execution completed successfully.")

    def process_evaluation(self):

        print("Initializing evaluation framework...")

        evaluation_framework = None

        try:
            evaluation_framework = EvaluationFramework(self.config)

            evaluation_framework.register_algorithm(
                SequentialInsert,
                self.config['algorithms']['sequential']
            )

            evaluation_framework.register_algorithm(
                NaiveParallelInsert,
                self.config['algorithms']['naive_parallel']
            )

            evaluation_framework.register_algorithm(
                MixAndBatchInsert,
                self.config['algorithms']['mix_and_batch']
            )

            evaluation_framework.register_algorithm(
                ApocInsert,
                self.config['algorithms']['apoc']
            )

            evaluation_framework.register_algorithm(
                ColorBatchInsert,
                self.config['algorithms']['color_batch']
            )

            print("\nStarting algorithm evaluation experiments...")
            evaluation_framework.run_evaluation()

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise
        finally:
            if evaluation_framework:
                evaluation_framework.close()
