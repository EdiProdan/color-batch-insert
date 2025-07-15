from src.pipeline.image_pipeline import ImagePipeline
from src.pipeline.text_pipeline import TextPipeline
from src.evaluation import EvaluationFramework, ColorBatchInsert, MixAndBatchInsert, NaiveParallelInsert, SequentialInsert


class PipelineController:

    def __init__(self, config):
        self.image_pipeline = None
        self.config = config
        self.text_pipeline = TextPipeline(self.config)
        self.image_pipeline = ImagePipeline(self.config)
        # evaluation: enabled: true
        self.evaluation_enabled = self.config["evaluation"]["enabled"]

    def run_pipeline(self):
        if self.evaluation_enabled:
            print("Evaluation is enabled")
            self.process_evaluation()
        else:
            print("Running full text processing pipeline")
            #self.text_pipeline.process()
            self.image_pipeline.process()
            # self.process_evaluation()

        print("\nPipeline execution completed successfully.")

    def process_evaluation(self):
        """
        Evaluation framework execution with proper algorithm initialization
        """
        print("Initializing evaluation framework...")

        evaluation_framework = None

        try:
            # Initialize evaluation framework
            evaluation_framework = EvaluationFramework(self.config)

            # 1. REGISTER SIMPLE BASELINE (Control Group)
            evaluation_framework.register_algorithm(
                SequentialInsert,
                self.config['algorithms']['simple_baseline']
            # )

            # 2. REGISTER OTHER BASELINES FOR COMPARISON
            # from src.evaluation import BaselineAlgorithm
            evaluation_framework.register_algorithm(
                SimpleParallelBaseline,
                self.config['algorithms']['simple_parallel']
            )
            #
            # # 3. REGISTER PARALLEL APPROACHES
            # from src.evaluation import WorkingParallelBaseline
            # evaluation_framework.register_algorithm(
            #     WorkingParallelBaseline,
            #     self.config['algorithms']['working_parallel']
            # )
            #
            # # 4. REGISTER MIX AND BATCH
            # from src.evaluation import MixAndBatchAlgorithm
            # evaluation_framework.register_algorithm(
            #     MixAndBatchAlgorithm,
            #     self.config['algorithms']['mix_and_batch']
            # )

            # 5. REGISTER YOUR ADAPTIVE ALGORITHM (when ready)
            # from src.evaluation import AdaptiveDynamicAlgorithm
            # evaluation_framework.register_algorithm(
            #     AdaptiveDynamicAlgorithm,
            #     self.config['algorithms']['adaptive_dynamic']
            # )

            # Execute comprehensive evaluation
            print("\nStarting algorithm evaluation experiments...")
            evaluation_framework.run_evaluation()

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise
        finally:
            if evaluation_framework:
                evaluation_framework.close()
