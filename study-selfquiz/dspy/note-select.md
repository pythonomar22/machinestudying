# DSPy — corrections from studying (your beliefs vs. this repository)

## Repo map
- `dspy/teleprompt/`: grpo.py, mipro_optimizer_v2.py, gepa.py
- `dspy/adapters/`: base.py, tool.py, json_adapter.py
- `dspy/clients/`: lm.py, lm_local.py, databricks.py
- `dspy/predict/`: rlm.py, predict.py, react.py
- `dspy/primitives/`: python_interpreter.py, runner.js, module.py
- `dspy/utils/`: callback.py, parallelizer.py, dummies.py
- `dspy/dsp/`: settings.py, colbertv2.py, dpr.py
- `dspy/datasets/`: base_config.yml, dataloader.py, dataset.py
- `dspy/signatures/`: signature.py, field.py, __init__.py
- `dspy/retrievers/`: databricks_rm.py, embeddings.py, weaviate_rm.py
- `dspy/streaming/`: streaming_listener.py, streamify.py, messages.py
- `dspy/evaluate/`: evaluate.py, metrics.py, auto_evaluation.py
- `dspy/propose/`: grounded_proposer.py, utils.py, dataset_summary_generator.py
- `dspy/`: __init__.py, __metadata__.py
- `dspy/experimental/`: __init__.py

## dspy/teleprompt
- **You believe BootstrapFewShot validates the student program's syntax and runtime execution capabilities within the `_bootstrap()` method to catch immediate exceptions like `SyntaxError` or `ImportError`.** The system instead validates that the student program is UNCOMPILED within `_prepare_student_and_teacher()` to ensure fresh traces are generated without reusing cached predictions from the student model.
  > `dspy/teleprompt/bootstrap.py:102`: `assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."`

## dspy/adapters
- **you believe the required parameter for initializing a TwoStepAdapter is `model`.** the required parameter is actually `extraction_model`, and it must be an instance of `BaseLM`.
  > `dspy/adapters/two_step_adapter.py:42`: `def __init__(self, extraction_model: BaseLM, **kwargs):`

## dspy/clients
- **You believe there are multiple parameter options like **`enable_cache`**, **`use_response_cache`**, or **`cache_enabled`** that can be used to disable response caching when creating an LM instance.** You should set the **`cache`** parameter to **`False`** when creating an LM instance to disable response caching, not use any other parameter name like `enable_cache`, `use_response_cache`, or `cache_enabled`.
  > `dspy/clients/lm.py:39`: `cache: bool = True,`

## dspy/predict
- **you believe that passing positional arguments to a dspy.Predict module results in a TypeError** passing positional arguments raises a ValueError requiring keyword arguments matching signature input fields
  > `dspy/predict/predict.py:129`: `raise ValueError(self._get_positional_args_error_message())`

## dspy/primitives
- **you believe that if neither Prediction has a 'score' field, comparison operations (<, >, <=, >=) will typically return False rather than raising an exception.** Attempting to compare Prediction objects using these operators when they lack a 'score' field raises a ValueError with the message "Prediction object does not have a 'score' field to convert to float.".
  > `dspy/primitives/prediction.py:55`: `raise ValueError("Prediction object does not have a 'score' field to convert to float.")`

## dspy/utils
- **you believe the prefix in the exception's string representation is "Context window"** the prefix consists of the model identifier enclosed in square brackets followed by a single space
  > `dspy/utils/exceptions.py:21`: `prefix = f"[{model}] " if model else ""`

## dspy/dsp
- **You believe that omitting the `k` keyword argument defaults to a system-defined context limit or a batch size like 100, rather than a fixed small integer.** The method signature explicitly sets the default value for `k` to 7, regardless of system configuration.
  > `dspy/dsp/colbertv2.py:165`: `def forward(self, query: str, k: int = 7, **kwargs):`

## dspy/datasets
- **You believe that the `load_dataset_from_csv()` method (or similar variants like `load_from_file`) is the correct approach to load a dataset from a local CSV file when using the `DataLoader` class.** You must call the `from_csv` method of the `DataLoader` class, ensuring you pass the file location as the `file_path` argument and specify the input columns as a tuple within the `input_keys` argument.
  > `dspy/datasets/dataloader.py:63`: `def from_csv(`

## dspy/signatures
- **you believe that calling `MySig.insert(-1, ...)` on an empty inputs list raises an `IndexError`.** actually, calling `MySig.insert(-1, ...)` on an empty inputs list converts the negative index to 0 using the formula `index += len(lst) + 1`, so the field is successfully inserted at position 0.
  > `dspy/signatures/signature.py:461`: `if index < 0:
            index += len(lst) + 1`

## dspy/retrievers
- **you believe that the provided note content lacks specific information regarding how the 'forward' method input handling differs between the base 'Retrieve' class and the 'WeaviateRM' subclass for multiple queries.** Actually, the WeaviateRM implementation accepts both `str` and `list[str]` via `query_or_queries: str | list[str]` with built-in batching logic, while the base Retrieve class only accepts a single `str`.
  > `dspy/retrievers/weaviate_rm.py:73`: `def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> Prediction:`

## dspy/streaming
- **you believe the optional argument to provide custom status messages when wrapping a DSPy program is `status_callback` or `status_message`.** you should pass the argument `status_message_provider` to configure custom status messages when wrapping a DSPy program with streaming.
  > `dspy/streaming/streamify.py:29`: `status_message_provider: StatusMessageProvider | None = None,`

## dspy/evaluate
- **you believe that -1.0 replaces the None tuple for an item's score in the final results.** 0.0 replaces the None tuple in the final results by default, controlled by the `failure_score` parameter.
  > `dspy/evaluate/evaluate.py:81`: `failure_score: float = 0.0`
