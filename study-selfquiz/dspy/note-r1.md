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
- **You believe the exact internal parameter validation logic cannot be referenced because the source code was assumed missing.** A ValueError is raised immediately during argument validation if auto is set while manually specifying num_trials or num_candidates.
  > `dspy/teleprompt/mipro_optimizer_v2.py:165`: `if self.auto is not None and (self.num_candidates is not None or num_trials is not None):`

## dspy/adapters
- **you believe the required parameter for initializing a TwoStepAdapter is `model`.** the required parameter is actually `extraction_model`, and it must be an instance of `BaseLM`.
  > `dspy/adapters/two_step_adapter.py:42`: `def __init__(self, extraction_model: BaseLM, **kwargs):`
- **You believe that a demo is considered complete based on generic requirements like mandatory keys, non-null values, and data type consistency.** A demo is considered 'complete' when ALL fields from the signature's `signature.fields` are present and have non-None values. It is considered 'incomplete' only if those criteria fail BUT it contains at least one input field and at least one output field from `signature.input_fields` and `signature.output_fields`.
  > `dspy/adapters/base.py:414`: `is_complete = all(k in demo and demo[k] is not None for k in signature.fields)`
- **You believe the regular expression pattern `r'^\['` identifies field header markers in the LM response within ChatAdapter.parse().** It actually uses the pattern `r"\[\[ ## (\w+) ## \]\]"` to match specifically delimited field headers.
  > `dspy/adapters/chat_adapter.py:20`: `field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")`

## dspy/clients
- **You believe there are multiple parameter options like **`enable_cache`**, **`use_response_cache`**, or **`cache_enabled`** that can be used to disable response caching when creating an LM instance.** You should set the **`cache`** parameter to **`False`** when creating an LM instance to disable response caching, not use any other parameter name like `enable_cache`, `use_response_cache`, or `cache_enabled`.
  > `dspy/clients/lm.py:39`: `cache: bool = True,`
- **You believe the logic for detecting whether a model supports function calling, reasoning, or response schema is defined within core model interface classes of typical orchestration frameworks like LangChain and delegated to provider-specific SDKs.** The inference code is actually defined within the `LM` class in `dspy/clients/lm.py` as property methods (`supports_function_calling`, `supports_reasoning`, `supports_response_schema`) that delegate specifically to the `litellm` library.
  > `dspy/clients/lm.py:124`: `return litellm.supports_function_calling(model=self.model)`
- **You believe the transfer of cached data from disk to memory is a secondary step occurring after the data is fully processed or loaded into the application context.** The system performs a direct addition (promotion) of disk-retrieved entries into the memory cache immediately upon detecting a disk hit, optimizing future latency before returning the response.
  > `dspy/clients/cache.py:117`: `# Found on disk but not in memory cache, add to memory cache`

## dspy/predict
- **you believe that passing positional arguments to a dspy.Predict module results in a TypeError** passing positional arguments raises a ValueError requiring keyword arguments matching signature input fields
  > `dspy/predict/predict.py:129`: `raise ValueError(self._get_positional_args_error_message())`
- **You believe tools simply need to be wrapped in a `list` data structure before use.** Every tool must be wrapped in a `dspy.Tool` instance before use, though the class may auto-convert regular callables.
  > `dspy/predict/react.py:44`: `tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]`
- **You believe dspy.ChainOfThought modifies the underlying Predict module by overriding or extending its prompt template.** It modifies the underlying Predict module by creating an extended_signature that includes the new field and passing that extended signature to the dspy.Predict constructor.
  > `dspy/predict/chain_of_thought.py:35`: `self.predict = dspy.Predict(extended_signature, **config)`
