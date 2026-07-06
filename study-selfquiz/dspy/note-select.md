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
- **You believe that BootstrapFinetune raises a ValueError with the message "Predictors must have language models assigned" when a predictor lacks a language model.** BootstrapFinetune raises a ValueError with the message fragment "Predictor {pred_ind} does not have an LM assigned." and informs you to set the LM using your_module.set_lm(your_lm).
  > `dspy/teleprompt/bootstrap_finetune.py:83`: `raise ValueError(
                    f"Predictor {pred_ind} does not have an LM assigned. "
                    f"Please ensure the module's predictors have their LM set before fine-tuning. "
                    f"You can set it using: your_module.set_lm(your_lm)"
                )`

## dspy/adapters
- **You believe the regular expression pattern used is `r'\[\[([^\]]+)\]\]'` which captures any content inside double brackets.** The actual pattern requires the `##` delimiters and specifically captures word characters using `(\w+)`.
  > `dspy/adapters/chat_adapter.py:20`: `field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")`

## dspy/clients
- **You believe the code for detecting whether a model supports function calling, reasoning, or response schema is located in generic utility files or provider adapters that delegate to external libraries.** The detection logic is implemented natively in the `BaseLM` class within `dspy/clients/base_lm.py` via properties that return `False` by default, rather than delegating to external SDKs.
  > `dspy/clients/base_lm.py:70`: `@property
def supports_function_calling(self) -> bool:
    """Whether the model supports function calling (tool use)."""
    return False`

## dspy/predict
- **You believe that an LM instance or a retriever/embedding module is required when passing parameters to the dspy.KNN constructor alongside k and trainset.** You must pass a specific `vectorizer` parameter instead of an LM or generic retriever, and this `vectorizer` argument expects an `Embedder` type.
  > `dspy/predict/knn.py:15`: `vectorizer: The `Embedder` to use for vectorization`

## dspy/primitives
- **You believe CodeInterpreterError is defined in dspy/primitives/python_interpreter.py and FinalOutput might reside in module.py.** Both CodeInterpreterError and FinalOutput are actually defined in dspy/primitives/code_interpreter.py, and their public exports originate from dspy/primitives/__init__.py.
  > `dspy/primitives/__init__.py:2`: `from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput`

## dspy/utils
- **You believe there is no documentation or information about a URL-based file download utility function in dspy.utils.** The utility function is `download` located in `dspy/utils/__init__.py`. When a file already exists at a different size than expected, the function re-downloads the file because the local size does not match the remote Content-Length header.
  > `dspy/utils/__init__.py:19`: `if not os.path.exists(filename) or local_size != remote_size:`

## dspy/dsp
- **You believe the parameter name for enabling POST requests is `use_post` when configuring the `ColBERTv2` class.** You must set the `post_requests` parameter to `True` when initializing the `ColBERTv2` class (not `use_post`). When `self.post_requests` is `True`, the internal retrieval function switches to calling `colbertv2_post_request()` instead of `colbertv2_get_request()`.
  > `dspy/dsp/colbertv2.py:18`: `post_requests: bool = False,`

## dspy/datasets
- **You believe that CSV loading specifications and column key definitions are not covered in available documentation, forcing you to check source code directly without guidance.** You should use the `from_csv` method of the DataLoader class and explicitly pass `input_keys=('label',)` to designate the 'label' column as the input key when loading from a local CSV file.
  > `dspy/datasets/dataloader.py:67`: `input_keys: tuple[str] = (),`

## dspy/signatures
- **You believe that calling `MySig.insert(-1, ...)` on an empty inputs list raises an `IndexError` because DSPy delegates to standard list operations without special negative index handling.** DSPy normalizes negative insertion indices by adding `len(lst) + 1`, allowing `insert(-1)` to successfully place the field at index 0 on an empty list rather than raising an error.
  > `dspy/signatures/signature.py:460`: `# We support negative insert indices
        if index < 0:
            index += len(lst) + 1`

## dspy/retrievers
- **You believe the 'Embeddings' class switches from brute-force search to building a FAISS index during initialization specifically when `candidates` are provided via a `trainset`.** The 'Embeddings' class switches from brute-force search to building a FAISS index during initialization when the length of the corpus is greater than or equal to the `brute_force_threshold` parameter (which defaults to 20,000). If the corpus length is below this threshold, `self.index` is set to `None` and brute-force search is used instead.
  > `dspy/retrievers/embeddings.py:38`: `self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None`

## dspy/streaming
- **You believe that the optional argument passed to provide custom status messages for execution progress is `message_func`.** The correct optional argument passed to wrap a DSPy program for streaming execution is `status_message_provider`, which accepts a `StatusMessageProvider` instance or `None`.
  > `dspy/streaming/streamify.py:29`: `status_message_provider: StatusMessageProvider | None = None,`

## dspy/evaluate
- **You believe the `metrics` parameter (plural) allows you to override the metric defined in the constructor when calling the instantiated `Evaluate` class method.** Actually, the `metric` argument (singular) in the `__call__` method allows you to override the metric defined in the constructor by assigning it before falling back to `self.metric`.
  > `dspy/evaluate/evaluate.py:120`: `metric: Callable | None = None,`
