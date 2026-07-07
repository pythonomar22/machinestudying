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
- **You believe there is no explicit code excerpt or documentation covering the specific parameter validation logic for MIPROv2's `compile()` method when both `auto='medium'` and `num_trials=10` are used simultaneously.** The validation logic exists directly in `dspy/teleprompt/mipro_optimizer_v2.py`. When `auto='medium'` is specified alongside `num_trials=10`, a `ValueError` is raised because the optimizer prevents manual overriding of automatic hyperparameter selection. The error requires you to either set `auto=None` or not specify candidate/trial counts.
  > `dspy/teleprompt/mipro_optimizer_v2.py:165`: `if self.auto is not None and (self.num_candidates is not None or num_trials is not None):`

## dspy/adapters
- **You believe the parameter required to initialize a TwoStepAdapter that must be an instance of BaseLM is named `model`.** The correct parameter name is `extraction_model`.
  > `dspy/adapters/two_step_adapter.py:42`: `def __init__(self, extraction_model: BaseLM, **kwargs):`
- **You believe the regular expression pattern used is `r'\[\[([^\]]+)\]\]'` which captures any content inside double brackets.** The actual pattern requires the `##` delimiters and specifically captures word characters using `(\w+)`.
  > `dspy/adapters/chat_adapter.py:20`: `field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")`
- **you believe a demo is considered 'complete' if all mandatory keys are present and the corresponding values are non-null and not empty strings.** a demo is classified as complete if ALL signature fields are present and non-None, and incomplete if missing fields but retaining at least one input and one output field, while others are discarded.
  > `dspy/adapters/base.py:414`: ``is_complete = all(k in demo and demo[k] is not None for k in signature.fields)``
- **you believe that ValueError is raised by ChatAdapter.parse() if the resulting dictionary keys do not match the signature's output field names exactly** an AdapterParseError is raised when the resulting dictionary keys do not match the signature's output field names exactly
  > `dspy/adapters/chat_adapter.py:239`: `raise AdapterParseError(`
- **you believe that a demo is classified as complete solely if all signature fields are present and non-None, unaware that input and output field presence is also required to validate the demo.** besides checking signature fields, the method specifically checks if the demo has at least one input field and one output field to ensure validity.
  > `dspy/adapters/base.py:417`: `has_input = any(k in demo for k in signature.input_fields)`

## dspy/clients
- **you believe that setting the cache parameter to `None` or using varying parameter names across frameworks like LangChain is sufficient to disable response caching.** in dspy, you must explicitly set the `cache` parameter to `False` when creating the LM instance.
  > `dspy/clients/base_lm.py:63`: `def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):`
- **You believe the code for detecting whether a model supports function calling, reasoning, or response schema is located in generic utility files or provider adapters that delegate to external libraries.** The detection logic is implemented natively in the `BaseLM` class within `dspy/clients/base_lm.py` via properties that return `False` by default, rather than delegating to external SDKs.
  > `dspy/clients/base_lm.py:70`: `@property
def supports_function_calling(self) -> bool:
    """Whether the model supports function calling (tool use)."""
    return False`
- **You believe that passing a `rollout_id` alongside `temperature=0` allows DSPy to track the inference request with that identifier for caching or experiment association despite the deterministic sampling strategy.** DSPy logs a warning that the parameter is ineffective and explicitly removes `rollout_id` from the arguments before forwarding the request.
  > `dspy/clients/lm.py:142`: `rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.`
- **You believe that the LM.cache system utilizes a spillover policy where entries are initially stored in memory and subsequently moved to the disk cache if the memory usage reaches a threshold.** In reality, the system stores the response in both the memory cache and the disk cache independently during the `cache.put()` operation. There is no automatic migration of new entries from memory to disk based on capacity; disk write failures are logged but do not block memory storage. Additionally, disk cache hits are promoted to the memory cache upon retrieval.
  > `dspy/clients/cache.py:157`: `self.disk_cache[key] = value`
- **You believe that the implementation explicitly removes `rollout_id` from the arguments before forwarding the request when temperature is 0.** The system issues a warning but does not remove the argument; the parameter has no practical effect because the cache key construction does not incorporate `rollout_id`, resulting in the same cached response for identical prompts regardless of the ID value.
  > `dspy/clients/lm.py:142`: `"rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache."`

## dspy/predict
- **You believe that `dspy.Refine` final prediction selection differs from `dspy.BestOfN` only through sequential input passing versus parallel generation, without recognizing the critical feedback generation mechanism that defines `Refine`'s adaptive loop.** `dspy.Refine` and `dspy.BestOfN` share identical initial reward-based selection and threshold exit logic—but `Refine` uniquely implements **feedback generation** (`advice`) after each failed attempt that gets fed into subsequent iterations via `hint_`, whereas `BestOfN` runs completely independent trials with no adaptive behavior.
  > `dspy/predict/refine.py:167`: `advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice`
- **You believe that `dspy.ChainOfThought` updates the underlying `Predict` module's configuration directly in place to accommodate the reasoning field.** `ChainOfThought` creates a new instance of `dspy.Predict` with the extended signature instead of modifying the existing object.
  > `dspy/predict/chain_of_thought.py:35`: `self.predict = dspy.Predict(extended_signature, **config)`
- **You believe that tools only need to be organized in a `list` or `dict` structure without necessarily being instantiated as `dspy.Tool` objects.** All tools must be wrapped in `dspy.Tool` instances before use.
  > `dspy/predict/react.py:44`: `tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]`
- **You believe that an LM instance or a retriever/embedding module is required when passing parameters to the dspy.KNN constructor alongside k and trainset.** You must pass a specific `vectorizer` parameter instead of an LM or generic retriever, and this `vectorizer` argument expects an `Embedder` type.
  > `dspy/predict/knn.py:15`: `vectorizer: The `Embedder` to use for vectorization`
- **You believe all tools must be wrapped in `dspy.Tool` instances before use.** The `tools` parameter requires a list where items can be plain callables or existing `Tool` instances; non-`Tool` items are automatically converted during initialization.
  > `dspy/predict/react.py:44`: `tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]`

## dspy/primitives
- **You believe that DSPy automatically determines which fields serve as inputs to a module based solely on the Module's signature definition (using InputField/OutputField annotations), without requiring an explicit specification on the Example object itself.** You must call .with_inputs() on the Example object after initialization to explicitly declare which fields should be passed to the module during inference. All other fields automatically become available as labels/metadata through the example.labels() method.
  > `dspy/primitives/example.py:28`: `.with_inputs("question")`
- **You believe CodeInterpreterError is defined in dspy/primitives/python_interpreter.py and FinalOutput might reside in module.py.** Both CodeInterpreterError and FinalOutput are actually defined in dspy/primitives/code_interpreter.py, and their public exports originate from dspy/primitives/__init__.py.
  > `dspy/primitives/__init__.py:2`: `from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput`
- **you believe that comparing two Prediction objects without a valid 'score' field causes the comparison to simply evaluate to False or be treated as invalid rather than raising an exception** if neither Prediction has a 'score' field, attempting to use comparison operators (<, >, <=, >=) raises a ValueError with the message "Prediction object does not have a 'score' field to convert to float." because the comparison relies on __float__() which requires the score field
  > `dspy/primitives/prediction.py:55`: `raise ValueError("Prediction object does not have a 'score' field to convert to float.")`
- **You believe the restriction exists to prevent bypassing signature validation and internal state management during predictor routing.** The restriction exists to ensure LM usage tracking functions correctly by capturing token usage and recording statistics within the `__call__` method.
  > `dspy/primitives/module.py:106`: `self._set_lm_usage(tokens, output)`

## dspy/utils
- **You believe there is no documentation or information about a URL-based file download utility function in dspy.utils.** The utility function is `download` located in `dspy/utils/__init__.py`. When a file already exists at a different size than expected, the function re-downloads the file because the local size does not match the remote Content-Length header.
  > `dspy/utils/__init__.py:19`: `if not os.path.exists(filename) or local_size != remote_size:`
- **You believe the exception message contains no model identifier prefix.** When raising a ContextWindowExceededError with a model identifier like 'gpt-4', the message string will begin with the model identifier enclosed in square brackets followed by a space (e.g., `[gpt-4] `).
  > `dspy/utils/exceptions.py:21`: `prefix = f"[{model}] " if model else ""`

## dspy/dsp
- **You believe the parameter name for enabling POST requests is `use_post` when configuring the `ColBERTv2` class.** You must set the `post_requests` parameter to `True` when initializing the `ColBERTv2` class (not `use_post`). When `self.post_requests` is `True`, the internal retrieval function switches to calling `colbertv2_post_request()` instead of `colbertv2_get_request()`.
  > `dspy/dsp/colbertv2.py:18`: `post_requests: bool = False,`
- **You believe the way to change settings in another thread is to instantiate Local Clients with specific parameters or apply configurations on the main thread before spawning workers.** To change settings in another thread, you must use `dspy.context(**kwargs)` which creates thread-local overrides that propagate to child threads within DSPy primitives.
  > `dspy/dsp/utils/settings.py:161`: `raise RuntimeError(
    "dspy.configure(...) can only be called from the same async task that called it first. Please "
    "use `dspy.context(...)` in other async tasks instead."
)`
- **You believe that omitting the `k` keyword argument utilizes a default of `None` or a configured maximum limit, potentially returning all retrieved candidates above the relevance threshold.** The method actually defaults `k` to `7` specifically in its signature, restricting the return to 7 passages rather than all available ones.
  > `dspy/dsp/colbertv2.py:165`: `def forward(self, query: str, k: int = 7, **kwargs):`

## dspy/datasets
- **You believe that CSV loading specifications and column key definitions are not covered in available documentation, forcing you to check source code directly without guidance.** You should use the `from_csv` method of the DataLoader class and explicitly pass `input_keys=('label',)` to designate the 'label' column as the input key when loading from a local CSV file.
  > `dspy/datasets/dataloader.py:67`: `input_keys: tuple[str] = (),`
- **You believe the `.train` property of the `Dataset` class does not automatically cache the result internally and is re-evaluated every time it is accessed.** The `.train` property automatically caches the result internally. On first access, it computes and stores the shuffled/sampled training data in `self._train_`. Subsequent accesses return this cached value without re-computation. The cache is invalidated when `reset_seeds()` is called.
  > `dspy/datasets/dataset.py:60`: `if not hasattr(self, "_train_"):`
- **You believe there is no information available about a `Colors` dataset implementation within DSPy and that the sorting logic for grouping color names by suffixes is not documented.** The sorting logic is applied in the `sorted_by_suffix` method of the `Colors` dataset class.
  > `dspy/datasets/colors.py:165`: `def sorted_by_suffix(self, colors):`

## dspy/signatures
- **You believe that calling `MySig.insert(-1, ...)` on an empty inputs list raises an `IndexError` because DSPy delegates to standard list operations without special negative index handling.** DSPy normalizes negative insertion indices by adding `len(lst) + 1`, allowing `insert(-1)` to successfully place the field at index 0 on an empty list rather than raising an error.
  > `dspy/signatures/signature.py:460`: `# We support negative insert indices
        if index < 0:
            index += len(lst) + 1`
- **You believe that DSPy Signature fields should be defined without type annotations and that when calling through a Predict module, you must explicitly pass all input/output field names as separate keyword arguments including output fields with None values.** Define Signature fields with proper type annotations (e.g., `input: str = InputField(desc="...")`) before the field assignment. When creating a signature instance, pass only input values as keyword arguments during initialization, and let the signature handle output prediction through its `.predict()` method. Instructions belong in the class docstring, not as regular comments.
  > `dspy/signatures/signature.py:3`: `class MySignature(dspy.Signature):
    input: str = InputField(desc="...")
    output: int = OutputField(desc="...")`

## dspy/retrievers
- **You believe the 'Embeddings' class switches from brute-force search to building a FAISS index during initialization specifically when `candidates` are provided via a `trainset`.** The 'Embeddings' class switches from brute-force search to building a FAISS index during initialization when the length of the corpus is greater than or equal to the `brute_force_threshold` parameter (which defaults to 20,000). If the corpus length is below this threshold, `self.index` is set to `None` and brute-force search is used instead.
  > `dspy/retrievers/embeddings.py:38`: `self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None`
- **you believe that the 'Embeddings' retriever accepts the 'cache' argument following general DSPy patterns seen in LMs, implying you might simply need to disable it, and that this specific behavior is not explicitly documented in the study notes.** Initializing the 'Embeddings' retriever with 'cache' set to True triggers an AssertionError immediately during initialization because caching is explicitly forbidden for this retriever type.
  > `dspy/retrievers/embeddings.py:28`: `"Caching is not supported for embeddings-based retrievers"`
- **You believe the specific details about the input handling differences between the base Retrieve class and the WeaviateRM subclass regarding multiple queries are not covered in the referenced documentation sections.** In reality, the base Retrieve.forward() method strictly accepts a single query string (`query: str`), while the WeaviateRM.forward() method supports batched query processing by accepting either a single string or a list of strings (`str | list[str]`) and iterating through them internally.
  > `dspy/retrievers/weaviate_rm.py:73`: `def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs)`

## dspy/streaming
- **You believe that the optional argument passed to provide custom status messages for execution progress is `message_func`.** The correct optional argument passed to wrap a DSPy program for streaming execution is `status_message_provider`, which accepts a `StatusMessageProvider` instance or `None`.
  > `dspy/streaming/streamify.py:29`: `status_message_provider: StatusMessageProvider | None = None,`
- **You believe that the streaming_response utility formats output by directly concatenating the `data:` prefix to raw text chunks without first serializing them into JSON.** The utility actually serializes each data chunk as JSON using `orjson.dumps()` before decoding the result and wrapping it in the SSE-compliant delimiter.
  > `dspy/streaming/streamify.py:273`: `yield f"data: {orjson.dumps(data).decode()}\n\n"`

## dspy/evaluate
- **You believe the `metrics` parameter (plural) allows you to override the metric defined in the constructor when calling the instantiated `Evaluate` class method.** Actually, the `metric` argument (singular) in the `__call__` method allows you to override the metric defined in the constructor by assigning it before falling back to `self.metric`.
  > `dspy/evaluate/evaluate.py:120`: `metric: Callable | None = None,`
- **You believe passing the `return_outputs` keyword argument to the `Evaluate` constructor raises a TypeError because the parameter is not recognized in the initialization method signature.** A ValueError is actually raised with the specific message that "`return_outputs` is no longer supported" because DSPy explicitly checks for this deprecated keyword and raises a ValueError when detected.
  > `dspy/evaluate/evaluate.py:114`: `raise ValueError("`return_outputs` is no longer supported. Results are always returned inside the `results` field of the `EvaluationResult` object.")`
- **You believe that there is no explicit function documented in the DSPy study notes for removing English articles ('a', 'an', 'the') during text normalization.** The actual implementation uses the `remove_articles` function, which is a nested helper within the `normalize_text` function located in `dspy/evaluate/metrics.py`.
  > `dspy/evaluate/metrics.py:110`: `def remove_articles(text):`

## dspy/propose
- **You believe that initializing GroundedProposer with program_aware=True causes automatic disablement due to issues like no retriever configured, missing corpus, or signature mismatches.** Actually, the proposer disables itself without raising an exception specifically when retrieving the program's source code fails, triggering an exception in get_dspy_source_code which sets self.program_aware to False.
  > `dspy/propose/grounded_proposer.py:291`: `self.program_aware = False`
- **You believe that the provided study notes contain no information about the `create_dataset_summary` function or its behavior when receiving multiple consecutive 'COMPLETE' responses from the observation model.** The function modifies processing by skipping data accumulation for affected batches and exits the main iteration loop immediately once a threshold of 5 consecutive 'COMPLETE' responses is reached.
  > `dspy/propose/dataset_summary_generator.py:75`: `if skips >= 5:`
- **You believe that the value of `use_tip` during execution is determined by the success of a retrieval operation, specifically checking if `is_grounded_by_retrieval()` returns true or if `has_results()` returns True.** Actually, `use_tip` is determined solely by the truthiness of the selected tip string retrieved from the TIPS dictionary, independent of retrieval status; `self.use_tip` is set to `bool(selected_tip)`, meaning empty strings (e.g., the "none" tip) yield False while non-empty text yields True.
  > `dspy/propose/grounded_proposer.py:346`: `self.use_tip = bool(`

## dspy
- **You believe that global configuration requires using `dspy.settings.configure` or explicit module assignment rather than passing the LM instance directly to `dspy.configure`.** Instantiate the LM using `dspy.LM()` and pass that object directly as the `lm` argument to `dspy.configure`.
  > `dspy/predict/predict.py:151`: ``dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))``
- **You believe that passing positional arguments to a `dspy.Predict` instance results in a generic error rather than raising a specific `ValueError`.** Calling a `dspy.Predict` module instance with positional arguments specifically raises a `ValueError` instructing users to use keyword arguments.
  > `dspy/predict/predict.py:121`: `Positional arguments are not allowed when calling `dspy.Predict`, must use keyword arguments`

## dspy/experimental
- **You believe the provided study notes do not contain specific information about where the actual `Document` type implementation is located when importing from `dspy.experimental`.** The actual `Document` type implementation is located at `dspy/adapters/types/document.py` and is re-exported by `dspy/experimental/__init__.py` via an import statement.
  > `dspy/experimental/__init__.py:2`: `from dspy.adapters.types.document import Document`
- **You believe the specific items exported by the `__all__` list in `dspy/experimental/__init__.py` are unspecified or unknown from the provided context.** The `__all__` list explicitly exports the public API items `"Citations"` and `"Document"`.
  > `dspy/experimental/__init__.py:5`: `"Citations",`
