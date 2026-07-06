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
- **You expect BootstrapFinetune to raise ValueError 'Predictors must have language models assigned' for missing LMs.** Raises ValueError 'Predictor {pred_ind} does not have an LM assigned.'; set LM using your_module.set_lm(your_lm).
  > `dspy/teleprompt/bootstrap_finetune.py:83`: `raise ValueError(
                    f"Predictor {pred_ind} does not have an LM assigned. "
                    f"Please ensure the module's predictors have their LM set before fine-tuning. "
                    f"You can set it using: your_module.set_lm(your_lm)"
                )`
- **Assumed no documentation covers MIPROv2 compile() validation for conflicting auto/hyperparameters.** Validation logic is in dspy/teleprompt/mipro_optimizer_v2.py. auto='medium' with explicit trials triggers ValueError; use auto=None.
  > `dspy/teleprompt/mipro_optimizer_v2.py:165`: `if self.auto is not None and (self.num_candidates is not None or num_trials is not None):`

## dspy/adapters
- **You believe the parameter required to initialize a TwoStepAdapter that must be an instance of BaseLM is named `model`.** The correct parameter name is `extraction_model`.
  > `dspy/adapters/two_step_adapter.py:42`: `def __init__(self, extraction_model: BaseLM, **kwargs):`
- **You believe the regular expression pattern used is `r'[[(\[^]+\)]]'` which captures any content inside double brackets.** The actual pattern requires the `##` delimiters and specifically captures word characters using `(\w+)`.
  > `dspy/adapters/chat_adapter.py:20`: `field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")`
- **You believe a demo is classified as 'complete' solely if all signature fields are present and non-None.** A demo is complete if ALL signature fields are present and non-None; however, the method also validates that there is at least one input and one output field present to ensure the demo is valid.
  > `dspy/adapters/base.py:414`: ``is_complete = all(k in demo and demo[k] is not None for k in signature.fields)``
- **You believe that ValueError is raised by ChatAdapter.parse() if the resulting dictionary keys do not match the signature's output field names exactly.** An AdapterParseError is raised when the resulting dictionary keys do not match the signature's output field names exactly.
  > `dspy/adapters/chat_adapter.py:239`: `raise AdapterParseError(`

## dspy/clients
- **you believe that setting the cache parameter to `None` or using varying parameter names across frameworks like LangChain is sufficient to disable response caching.** in dspy, you must explicitly set the `cache` parameter to `False` when creating the LM instance.
  > `dspy/clients/base_lm.py:63`: `def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):`
- **the code for detecting whether a model supports function calling, reasoning, or response schema is located in generic utility files or provider adapters that delegate to external libraries.** The detection logic is implemented natively in the `BaseLM` class within `dspy/clients/base_lm.py` via properties that return `False` by default, rather than delegating to external SDKs.
  > `dspy/clients/base_lm.py:70`: `@property
def supports_function_calling(self) -> bool:
    """Whether the model supports function calling (tool use)."""
    return False`
- **passing a `rollout_id` alongside `temperature=0` allows DSPy to track the inference request with that identifier for caching or experiment association despite the deterministic sampling strategy.** The system issues a warning that the parameter is ineffective because the cache key construction does not incorporate `rollout_id`, resulting in the same cached response for identical prompts regardless of the ID value.
  > `dspy/clients/lm.py:142`: `"rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache."`
- **the LM.cache system utilizes a spillover policy where entries are initially stored in memory and subsequently moved to the disk cache if the memory usage reaches a threshold.** In reality, the system stores the response in both the memory cache and the disk cache independently during the `cache.put()` operation. There is no automatic migration of new entries from memory to disk based on capacity; disk write failures are logged but do not block memory storage. Additionally, disk cache hits are promoted to the memory cache upon retrieval.
  > `dspy/clients/cache.py:157`: `self.disk_cache[key] = value`

## dspy/predict
- **You believe that `dspy.Refine` final prediction selection differs from `dspy.BestOfN` only through sequential input passing versus parallel generation, without recognizing the critical feedback generation mechanism that defines `Refine`'s adaptive loop.** `Refine` and `BestOfN` share identical initial reward-based selection and threshold exit logic—but `Refine` uniquely implements **feedback generation** (`advice`) after each failed attempt that gets fed into subsequent iterations via `hint_`, whereas `BestOfN` runs completely independent trials with no adaptive behavior.
  > `dspy/predict/refine.py:167`: `advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice`
- **You believe that `dspy.ChainOfThought` updates the underlying `Predict` module's configuration directly in place to accommodate the reasoning field.** `ChainOfThought` creates a new instance of `dspy.Predict` with the extended signature instead of modifying the existing object.
  > `dspy/predict/chain_of_thought.py:35`: `self.predict = dspy.Predict(extended_signature, **config)`
- **You believe that tools must be explicitly instantiated as `dspy.Tool` objects before usage, ignoring that the implementation supports both plain callables and existing `Tool` instances.** The `tools` parameter requires a list where items can be plain callables or existing `Tool` instances; non-`Tool` items are automatically converted during initialization.
  > `dspy/predict/react.py:44`: `tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]`
- **You believe that an LM instance or a retriever/embedding module is required when passing parameters to the dspy.KNN constructor alongside k and trainset.** You must pass a specific `vectorizer` parameter instead of an LM or generic retriever, and this `vectorizer` argument expects an `Embedder` type.
  > `dspy/predict/knn.py:15`: `vectorizer: The `Embedder` to use for vectorization`

## dspy/primitives
- **DSPy does not infer module input fields solely from Module signatures; Example objects require .with_inputs() to specify inference fields.** All other fields become labels/metadata via example.labels(); pass only declared fields to .with_inputs().
  > `dspy/primitives/example.py:28`: `.with_inputs("question")`
- **CodeInterpreterError is defined in python_interpreter.py and FinalOutput in module.py.** Both CodeInterpreterError and FinalOutput are defined in code_interpreter.py, with public exports originating from __init__.py.
  > `dspy/primitives/__init__.py:2`: `from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput`
- **Comparing Prediction objects without a valid 'score' field evaluates to False or remains invalid.** If neither Prediction has a 'score' field, comparisons raise ValueError because operators rely on __float__() which requires the score field.
  > `dspy/primitives/prediction.py:55`: `raise ValueError("Prediction object does not have a 'score' field to convert to float.")`
- **Restrictions exist to prevent bypassing signature validation and internal state management during routing.** Restrictions ensure LM usage tracking functions correctly by capturing token usage within the `__call__` method.
  > `dspy/primitives/module.py:106`: `self._set_lm_usage(tokens, output)`

## dspy/utils
- **You believe there is no documentation about a URL-based file download utility in dspy.utils.** The utility function is `download` located in `dspy/utils/__init__.py`. When a file already exists at a different size than expected, the function re-downloads the file because the local size does not match the remote Content-Length header.
  > `dspy/utils/__init__.py:19`: `if not os.path.exists(filename) or local_size != remote_size:`
- **You believe the exception message contains no model identifier prefix.** When raising a ContextWindowExceededError with a model identifier like 'gpt-4', the message string will begin with the model identifier enclosed in square brackets followed by a space (e.g., `[gpt-4] `).
  > `dspy/utils/exceptions.py:21`: `prefix = f"[{model}] " if model else ""`

## dspy/dsp
- **The `ColBERTv2` parameter for enabling POST requests is `use_post`.** Set `post_requests=True` to initialize with `post_requests`; this triggers `colbertv2_post_request()`.
  > `dspy/dsp/colbertv2.py:18`: `post_requests: bool = False,`
- **Configs applied on the main thread propagate to worker threads.** Use `dspy.context(**kwargs)` for thread-local overrides in child tasks.
  > `dspy/dsp/utils/settings.py:161`: `raise RuntimeError(
    "dspy.configure(...) can only be called from the same async task that called it first. Please "
    "use `dspy.context(...)` in other async tasks instead."
)`
- **Omitting `k` utilizes a configured maximum limit or `None`.** `k` defaults to 7 in the signature, restricting returns to 7 passages.
  > `dspy/dsp/colbertv2.py:165`: `def forward(self, query: str, k: int = 7, **kwargs):`

## dspy/datasets
- **CSV loading specifications lack documentation on column key definitions.** Use DataLoader's `from_csv` method with `input_keys=('label',)` to designate the 'label' column.
  > `dspy/datasets/dataloader.py:67`: `input_keys: tuple[str] = (),`
- **Believed `Dataset.train` re-evaluated data on every access without caching.** `Dataset.train` caches results in `self._train_` after first access, returning cached values until `reset_seeds()` is called.
  > `dspy/datasets/dataset.py:60`: `if not hasattr(self, "_train_"):`
- **No documentation found for `Colors` dataset implementation or sorting logic.** Suffix-based grouping is handled by the `Colors` dataset's `sorted_by_suffix` method.
  > `dspy/datasets/colors.py:165`: `def sorted_by_suffix(self, colors):`

## dspy/signatures
- **Negative insertion indices raise `IndexError` on empty lists, assuming standard list behavior.** DSPy normalizes negative indices via `index += len(lst) + 1`, allowing `insert(-1)` to map to index 0.
  > `dspy/signatures/signature.py:460`: `# We support negative insert indices
        if index < 0:
            index += len(lst) + 1`
- **Signature fields lack type annotations and require explicit kwargs for all fields in Predict calls.** Define typed fields (e.g., `input: str = InputField(...)`). Pass input kwargs only upon init; predict outputs via `.predict()`. Instructions live in docstrings.
  > `dspy/signatures/signature.py:3`: `class MySignature(dspy.Signature):
    input: str = InputField(desc="...")
    output: int = OutputField(desc="...")`

## dspy/retrievers
- **Assumed 'Embeddings' class uses FAISS when candidates come from trainset.** FAISS index built during init only if corpus length >= brute_force_threshold (default 20k); otherwise self.index is None and brute-force search applies.
  > `dspy/retrievers/embeddings.py:38`: `self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None`
- **Believed 'Embeddings' retriever accepts 'cache' argument like other DSPy components.** Passing 'cache=True' raises AssertionError immediately; caching unsupported for embeddings-based retrievers.
  > `dspy/retrievers/embeddings.py:28`: `"Caching is not supported for embeddings-based retrievers"`
- **Uncertain if base Retrieve.forward() supports multiple queries unlike WeaviateRM.** Retrieve.forward() accepts single query; WeaviateRM.forward() processes batches via str | list[str].
  > `dspy/retrievers/weaviate_rm.py:73`: `def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs)`

## dspy/streaming
- **You believe the optional argument for custom status messages is `message_func`.** The correct argument for streaming execution is `status_message_provider`, accepting a `StatusMessageProvider` or `None`.
  > `dspy/streaming/streamify.py:29`: `status_message_provider: StatusMessageProvider | None = None,`
- **You believe `streaming_response` formats output by appending `data:` prefixes to raw text chunks without serialization.** The utility serializes each chunk as JSON using `orjson.dumps()` before wrapping in the SSE delimiter.
  > `dspy/streaming/streamify.py:273`: `yield f"data: {orjson.dumps(data).decode()}\n\n"`

## dspy/evaluate
- **You believed the `metrics` (plural) parameter overrides constructor metrics in `Evaluate.__call__`.** Use `metric` (singular) in `Evaluate.__call__` to override the constructor metric; it is assigned before falling back to `self.metric`.
  > `dspy/evaluate/evaluate.py:120`: `metric: Callable | None = None,`
- **You believed passing `return_outputs` to `Evaluate` constructor raises a `TypeError`.** It raises a `ValueError` stating '`return_outputs` is no longer supported' because DSPy explicitly checks for this deprecated keyword.
  > `dspy/evaluate/evaluate.py:114`: `raise ValueError("`return_outputs` is no longer supported. Results are always returned inside the `results` field of the `EvaluationResult` object.")`
- **You believed no explicit function removes English articles during text normalization.** The `remove_articles` helper within `normalize_text` in `dspy/evaluate/metrics.py` handles article removal.
  > `dspy/evaluate/metrics.py:110`: `def remove_articles(text):`

## dspy/propose
- **You believe that initializing GroundedProposer with program_aware=True causes automatic disablement due to various configuration issues like missing retrievers or signatures.** Actually, the proposer disables itself without raising an exception specifically when retrieving the program's source code fails, triggering an exception in get_dspy_source_code which sets self.program_aware to False.
  > `dspy/propose/grounded_proposer.py:291`: `self.program_aware = False`
- **You believe that the provided study notes contain no information about the create_dataset_summary function or its behavior with multiple consecutive 'COMPLETE' responses.** The function modifies processing by skipping data accumulation for affected batches and exits the main iteration loop immediately once a threshold of 5 consecutive 'COMPLETE' responses is reached.
  > `dspy/propose/dataset_summary_generator.py:75`: `if skips >= 5:`
- **You believe that the value of use_tip during execution is determined by retrieval operation success via is_grounded_by_retrieval() or has_results().** Actually, use_tip is determined solely by the truthiness of the selected tip string retrieved from the TIPS dictionary, independent of retrieval status; self.use_tip is set to bool(selected_tip), meaning empty strings yield False while non-empty text yields True.
  > `dspy/propose/grounded_proposer.py:346`: `self.use_tip = bool(`

## dspy
- **Global configuration requires `dspy.settings.configure` or explicit module assignment, not direct LM passing to `dspy.configure`.** Instantiate LM via `dspy.LM()` and pass it as the `lm` argument to `dspy.configure`.
  > `dspy/predict/predict.py:151`: ``dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))``
- **Passing positional arguments to `dspy.Predict` yields a generic error instead of a specific `ValueError`.** Calling `dspy.Predict` with positional arguments raises a `ValueError` demanding keyword arguments.
  > `dspy/predict/predict.py:121`: `Positional arguments are not allowed when calling `dspy.Predict`, must use keyword arguments`

## dspy/experimental
- **Believed `Document` implementation location unknown in `dspy.experimental`.** `Document` implemented at `dspy/adapters/types/document.py`, re-exported in `dspy/experimental/__init__.py`.
  > `dspy/experimental/__init__.py:2`: `from dspy.adapters.types.document import Document`
- **Unsure of `__all__` contents in `dspy/experimental/__init__.py`.** `__all__` lists `"Citations"` and `"Document"`.
  > `dspy/experimental/__init__.py:5`: `"Citations",`
