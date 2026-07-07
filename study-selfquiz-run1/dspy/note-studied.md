# DSPy — studied reference (grounded in prior study)

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
This summary outlines critical behaviors for optimizers within the `dspy/teleprompt` module. When initializing `BetterTogether`, keyword argument names (e.g., `gepa`, `ft`) map directly to strategy string identifiers separated by `STRAT_SEP` (default `" -> "`). For instance, `BetterTogether(metric=metric, gepa=GEPA(...), ft=BootstrapFinetune(...))` requires `strategy="ft -> gepa"` to execute the fine-tuning strategy after GEPA. 

`BootstrapFinetune` enforces that student predictors must have assigned Language Models (LMs). Attempting to compile a student module with an unassigned predictor raises a `ValueError` with the fragment: `"Predictor {pred_ind} does not have an LM assigned. Please ensure the module's predictors have their LM set before fine-tuning."`. This is resolved by calling `your_module.set_lm(your_lm)`. The `multitask` flag dictates job launch logic based on training keys `(pred.lm, data_pred_ind)`, where `data_pred_ind` is `None` if `self.multitask` else `pred_ind`. With 3 predictors sharing the same LM: `multitask=True` launches **1** job (deduplication via `(same_lm, None)` key), while `multitask=False` launches **3** jobs (unique indices). If `multitask=False`, the job count equals the predictor count.

Before bootstrapping, `BootstrapFewShot` validates the student state at line 102 in `_prepare_student_and_teacher()`, asserting `getattr(self.student, "_compiled", False) is False`. This ensures the student program remains uncompiled to prevent cached results or optimized prompts from contaminating fresh inference traces gathered for few-shot learning enhancements. Finally, `MIPROv2` prohibits overriding automatic hyperparameter selection. Specifying `auto='medium'` alongside `num_trials=10` raises a `ValueError`: `"If auto is not None, num_candidates and num_trials cannot be set, since they would be overridden by the auto settings."` Users must set `auto=None` to manually specify trial counts.

## dspy/adapters
The DSPy adapters module standardizes interaction patterns between application signatures and Language Models (LMs) via JSONAdapter, ChatAdapter, and TwoStepAdapter classes. Data preprocessing occurs in `Adapter.base.format_demos()`. Here, a demo is considered `complete` strictly if `all(k in demo and demo[k] is not None for k in signature.fields)`. Demos marked `incomplete` require retention only if they satisfy `has_input = any(k in demo for k in signature.input_fields)` and `has_output = any(k in demo for k in signature.output_fields)`. Any demo missing both input or output categories is excluded entirely.

Parsing specifics reside in `ChatAdapter.parse()`. It utilizes the regex `field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")` located at line 20 to extract field markers from LM responses (line 215). Upon parsing, `parse()` validates the resulting dictionary keys against `signature.output_fields.keys()`. A mismatch triggers an `AdapterParseError` (raised at line 239), imported from `dspy.utils.exceptions` alongside `ContextWindowExceededError`.

`JSONAdapter` manages response formatting via `_json_adapter_call_common`. At line 52, if `not lm.supports_response_schema` is True (or `_has_open_ended_mapping` is active), it forces `lm_kwargs['response_format']` to `{'type': 'json_object'}` (line 55). This bypasses native function calling attempts where supported. Additionally, `JSONAdapter` falls back to `{'type': 'json_object'}` as a recovery mechanism when structured output attempts fail (lines 76-78).

`TwoStepAdapter` mandates an `extraction_model` instance of `BaseLM` in its constructor (line 42). Type checking ensures compatibility: `if not isinstance(extraction_model, BaseLM):` (line 44), followed by raising `ValueError("extraction_model must be an instance of dspy.BaseLM")` (line 45). All adapter operations rely on correct signature alignment to prevent runtime parsing exceptions. Developers must ensure output field names match precisely to avoid immediate `AdapterParseError` exceptions during execution.

## dspy/clients
**DSPY Clients Module Reference**

**Initialization & Caching**
Instances are created using `dspy.LM(model="...", model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs)`. The `cache` parameter defaults to `True`. Developers must set `cache=False` to disable caching explicitly, which ensures fresh API responses during debugging and prevents cached cost reporting (cache hits return `None`). The underlying cache system imports `LRUCache` from `cachetools` for memory management and `FanoutCache` from `diskcache` for persistence. Storage occurs independently: if disk write fails, it logs an error but preserves the in-memory entry. Retrieval prioritizes the memory cache layer before checking disk.

**OpenAI API Transformation**
When converting standard chat requests to the 'responses' API format within `dspy/clients/lm.py`, specific JSON structures transform. For image items matching `{"type": "image_url"}`, the nested `image_url.url` value extracts to become a direct string under the new key `type: "input_image"` (lines 565-567). Text nodes transition the `type` identifier from `"text"` to `"input_text"`. This mapping ensures compatibility with newer endpoint expectations.

**Model Capability Flags**
The `BaseLM` class (`dspy/clients/base_lm.py`) defines boolean properties for feature support: `supports_function_calling` (line 70), `supports_reasoning` (line 76), and `supports_response_schema` (line 81). These initialize to `False` by default. Provider-specific subclasses (e.g., OpenAI, Anthropic) override these definitions to detect capabilities based on configured model names rather than relying on external library delegation.

**Rollout ID Handling**
During generation, the `forward` method validates the `rollout_id` parameter. If `temperature` equals `0`, the system triggers a specific warning logic at line 142: "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache." Simultaneously, `_warn_zero_temp_rollout` executes, and line 175 removes `rollout_id` from `kwargs` entirely. Deterministic generation at zero temperature renders the `rollout_id` ineffective regarding cache keys; developers must utilize non-zero temperatures for seeding distinct responses.

## dspy/predict
The `dspy/predict` module defines core inference components requiring precise initialization parameters. For `dspy.KNN`, construction mandates `k` (int), `trainset` (list[Example]), and `vectorizer` (an `Embedder` instance). The `vectorizer` parameter accepts string inputs and returns numpy arrays; it is stored in `self.embedding` to facilitate nearest neighbor search. This is defined in the Args section (lines 12-15) alongside the `__init__` signature (line 8) in `knn.py`.

Calling `dspy.Predict` modules restricts argument passing to keywords. Providing positional arguments like `predict("val")` raises a `ValueError` specifying that inputs must match signature fields via keywords. This validation occurs at line 127 in `predict.py`.

Differentiation exists between optimization loops: `dspy.BestOfN` tracks rewards across N runs and breaks on threshold meeting, selecting the maximum reward statically (lines 74-77 in `best_of_n.py`). Conversely, `dspy.Refine` incorporates a feedback mechanism. After attempting N times with varying rollout IDs, `Refine` generates feedback for subsequent attempts, allowing dynamic adjustment compared to `BestOfN`'s static selection. Line 137 in `refine.py` calculates reward for potential feedback integration.

`dspy.ChainOfThought` modifies a parent signature by appending a `reasoning` field to the front. It executes `signature.prepend(name="reasoning", field=...)` (line 34) in `chain_of_thought.py`. The field description defaults to `"${reasoning}"` (line 31). This creates a child `Predict` module where the model outputs reasoning text before the target prediction, controlled by `rationale_field_type` (default `str`).

For `dspy.ReAct`, two minimum parameters are needed: `signature` (type definition) and `tools` (list of Callables). `max_iters` defaults to 20. Tools must be instances of `dspy.Tool`; if provided as callables, they are automatically wrapped during initialization (line 44). Finally, the processed tools are converted into a dictionary keyed by tool names (line 45), enabling structured access during execution.

## dspy/primitives
The `dspy.primitives` module establishes foundational data structures and runtime behaviors for modules. The `Prediction` class extends `Example` (`class Prediction(Example)`) to handle LLM outputs. During `Prediction.__init__`, inherited attributes `_demos` and `_input_keys` must be removed. Specifically, `del self._demos` and `del self._input_keys` occur after `super().__init__()` (referencing lines 108-109 of `example.py`). Attempting deletion before the parent call raises an `AttributeError` as these fields do not exist yet. For standard `Example` objects, separate inputs from labels using `.with_inputs("question")`. Subsequent inputs marked this way pass to the module; others act as metadata, accessible via `example.labels()`. Retrieving inputs requires `example.inputs().toDict()`.

When creating `Example` objects, ensure `.with_inputs()` is called immediately after initialization to tag fields correctly. Arguments passed via `with_inputs([...])` dictate the input signature for downstream modules. If omitted, all fields default to label or metadata status depending on evaluation context.

Module execution mandates `module(args)` over `module.forward(args)`. Calling `forward` directly logs a warning: "Calling module.forward(...) on [ClassName] directly is discouraged..." (line 344). This restriction preserves the `caller_modules` context managed by `settings.context()` within `__call__` (lines 99-101). Furthermore, `__call__` tracks LM token usage via `usage_tracker` when enabled (lines 102-106). The logic verifying direct calls checks the execution stack at line 341. The `Module` class implements custom `__getattribute__` logic (lines 335-347) to detect direct forwarding calls. This mechanism prevents breaking usage tracking infrastructure. By enforcing `__call__` wrappers, developers guarantee that usage metrics, including token consumption, are recorded on prediction objects for analysis.

Predictions compare numerically using comparison operators (`<`, `>`, etc.). These map to `__float__()`, which seeks a `'score'` value in the `_store` dictionary. If the `'score'` key is missing, `__float__` raises a `ValueError`: "Prediction object does not have a 'score' field to convert to float." (lines 54-55).

Artifact definitions include `CodeInterpreterError` and `FinalOutput`, located in `dspy/primitives/code_interpreter.py` (lines 16, 39). Public availability is handled by `dspy/primitives/__init__.py` (lines 2, 14-15), which re-exports these classes for direct consumer import. Developers must verify these signatures against the cited evidence to ensure accurate implementation without importing private internals.

## dspy/utils
The `dspy.utils` module offers essential helper functions for file management, synchronization, annotations, and concurrency control within DSPy pipelines. Specifically, the `download(url)` function found in `dspy/utils/__init__.py` facilitates retrieving remote files locally. It utilizes `requests.head` to obtain the `Content-Length` header and compares the `remote_size` against the `local_size` of an existing file. If the file is missing or the sizes differ, the function triggers a re-download to prevent stale data usage (lines 14-17).

For integrating async capabilities, the `syncify()` utility in `dspy/utils/syncify.py` converts async modules to sync versions. Setting `in_place=True` allows modification of the existing object, replacing its `forward` method with a new wrapper that simply invokes `self.aforward()`. Note that this operation overwrites any prior custom `forward` logic. If the module contained unique processing steps originally implemented in `forward`, they will be discarded, potentially altering expected workflow behaviors (lines 31-48).

Documentation warnings are managed via the `@experimental` decorator in `dspy/utils/annotation.py`. This decorator injects a notice text into the function's docstring. For functions with an existing docstring, the new text is prepended using concatenation (`notice + "\n\n" + api.__doc__`). Without an existing docstring, the notice stands alone (lines 63-65).

Exception handling employs `ContextWindowExceededError` from `dspy/utils/exceptions.py`. When instantiated with a specific model identifier (e.g., `gpt-4`), the exception message includes a bracketed prefix `[model]`. This prefixes the detailed error message, aiding in traceability during runtime failures (lines 21-22).

Thread safety is addressed in `ParallelExecutor` within `dspy/utils/parallelizer.py`. When executing worker threads, it isolates thread-local settings such as `usage_tracker`. It copies parent overrides and performs `copy.deepcopy()` on tracker objects to ensure no cross-contamination between threads. Once execution finishes, the thread-local state is reset (lines 123-126).

These utilities collectively ensure robust state management, error visibility, and functional correctness across distributed execution environments in DSPy applications.

## dspy/dsp
The `dspy/dsp` module encapsulates critical utilities for configuration management, type extensions, and search retrieval logic within the broader library ecosystem. Specifically, the `dotdict` class facilitates flexible attribute-style access, allowing expressions like `obj.a` to evaluate to values (e.g., `1`) stored in a dictionary under keys (e.g., `{'a': 1}`). This functionality is enabled by overriding `__getattr__(self, key)` to convert attribute reads into dictionary lookups via `return self[key]`. In contrast, standard Python dictionaries raise an `AttributeError` with the message `'dict' object has no attribute 'a'` if accessed this way. Configuration management imposes strict threading rules to preserve state integrity; `dspy.settings.configure()` verifies `config_owner_thread_id` and raises a `RuntimeError` if invoked from a thread different from the one that initially configured it. Developers attempting to alter settings in worker threads must utilize `dspy.context(**kwargs)` instead, which creates thread-local overrides propagated to child thread contexts effectively. For vector search operations, the `ColBERTv2` class initializes with `post_requests: bool = False`. Enabling `post_requests=True` switches internal execution routing from `colbertv2_get_request` to `colbertv2_post_request`, enabling remote server interaction via POST requests. Document localization in `dpr.py`'s `locate_answers()` requires text processing of answers and body via two mandatory steps: Unicode Normalization Form D (NFD) decomposition via `DPR_tokenize(text)` to resolve character representation differences consistently, and lowercasing via `.words(uncased=True)` for case-insensitive matching. Finally, `ColBERTv2RetrieverLocal.forward(query, k: int = 7, **kwargs)` defaults to retrieving seven passages based on FAISS vector search. While passing `k=0` returns an empty result set gracefully, initializing with an empty passage list causes errors during object construction. Adherence to these concrete signatures ensures stable performance and prevents unexpected runtime failures during pipeline execution.

## dspy/datasets
This module defines dataset utilities within the `dspy/datasets` package, managing loading, shuffling, and construction logic. The primary entry point for custom data ingestion is the `DataLoader` class. Specifically, the `from_csv` method (found in `dspy/datasets/dataloader.py`) initializes datasets from local CSV files. Developers must pass `file_path` to specify the location and utilize `input_keys` to identify feature columns. For example, to treat a column named 'label' as the input key, the signature requires `input_keys=('label',)`. Usage takes the form: `dataloader.from_csv(file_path='path/to/data.csv', input_keys=('label',))`. Note that default `input_keys` is an empty tuple defined in `dataloader.py` line 67.

Pre-configured datasets require specific attention to their constructors. The `HotPotQA` dataset instance restricts flexibility via an assertion found in `hotpotqa.py`; passing `only_hard_examples=False` results in an `AssertionError` (lines 16-17), preventing instantiation for non-hard variants. Inside `prepare_by_seed` functionality within `dataset.py`, evaluation data partitioning depends on the `divide_eval_per_seed` boolean flag. When enabled, individual evaluation subset lengths equal `dev_size // len(train_seeds)`, utilizing integer division to distribute data across seeds equally. Code at line 123 calculates `examples_per_seed`, followed by an assertion at line 133 verifying uniform subset sizes (`assert len(eval_sets[-1]) == examples_per_seed`).

The core `Dataset` class properties manage state dynamically. Accessing the `.train` property triggers lazy evaluation. It checks for the existence of a private cache attribute, `self._train_` (lines 59-61). If absent, it invokes `_shuffle_and_sample` using `self._train` and configured parameters to populate the cache. Subsequent access retrieves this cached list without re-computation. State resets occur via `reset_seeds()`, which deletes the `_train_` attribute.

Specialized datasets like `Colors` implement sorting logic prior to train/dev splitting. The `sorted_by_suffix` method orders items by reversing strings (key=lambda x: x[::-1]), ensuring color names with shared suffixes group together. This reordered list replaces `all_colors` (line 152) before the split occurs, optimizing the distribution of structurally similar data (defined in `colors.py`). All provided information derives directly from cited source lines within `dspy/datasets` modules.

## dspy/signatures
The `dspy.Signatures` module establishes structured interfaces for language model interactions. To programmatically verify if two `Signature` instances share identical structures, utilize the classmethod `Signature.equals(cls, other)`. This method returns a boolean confirming four specific properties: (1) both classes are subclasses of Pydantic's `BaseModel`, (2) the `instructions` attribute matches exactly, (3) all field names possess symmetric membership across both signatures, and (4) the `json_schema_extra` dictionary remains identical for every corresponding field. Note this compares definition structure rather than instantiated runtime values.

Dynamic schema modification relies on `MySig.insert(index, 'new_field', InputField())`. Indices support normalization where negative values adjust automatically; specifically, `-1` on an empty list inserts at index 0. Validation occurs before insertion, accepting indices within ranges derived from `len(lst)`. For field definitions, avoid deprecated arguments such as `prefix=` or `format=` within `InputField()`. Usage generates a `DeprecationWarning` because these parameters function nowhere and are scheduled for removal. Adhering to current standards prevents future compatibility breaks.

Implementing new schemas follows a subclassing pattern: `class MySignature(dspy.Signature):`. Place detailed instructions in the class docstring. Declare fields using `dspy.InputField(desc="description")` or `dspy.OutputField(desc="description")`. Instantiation requires keyword arguments aligned with field names (e.g., `MySignature(input="value1", another_output=42)`). Execution invokes the method directly or via `.predict()` on the instance. Batch operations can generate multiple signatures via list comprehensions: `[MySignature(input=i, another_output=i*2).predict() for i in data]`. Finally, when constructing string-based signatures (e.g., "question, context -> answer"), enforce distinct naming between input and output sets. The internal `_parse_signature` function validates this; conflicting names raise a `ValueError` stating "Input and output fields must have distinct names, but found duplicates." This ensures clear separation of model inputs and expected outputs throughout the pipeline execution.

## dspy/retrievers
The `dspy` retrieval module distinguishes retrieval logic through the `Retrieve` base class and specialized subclasses like `Embeddings` and `WeaviateRM`. Correct configuration relies on understanding initialization constraints and method signatures.

The `Embeddings` class uses corpus volume to switch search strategies. Upon instantiation, the system evaluates `if len(corpus) >= brute_force_threshold` (default `brute_force_threshold`: 20_000). If true, `self._build_faiss()` creates an index (`self.index`). Otherwise, `self.index` is `None`, using brute-force logic. If FAISS is required but `faiss-cpu` is missing, an `ImportError` states: "Please `pip install faiss-cpu` or increase `brute_force_threshold` to avoid FAISS." No automatic fallback occurs if the library is absent. Caching is also strictly prohibited; passing `cache=True` triggers `AssertionError: "Caching is not supported for embeddings-based retrievers"`.

Operationally, the base `Retrieve` class requires `dspy.settings.rm` to be set. Missing this raises `AssertionError("No RM is loaded.")`. Its `forward` signature enforces single-query processing: `def forward(self, query: str, k: int | None = None, **kwargs)`.

Conversely, `WeaviateRM` supports batched inputs via `forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs)`. It normalizes input by wrapping strings in lists if necessary, then iterating through all queries to accumulate results. This contrasts with the base class by accepting multiple queries per invocation rather than requiring sequential object calls.

Adhering to these initialization checks and interface distinctions ensures stable operation across retrieval backends.

## dspy/streaming
To enable incremental output streaming within a DSPy program, developers must wrap the program using the `streamify` function. This wrapper accepts an optional `status_message_provider` argument (of type `StatusMessageProvider | None`) to supply custom status messages for execution progress instead of the default. Regarding the final result, the parameter `include_final_prediction_in_output_stream` defaults to `True`. However, when set to `False`, the final `Prediction` object is excluded from the stream exclusively under three conditions: `stream_listeners` is provided, no cache hits occurred, and at least one listener successfully captured streaming content (specifically `stream_start` events).

Handling asynchronous execution requires configuration regarding `async_streaming`. If set to `False`, the underlying asynchronous generator is converted for synchronous consumption. This conversion employs a producer-consumer threading pattern where a daemon thread executes the generator via `asyncio.run(runner())`. Values flow through a `queue.Queue` (line 229), signaling completion with a `stop_sentinel` object (lines 230, 244). Context variables are propagated using `contextvars.copy_context()` (line 233), resulting in a returned standard `Generator` object suitable for blocking threads.

For API compatibility, the `streaming_response` utility formats output chunks according to the Server-Sent Events specification. Each chunk is serialized as JSON using `orjson.dumps()` and wrapped in the delimiter `data: <payload>\n\n`. The stream concludes with a terminator message `data: [DONE]\n\n`. Developers must exercise caution when defining `StreamListener` instances; specifically, the `signature_field_name` must not appear in the output fields of multiple distinct predictors within the same program. If duplicates exist, initialization raises a `ValueError` stating the field is not unique. In such cases, users must either ensure unique field names across predictors or explicitly specify the predictor to listen to.

## dspy/evaluate
The `Evaluate` class in the `dspy` library is central to program verification. Developers must adhere to strict initialization rules. Passing the `return_outputs` keyword argument to the `Evaluate` constructor triggers a `ValueError`. The error message confirms that results are always contained within the `results` field of the `EvaluationResult` object, rendering the `return_outputs` feature unsupported.

Metric flexibility is achieved through the `__call__` method. While instantiation stores a metric, the `metric` argument within `__call__` permits runtime overrides. Specifically, line 149 of `evaluate.py` dictates `metric = metric if metric is not None else self.metric`. This logic ensures that a function passed directly during invocation replaces the constructed metric, provided it is not `None`.

Exception handling relies on the `failure_score` parameter. Defined in `evaluate.py` line 81, `failure_score` defaults to 0.0. If a metric calculation throws an exception or returns `None`, the result list is processed using a list comprehension (line 178). Any `None` tuple entry is replaced by `self.failure_score`. Consequently, individual failures default to a float score of 0.0 in the final aggregation unless the `failure_score` is customized in the constructor.

Auto-evaluation capabilities include the `SemanticF1` module defined in `auto_evaluation.py`. Its `__init__` signature (line 50) accepts `threshold=0.66` and `decompositional=False`. The `threshold` parameter controls the minimum F1 score during optimization. To enforce stricter standards, developers must initialize this module with `threshold=0.85`. The `decompositional` flag configures the decomposition behavior of the evaluation process.

Underlying metric calculations involve text preprocessing from `metrics.py`. The `normalize_text` function orchestrates standardization, employing a nested helper named `remove_articles`. As cited in line 111, `remove_articles` utilizes a regular expression `r"\b(a|an|the)\b"` to substitute English articles with spaces. This removes grammatical noise before subsequent comparison operations, ensuring that token-level differences caused by articles do not skew metric outputs.

## dspy/propose
Developers should import `GroundedProposer` via `from dspy.propose import GroundedProposer`. Initialization with `program_aware=True` calls `get_dspy_source_code(program)`. If this call raises any exception, a try-except block catches the error (lines 289-291), sets `self.program_aware = False`, and continues execution safely without crashing.

Instruction history management relies on `create_predictor_level_history_string`, which caps stored previous instruction attempts at `MAX_INSTRUCT_IN_HISTORY = 5` (line 15). Applied at line 380 during string construction using `trial_logs`, this prevents context overload. Utilities ensure uniqueness and performance by sorting history by score and slicing the top `n` items via `sorted(unique_predictor_history, ..., reverse=True)[:top_n]` (utils.py line 119), prioritizing proven successes over noise.

Data processing in `create_dataset_summary` terminates early upon convergence detection. The function inspects `output["observations"]` length and content (line 73), specifically checking if the first 8 elements equal uppercase "COMPLETE". If so, it increments a `skips` counter. If 5 such consecutive events occur, the loop breaks immediately to avoid processing guaranteed-complete batches, optimizing resource usage.

Regarding guidance tuning, enabling `set_tip_randomly=True` utilizes a stochastic strategy. The proposer selects a random key from `TIPS` (line 344) and assigns `selected_tip = TIPS[selected_tip_key]`. The `use_tip` boolean is set via `bool(selected_tip)` (line 346). Thus, tips mapped to empty strings (e.g., "none") result in `use_tip=False`, effectively disabling the strategy, while non-empty strings enable it. This design allows configuration of off-states via specific keys without raising exceptions.

## dspy
Developers must adhere to specific configuration and invocation constraints when working with DSPy modules to ensure stability and compatibility across library versions.

Regarding `dspy.Predict` modules, default configurations such as `temperature=0.5` persist within the module instance. To override these parameters for a specific call without reconstructing the module instance, pass a `config` dictionary as a keyword argument during invocation. For example, use `config={"temperature": 0.8}` inside the call arguments. This merges with `self.config` at runtime, ensuring local behavior changes do not affect subsequent calls.

When defining `InputField` or `OutputField` descriptors, developers must avoid using the keywords `'prefix'`, `'format'`, or `'parser'`. Although these parameter names resemble common patterns, they are deprecated in DSPy. Utilizing them triggers visible deprecation warnings and produces no functional effect; furthermore, they are scheduled for removal in future versions. Ignoring this causes unnecessary log noise and introduces refactoring debt.

Global language model configuration requires instantiating the Language Model wrapper before assignment. Do not attempt to use `dspy.configure(lm="model_name")` directly with a string. The correct syntax mandates wrapping the model identifier: `dspy.configure(lm=dspy.LM("model_name"))`. Providing a raw string bypasses validation and fails during initialization.

For custom `Signature` subclasses, all instructional text must be attached via the class docstring. Specifically, the `instructions` property reads directly from `cls.__doc__` using `inspect.cleandoc`. If a docstring is absent, the metaclass automatically assigns default instructions derived from the schema's input and output fields. Developers can also set instructions after creation, which updates the underlying `cls.__doc__` attribute.

Finally, invoking any `dspy.Predict` instance strictly requires keyword arguments that map precisely to the signature's input fields. Any attempt to pass positional arguments raises a `ValueError` immediately. The error message explicitly states positional arguments are not allowed, enforcing strict type safety on module interaction patterns.

## dspy/experimental
The `dspy.experimental` module functions as the central gateway for accessing experimental classes within the DSPy ecosystem. Access to this module's functionality is gated exclusively through its public export list, defined in `dspy/experimental/__init__.py`. To utilize the available types, developers should employ the standard import statement: `from dspy.experimental import Citations` or `from dspy.experimental import Citations, Document`. The module's `__all__` list restricts discovery to precisely two entries: `"Citations"` and `"Document"`. Any attempt to access undefined names, such as `dspy.experimental.Model`, triggers an `AttributeError` stating "module 'dspy.experimental' has no attribute 'Model'", enforcing strict boundary conditions on namespace exposure.

While the entry point resides in `dspy/experimental/__init__.py`, the physical implementation lies deeper in the dependency tree. The `Document` type is implemented in `dspy/adapters/types/document.py` (line 10), decorated with `@experimental(version="3.0.4")`. The `Citations` class is similarly accessible via `dspy.experimental` but relies on internal imports found in `dspy/adapters/base.py` (line 11) and `dspy/adapters/types/citation.py` (lines 22 and 27). Within those citation files, `Dspy.experimental` is referenced as the authoritative source for both `Citations` and `Document`, suggesting a standardized re-export mechanism across the package. For implementation purposes, note that `Document` extends a base `Type` class, as evidenced by `class Document(Type):`.

When writing tests or documentation, you must account for the fact that `dspy.experimental` simply aggregates these external classes rather than defining them locally. Consequently, importing from `dspy.experimental` ensures compatibility with versions of the library that have been marked experimental, such as version "3.0.4" indicated on the `Document` class. In summary, maintain the usage of `dspy.experimental` as the single interface. Avoid direct imports from `dspy/adapters/types` unless bypassing the experimental wrapper intentionally, as that breaks the versioning convention associated with the `@experimental` decorator. Ensure that your dependency graph satisfies the resolution of `dspy.adapters.types.document` and `dspy.adapters.types.citation` when initializing instances of these types. Strict adherence to the `__all__` exports prevents silent failures, as the runtime will raise explicit errors for non-existent attributes on the experimental module object.

---

# Verified corrections (trust these over the summaries)

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
