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
- **You believe ChatAdapter.parse() raises a `KeyError` exception if the resulting dictionary keys do not match the signature's output field names exactly.** ChatAdapter.parse() actually raises an `AdapterParseError` exception when the set of parsed field keys does not equal the set of expected output field keys from the signature.
  > `dspy/adapters/chat_adapter.py:239`: `raise AdapterParseError(`

## dspy/clients
- **You believe there are multiple parameter options like **`enable_cache`**, **`use_response_cache`**, or **`cache_enabled`** that can be used to disable response caching when creating an LM instance.** You should set the **`cache`** parameter to **`False`** when creating an LM instance to disable response caching, not use any other parameter name like `enable_cache`, `use_response_cache`, or `cache_enabled`.
  > `dspy/clients/lm.py:39`: `cache: bool = True,`
- **You believe the logic for detecting whether a model supports function calling, reasoning, or response schema is defined within core model interface classes of typical orchestration frameworks like LangChain and delegated to provider-specific SDKs.** The inference code is actually defined within the `LM` class in `dspy/clients/lm.py` as property methods (`supports_function_calling`, `supports_reasoning`, `supports_response_schema`) that delegate specifically to the `litellm` library.
  > `dspy/clients/lm.py:124`: `return litellm.supports_function_calling(model=self.model)`
- **You believe the transfer of cached data from disk to memory is a secondary step occurring after the data is fully processed or loaded into the application context.** The system performs a direct addition (promotion) of disk-retrieved entries into the memory cache immediately upon detecting a disk hit, optimizing future latency before returning the response.
  > `dspy/clients/cache.py:117`: `# Found on disk but not in memory cache, add to memory cache`
- **You believe the specific line of code responsible for promoting a disk cache hit to the memory cache is located at line 117 of `cache.py`.** The retrieval logic evaluates the disk cache condition at line 116, and the subsequent assignment updating the memory cache occurs at line 121; there is no promotion logic explicitly isolated to line 117.
  > `dspy/clients/cache.py:121`: `self.memory_cache[key] = response`

## dspy/predict
- **you believe that passing positional arguments to a dspy.Predict module results in a TypeError** passing positional arguments raises a ValueError requiring keyword arguments matching signature input fields
  > `dspy/predict/predict.py:129`: `raise ValueError(self._get_positional_args_error_message())`
- **You believe tools simply need to be wrapped in a `list` data structure before use.** Every tool must be wrapped in a `dspy.Tool` instance before use, though the class may auto-convert regular callables.
  > `dspy/predict/react.py:44`: `tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]`
- **You believe dspy.ChainOfThought modifies the underlying Predict module by overriding or extending its prompt template.** It modifies the underlying Predict module by creating an extended_signature that includes the new field and passing that extended signature to the dspy.Predict constructor.
  > `dspy/predict/chain_of_thought.py:35`: `self.predict = dspy.Predict(extended_signature, **config)`
- **you believe `dspy.BestOfN` resorts to random selection if no explicit scoring mechanism is provided during its N iterations.** `dspy.BestOfN` strictly selects the candidate with the highest calculated numerical reward score among all independent samples; it does not default to random selection in the absence of other metrics.
  > `dspy/predict/best_of_n.py:74`: `if reward > best_reward:`

## dspy/primitives
- **you believe that if neither Prediction has a 'score' field, comparison operations (<, >, <=, >=) will typically return False rather than raising an exception.** Attempting to compare Prediction objects using these operators when they lack a 'score' field raises a ValueError with the message "Prediction object does not have a 'score' field to convert to float.".
  > `dspy/primitives/prediction.py:55`: `raise ValueError("Prediction object does not have a 'score' field to convert to float.")`
- **You believe calling `module.forward(args)` triggers a DeprecationWarning about deprecation, and that the restriction exists primarily for input validation and signature matching purposes as defined in predict.py. You also believe the main issue is Teleprompter and compiler compatibility requirements.** When calling `module.forward(args)` directly, a warning is logged saying "Calling module.forward(...) on {ModuleClassName} directly is discouraged. Please use module(...) instead." The restriction exists because the `__call__` method contains critical infrastructure including caller module context setup via settings.context(), usage tracking with track_usage(), history and callbacks through @with_callbacks decorator, and proper LM usage attribution. Directly calling `forward()` bypasses all this instrumentation, potentially breaking tracking, callbacks, history collection, and usage statistics.
  > `dspy/primitives/module.py:345`: `logger.warning(
    f"Calling module.forward(...) on {self.__class__.__name__} directly is discouraged. "
    f"Please use module(...) instead."
)`
- **You believe that `CodeInterpreterError` and `FinalOutput` are defined in `dspy/primitives/python_interpreter.py`, and you believe that `dspy/primitives/module.py` provides the public exports for CodeInterpreter functionality.** Both `CodeInterpreterError` and `FinalOutput` are actually defined in `dspy/primitives/code_interpreter.py`. The public exports for CodeInterpreter functionality are provided by `dspy/primitives/__init__.py`, not `module.py`.
  > `dspy/primitives/__init__.py:2`: `from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput`

## dspy/utils
- **You believe the utility function for downloading files from a URL within `dspy.utils` is named `download_file_from_url`.** The correct function available in `dspy/utils/__init__.py` is named `download()`.
  > `dspy/utils/__init__.py:14`: `def download(url):`
- **You believe that calling `syncify(in_place=True)` on an async DSPy module with a custom `forward` method causes synchronization issues like race conditions or async mismatches.** Actually, it results in the custom `forward` logic being silently overwritten by a generic wrapper that only calls `aforward`, causing all original business logic, validations, and side effects to be lost.
  > `dspy/utils/syncify.py:31`: `There are two modes of this function:

- `in_place=True` (recommended): Modify the module in place. But this may not work if you already have a `forward`
    method which does different things from `aforward`.`
- **you believe the prefix in the exception's string representation is "Context window"** the prefix consists of the model identifier enclosed in square brackets followed by a single space
  > `dspy/utils/exceptions.py:21`: `prefix = f"[{model}] " if model else ""`
- **You believe the provided study notes do not contain the specific information regarding how the download utility behaves when a pre-existing file has a different size than expected.** The function compares the remote Content-Length header with the local file size and re-downloads the entire file if the sizes do not match exactly.
  > `dspy/utils/__init__.py:19`: `if not os.path.exists(filename) or local_size != remote_size:`

## dspy/dsp
- **You believe that omitting the `k` keyword argument defaults to a system-defined context limit or a batch size like 100, rather than a fixed small integer.** The method signature explicitly sets the default value for `k` to 7, regardless of system configuration.
  > `dspy/dsp/colbertv2.py:165`: `def forward(self, query: str, k: int = 7, **kwargs):`
- **you believe the parameter name used to enable POST requests for `ColBERTv2` initialization is `use_post`** The correct parameter name is `post_requests`. When initialized with `post_requests=True`, the retrieval logic sets the flag to `True` in the instance, causing the `__call__` method to execute `colbertv2_post_request()` instead of `colbertv2_get_request()`
  > `dspy/dsp/colbertv2.py:18`: `post_requests: bool = False,`
- **You believe the text comparison function converts the strings to lowercase using Python's standard built-in `.lower()` method.** The code does not use the built-in `.lower()` method; instead, it passes the `uncased=True` flag to the tokenizer's `.words()` method to handle casing during the word extraction process.
  > `dspy/dsp/utils/dpr.py:216`: `text_words, text_word_positions = tokenized_text.words(uncased=True), tokenized_text.offsets()`
- **You believe that passing an empty list to the `forward()` method would raise a TypeError simply because `k` expects an integer rather than a list.** The actual behavior is that passing any passages parameter to `forward()` is impossible because the method does NOT accept a `passages` argument at all—passage data is initialized exclusively in `__init__` and stored as `self.passages`. Any attempt to include a passages parameter would cause a TypeError due to signature mismatch.
  > `dspy/dsp/colbertv2.py:94`: `def __init__(self, passages: list[str], colbert_config=None, load_only: bool = False):`

## dspy/datasets
- **You believe that the `load_dataset_from_csv()` method (or similar variants like `load_from_file`) is the correct approach to load a dataset from a local CSV file when using the `DataLoader` class.** You must call the `from_csv` method of the `DataLoader` class, ensuring you pass the file location as the `file_path` argument and specify the input columns as a tuple within the `input_keys` argument.
  > `dspy/datasets/dataloader.py:63`: `def from_csv(`
- **You believe passing `only_hard_examples=False` will raise a TypeError because the parameter has been removed or deprecated in the current DSPy version.** Passing `only_hard_examples=False` instead raises an AssertionError indicating that the development set must consist entirely of hard examples to match the official dataset, although the training set can be flexible.
  > `dspy/datasets/hotpotqa.py:16`: `assert only_hard_examples, (
            "Care must be taken when adding support for easy examples."
            "Dev must be all hard to match official dev, but training can be flexible."
        )`
- **you believe the sorting logic is applied within the `load` method (or `_load_data`)** the sorting logic is instead applied within the `sorted_by_suffix` method, which reverses strings to group colors by suffix
  > `dspy/datasets/colors.py:165`: `def sorted_by_suffix(self, colors):`

## dspy/signatures
- **you believe that calling `MySig.insert(-1, ...)` on an empty inputs list raises an `IndexError`.** actually, calling `MySig.insert(-1, ...)` on an empty inputs list converts the negative index to 0 using the formula `index += len(lst) + 1`, so the field is successfully inserted at position 0.
  > `dspy/signatures/signature.py:461`: `if index < 0:
            index += len(lst) + 1`

## dspy/retrievers
- **you believe that the provided note content lacks specific information regarding how the 'forward' method input handling differs between the base 'Retrieve' class and the 'WeaviateRM' subclass for multiple queries.** Actually, the WeaviateRM implementation accepts both `str` and `list[str]` via `query_or_queries: str | list[str]` with built-in batching logic, while the base Retrieve class only accepts a single `str`.
  > `dspy/retrievers/weaviate_rm.py:73`: `def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> Prediction:`
- **You believe that attempting to initialize the 'Embeddings' retriever with `cache=True` will result in a TypeError because the parameter is not accepted by the constructor.** Actually, the system accepts the `cache` argument but enforces an AssertionError during initialization, explicitly stating that caching is not supported for embeddings-based retrievers when any value other than False is passed.
  > `dspy/retrievers/embeddings.py:28`: `assert cache is False, "Caching is not supported for embeddings-based retrievers"`
- **you believe the provided study notes do not directly answer whether a specific error is raised when calling Retrieve.forward() without dspy.settings.rm.** An AssertionError with the message "No RM is loaded." is raised immediately in the source code if the retrieval model is not configured.
  > `dspy/retrievers/retrieve.py:54`: `raise AssertionError("No RM is loaded.")`
- **You believe that the `Embeddings` class switches from brute-force search to building a FAISS index when the number of items exceeds around 1000 documents.** The `Embeddings` class switches from brute-force search to building a FAISS index when the corpus length (`len(corpus)`) is greater than or equal to the `brute_force_threshold` parameter, which defaults to 20,000 items.
  > `dspy/retrievers/embeddings.py:25`: `brute_force_threshold: int = 20_000,`

## dspy/streaming
- **You believe the streaming_response utility formats all output chunks using a generic JSON payload structure within SSE event delimiters without distinguishing data types or including a specific termination signal.** Specifically, it wraps prediction chunks as `{"prediction": <dict>}`, model response chunks as `{"chunk": <json_string>}`, passes existing SSE formatted values through unchanged, and explicitly yields a `"data: [DONE]"` marker to signal stream completion.
  > `dspy/streaming/streamify.py:284`: `yield "data: [DONE]\n\n"`
- **you believe the optional argument to provide custom status messages when wrapping a DSPy program is `status_callback` or `status_message`.** you should pass the argument `status_message_provider` to configure custom status messages when wrapping a DSPy program with streaming.
  > `dspy/streaming/streamify.py:29`: `status_message_provider: StatusMessageProvider | None = None,`

## dspy/evaluate
- **you believe that -1.0 replaces the None tuple for an item's score in the final results.** 0.0 replaces the None tuple in the final results by default, controlled by the `failure_score` parameter.
  > `dspy/evaluate/evaluate.py:81`: `failure_score: float = 0.0`
- **You believe that there is no specific information detailing the error raised when passing `return_outputs` to the `Evaluate` constructor and that it would likely result in a generic `TypeError` for unrecognized keyword arguments.** A `ValueError` is raised when passing the `return_outputs` keyword argument to the `Evaluate` constructor with the message that the argument is no longer supported.
  > `dspy/evaluate/evaluate.py:114`: `"`return_outputs` is no longer supported. Results are always returned inside the `results` field of the `EvaluationResult` object."`
- **You believe there is no information specifying a function responsible for removing English articles during text normalization.** The normalize_text function in dspy/evaluate/metrics.py handles this using a nested helper function called remove_articles.
  > `dspy/evaluate/metrics.py:110`: `def remove_articles(text):`

## dspy/propose
- **you believe the use_tip flag is determined by evaluating criteria such as confidence scores, ground truth availability, or error thresholds before applying any proposed tip** the actual behavior is that self.use_tip = bool(selected_tip), meaning use_tip is True for all selected tips except "none" which has an empty string value and results in use_tip=False
  > `dspy/propose/grounded_proposer.py:346`: `self.use_tip = bool(
                        selected_tip,
                    )`
- **You believe there is no information about the `create_dataset_summary` function's handling of consecutive 'COMPLETE' responses from an observation model in the available documentation or study notes.** Actually, the function tracks consecutive 'COMPLETE' responses using a counter variable named `skips` that increments each time a response begins with 'COMPLETE'. When this counter reaches 5 or more consecutive occurrences, the function immediately terminates batch processing with a `break` statement to prevent excessive API calls.
  > `dspy/propose/dataset_summary_generator.py:75`: `if skips >= 5:
break`
- **you believe the automatic disabling is not explicitly documented and depends on factors like grounding limitations or missing evidence retrieval results** the proposer sets `self.program_aware = False` when `get_dspy_source_code(program)` fails or raises an exception during initialization
  > `dspy/propose/grounded_proposer.py:291`: `self.program_aware = False`

## dspy
- **you believe that providing 'prefix', 'format', or 'parser' arguments to InputField or OutputField effectively customizes field handling and response parsing logic.** in reality, these arguments are deprecated and have no functional effect, only triggering DeprecationWarning messages during object creation.
  > `dspy/signatures/field.py:13`: `The 'prefix' argument in InputField/OutputField is deprecated and has no effect in DSPy.`
- **You believe that instructions for a custom Signature subclass are attached by passing an `instructions` string argument to the superclass constructor or by setting an instance attribute like `_instructions`, and that the property reads this internal instance variable.** Instructions are attached by setting the class's `__doc__` attribute (either via a class docstring, `with_instructions()`, or the setter). The `instructions` property reads this value directly from `cls.__doc__` using `inspect.cleandoc()`.
  > `dspy/signatures/signature.py:216`: `return inspect.cleandoc(getattr(cls, "__doc__", ""))`

## dspy/experimental
- **you believe the `Document` type implementation is located in a core module such as `dspy/primitives/` or `dspy/propose/` and is merely re-exported from `dspy/experimental/` rather than being defined in the adapters types module.** the `Document` type is actually implemented directly in `dspy/adapters/types/document.py`, which is then re-exported from `dspy/experimental/`.
  > `dspy/experimental/__init__.py:2`: `from dspy.adapters.types.document import Document`
- **You believed the contents of the `__all__` list in `dspy.experimental` were uncertain, potentially empty, or required external inspection beyond the provided context.** The `__all__` list explicitly exports exactly two items: "Citations" and "Document".
  > `dspy/experimental/__init__.py:5`: `"Citations",`
