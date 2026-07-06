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

## dspy/adapters
- **You believe the parameter required to initialize a TwoStepAdapter that must be an instance of BaseLM is named `model`.** The correct parameter name is `extraction_model`.
  > `dspy/adapters/two_step_adapter.py:42`: `def __init__(self, extraction_model: BaseLM, **kwargs):`
- **You believe the regular expression pattern used is `r'\[\[([^\]]+)\]\]'` which captures any content inside double brackets.** The actual pattern requires the `##` delimiters and specifically captures word characters using `(\w+)`.
  > `dspy/adapters/chat_adapter.py:20`: `field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")`

## dspy/clients
- **you believe that setting the cache parameter to `None` or using varying parameter names across frameworks like LangChain is sufficient to disable response caching.** in dspy, you must explicitly set the `cache` parameter to `False` when creating the LM instance.
  > `dspy/clients/base_lm.py:63`: `def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):`
- **You believe that passing a `rollout_id` alongside `temperature=0` allows DSPy to track the inference request with that identifier for caching or experiment association despite the deterministic sampling strategy.** DSPy logs a warning that the parameter is ineffective and explicitly removes `rollout_id` from the arguments before forwarding the request.
  > `dspy/clients/lm.py:142`: `rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.`
- **You believe that the LM.cache system utilizes a spillover policy where entries are initially stored in memory and subsequently moved to the disk cache if the memory usage reaches a threshold.** In reality, the system stores the response in both the memory cache and the disk cache independently during the `cache.put()` operation. There is no automatic migration of new entries from memory to disk based on capacity; disk write failures are logged but do not block memory storage. Additionally, disk cache hits are promoted to the memory cache upon retrieval.
  > `dspy/clients/cache.py:157`: `self.disk_cache[key] = value`

## dspy/predict
- **You believe that `dspy.Refine` final prediction selection differs from `dspy.BestOfN` only through sequential input passing versus parallel generation, without recognizing the critical feedback generation mechanism that defines `Refine`'s adaptive loop.** `dspy.Refine` and `dspy.BestOfN` share identical initial reward-based selection and threshold exit logic—but `Refine` uniquely implements **feedback generation** (`advice`) after each failed attempt that gets fed into subsequent iterations via `hint_`, whereas `BestOfN` runs completely independent trials with no adaptive behavior.
  > `dspy/predict/refine.py:167`: `advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice`
- **You believe that `dspy.ChainOfThought` updates the underlying `Predict` module's configuration directly in place to accommodate the reasoning field.** `ChainOfThought` creates a new instance of `dspy.Predict` with the extended signature instead of modifying the existing object.
  > `dspy/predict/chain_of_thought.py:35`: `self.predict = dspy.Predict(extended_signature, **config)`
- **You believe that tools only need to be organized in a `list` or `dict` structure without necessarily being instantiated as `dspy.Tool` objects.** All tools must be wrapped in `dspy.Tool` instances before use.
  > `dspy/predict/react.py:44`: `tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]`
