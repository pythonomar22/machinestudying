# DSPy Cheatsheet

## Quick Start

```python
import dspy

# Configure default language model
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Create your first predictor
predictor = dspy.Predict("question -> answer")
response = predictor(question="What is 2+2?")
print(response.answer)  # Output: "4"
```

## Core Concepts

### 1. Signatures & Fields

```python
# String-based signature
predictor = dspy.Predict("input_field -> output_field")

# Class-based signature
class MySignature(dspy.Signature):
    question: str = dspy.InputField(desc="Your question")
    answer: str = dspy.OutputField(desc="The answer")

# Custom field options
field = dspy.InputField(desc="Description here", prefix="Prefix:")
```

**Field Types:**
- `InputField` - Parameters to provide before calling
- `OutputField` - Values the model should generate
- Can use Python types: `int`, `str`, `list[str]`, `dict[str, Any]`

### 2. Language Models

```python
# Basic configuration
lm = dspy.LM("openai/gpt-4o-mini", temperature=0.7)

# Chat vs text completion
chat_lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
text_lm = dspy.LM("openai/text-davinci-003", model_type="text")

# Caching (automatic by default)
cache_lm = dspy.LM("openai/gpt-4o-mini", cache=True)

# Bypass cache for diverse sampling
sample_lm = lm.copy(rollout_id=rollout_id, temperature=1.0)
```

### 3. Examples & Data

```python
# Create an example
example = dspy.Example(
    question="What is Paris known for?",
    answer="Eiffel Tower"
).with_inputs("question")  # Mark which fields are inputs

# Access patterns
example.question  # Dot notation
example["question"]  # Dict-style

# Inputs/Labels separation
inputs = example.inputs()  # Fields marked as inputs
labels = example.labels()  # Remaining fields
```

### 4. Adapters (Input/Output Formatting)

```python
# Set adapter for output formatting
dspy.configure(adapter=dspy.ChatAdapter())  # Default
dspy.configure(adapter=dspy.JSONAdapter())  # Structured JSON

# Manual per-call
dspy.configure(adapter=dspy.XMLAdapter())
```

### 5. Common Prediction Patterns

#### Chain of Thought (CoT)
```python
cot = dspy.ChainOfThought("question -> answer")
prediction = cot(question="Explain why the sky is blue.")
print(prediction.reasoning)  # The reasoning steps
print(prediction.answer)     # Final answer
```

#### ReAct (Reasoning + Tools)
```python
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

react = dspy.ReAct(signature="question->answer", tools=[get_weather])
result = react(question="How's the weather in London?")
```

#### Best of N (Multiple Attempts)
```python
def one_word_answer(args, pred):
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_n = dspy.BestOfN(
    module=cot,
    N=3,
    reward_fn=one_word_answer,
    threshold=1.0
)
result = best_of_n(question="France's capital?")
```

#### KNN (Few-Shot from Training Data)
```python
from sentence_transformers import SentenceTransformer

trainset = [
    dspy.Example(input="hello", output="world"),
    dspy.Example(input="goodbye", output="farewell"),
]

knn = dspy.KNN(
    k=3,
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer().encode)
)

similar = knn(input="hi there")
```

#### Parallel Execution
```python
examples = [
    {"question": "Paris?"},
    {"question": "London?"},
    {"question": "Tokyo?"},
]

program = dspy.Predict("question -> answer")
parallel = dspy.Parallel(num_threads=3)
results = parallel([(program, ex) for ex in examples])
```

#### Aggregation (Voting)
```python
predictions = [pred1, pred2, pred3]
majority = dspy.majorty(predictions, normalize=str.lower)
```

### 6. Evaluation

```python
# Create evaluator
evaluator = dspy.Evaluate(
    devset=devset,
    metric=lambda gold, pred: 1 if gold.answer == pred.answer else 0,
    num_threads=8
)

# Run evaluation
result = evaluator(program, metric=custom_metric)
print(result.score)  # Percentage score
print(result.results)  # List of (example, prediction, score)
```

**Built-in Metrics:**
```python
from dspy.evaluate.metrics import EM, F1

em_score = EM("Paris", ["Paris", "French city"])  # True
f1_score = F1("The Eiffel Tower", ["Eiffel Tower"])  # ~1.0
hotpot_f1 = HotPotF1("yes", ["no"])  # 0.0
```

### 7. Teleprompt Optimization

```python
# Labeled Few-Shot (add demos to predictors)
teleprompter = dspy.LabeledFewShot(k=16)
optimized = teleprompter.compile(student, trainset=trainset)

# Bootstrap Few-Shot (self-improve)
teleprompter = dspy.BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1
)
optimized = teleprompter.compile(student, trainset=trainset)

# Ensemble (combine multiple trained programs)
ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=5)
optimized = ensemble.compile([prog1, prog2, prog3])
```

### 8. Advanced Features

#### Streaming Output
```python
program = dspy.streamify(dspy.Predict("q -> a"))

async def stream_response():
    async for chunk in program(q="Why is the sky blue?"):
        print(chunk)
```

#### Retrieval (RAG)
```python
corpus_embeddings = embedder(corpus)
retriever = dspy.Retrieve(
    corpus=corpus,
    k=5,
    normalize=True
)

results = retriever(query="search query")
passages = results.passages  # Retrieved passages
indices = results.indices   # Original indices
```

#### Tool Calling (Avatar Pattern)
```python
tools = [tool1, tool2, tool3]
avatar = dspy.Avatar(signature="goal -> result", tools=tools)
output = avatar(goal="Achieve X", input_value=value)
```

### 9. Settings & Configuration

```python
# Global configuration
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    track_usage=True,
    adapter=dspy.ChatAdapter(),
    callbacks=[],
    max_errors=10
)

# Thread-local override
with dspy.context(num_threads=4, lm=other_lm):
    # Runs with temporary settings
    ...

# Check settings
settings.lm  # Current LM
settings.adapter  # Current adapter
```

### 10. Utility Methods

```python
# Check available methods
module.named_predictors()    # Named Predict modules
module.predictors()         # All Predict instances
module.set_lm(lm)           # Set LM for all predictors
module.get_lm()             # Get current LM

# Deep copy a program
program_clone = program.deepcopy()

# Reset program state
program.reset()
```

### Common Pitfalls & Tips

1. **LM must be instance, not string**:
   ```python
   # Correct: dspy.configure(lm=dspy.LM("..."))
   # Wrong: dspy.configure(lm="...")  # Will raise error
   ```

2. **Cache behavior**: 
   - `temperature > 0.15` bypasses cache naturally
   - Use `rollout_id` for explicit cache bypass

3. **ContextWindowExceededError**: 
   - Increase `max_tokens` or reduce content size
   - Consider streaming for long outputs

4. **Structured Outputs**:
   - Use `JSONAdapter` for reliable JSON parsing
   - Enable native function calling when available

5. **Debugging**:
   ```python
   with dspy.settings.context(provide_traceback=True):
       ...
   print(dspy.settings.trace)  # View LM call history
   ```

### Quick Reference Table

| Component | Description | Example |
|-----------|-------------|---------|
| `Predict` | Basic LM wrapper | `dspy.Predict("x -> y")` |
| `Signature` | Task definition | Class with InputField/OutputField |
| `LM` | Model interface | `dspy.LM("provider/model")` |
| `Example` | Data container | `.with_inputs("field")` |
| `Evaluate` | Metric runner | `Evaluate(devset=...)` |
| `Module` | Program base | Inherit and implement forward() |
| `ChainOfThought` | Step-by-step | `dspy.ChainOfThought("x->y")` |
| `ReAct` | Tool-using agent | `dspy.ReAct(sig, tools)` |
| `BestOfN` | Try multiple times | `dspy.BestOfN(mod, N=3)` |
| `Parallel` | Concurrent exec | `dspy.Parallel(n=threads)` |

---

*This cheatsheet covers 95% of common DSPy usage patterns. Refer to official docs for advanced topics like finetuning, experimental features, and production deployment.*

---

# Corrections from self-quizzing (verified against the source; trust these over the summary above)

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
