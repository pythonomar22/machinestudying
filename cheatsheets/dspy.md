

# DSPy Cheatsheet

## 1. Core Building Blocks

### Signature
```python
import dspy

class MySig(dspy.Signature):
    """Your instructions go here"""
    question: str = dspy.InputField(desc="Input question")
    answer: str = dspy.OutputField(desc="The answer")

# Or using string format
sig = dspy.Signature("question, context -> answer")

# Signature methods
NewSig = MySig.with_instructions("New instructions")  # Create new signature
NewSig = MySig.append("new_field", dspy.InputField())  # Add field
NewSig = MySig.prepend("new_field", dspy.InputField())  # Prepend field
NewSig = MySig.delete("field_name")  # Remove field
```

### Predict
```python
predict = dspy.Predict("question -> answer")  # Simple predictor

# With custom config
predict = dspy.Predict("q->a", temperature=0.7)

# Call with keyword args matching signature
result = predict(question="What is 2+2?")
print(result.answer)
```

### Module
```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")
    
    def forward(self, question):
        return self.predictor(question=question)

# Use module
module = MyModule()
result = module(question="Hello")
```

### Example
```python
example = dspy.Example(
    question="What is the capital?",
    answer="Paris"
).with_inputs("question")  # Mark which fields are inputs

# Access
example.question
example.inputs().toDict()
example.labels()  # Non-input fields

# In trainset
trainset = [
    dspy.Example(q="...", a="...").with_inputs("q"),
]
```

## 2. Language Model Configuration

### LM Creation
```python
lm = dspy.LM(
    "openai/gpt-4o-mini",          # Model identifier
    model_type="chat",              # "chat", "text", or "responses"
    temperature=0.7,                # Sampling temperature
    max_tokens=None,                # Max tokens per response
    cache=True,                     # Enable caching
    num_retries=3,                  # Retry count
    provider=dspy.OpenAIProvider()  # Optional explicit provider
)

dspy.configure(lm=lm)  # Set global default
```

### Settings
```python
# Configure globally (only once per process)
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.JSONAdapter(),
    track_usage=True,
    callbacks=[my_callback],
    num_threads=8,
    max_errors=10,
)

# Temporary override in specific block
with dspy.context(lm=dspy.LM("anthropic/claude-3-5")):
    result = predict(...)
# Back to original LM
```

### LM Methods
```python
# Copy with modifications (bypasses cache when needed)
new_lm = lm.copy(rollout_id=1, temperature=1.0)

# Check capabilities
lm.supports_function_calling   # bool
lm.supports_reasoning          # bool
lm.supports_response_schema    # bool
lm.supported_params            # set of supported parameters
```

## 3. Chain of Thought & Aggregation

### ChainOfThought
```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="Explain photosynthesis")

# Access reasoning
str(result.reasoning)
result.reasoning.content  # Pydantic Reasoning type in newer versions
```

### Aggregation
```python
from dspy.evaluate import normalize_text
from dspy.aggregation import majority

# Majority voting over multiple completions
predictions = [p1, p2, p3]  # List of Prediction objects
majority_prediction = majority(predictions)
majority_prediction = majority([p1, p2, p3], field="answer")  # Specify field

# Normalize strings before comparison
normalized = lambda x: normalize_text(x)
majority_norm = majority(predictions, normalize=normalized)
```

## 4. Adapters (Prompt Formatting)

### ChatAdapter (Default)
```python
adapter = dspy.ChatAdapter(
    use_json_adapter_fallback=True,  # Auto-fallback if parsing fails
    use_native_function_calling=False
)

dspy.configure(adapter=adapter)
```

### JSONAdapter
```python
adapter = dspy.JSONAdapter(
    use_native_function_calling=True
)  # Uses structured output API when available
```

### XMLAdapter
```python
# Format like <field_name>content</field_name>
adapter = dspy.XMLAdapter()
```

### TwoStepAdapter
```python
# Main LM + extraction LM pattern (good for reasoning models)
main_lm = dspy.LM("openai/o3-mini")
extractor_lm = dspy.LM("openai/gpt-4o-mini")
adapter = dspy.TwoStepAdapter(extractor_model=extractor_lm)

dspy.configure(lm=main_lm, adapter=adapter)
```

## 5. Custom Types

### Image
```python
img = dspy.Image(
    url="https://example.com/image.jpg",
    download=False  # Download remote URLs to infer MIME type
)
# or from PIL Image
img = dspy.Image(pil_image_obj)
```

### Tool (Function Calling)
```python
def search(query: str) -> str:
    """Search function"""
    return f"Results for {query}"

tool = dspy.Tool(search)
# or with custom schema
tool = dspy.Tool(
    search,
    name="search",
    desc="Search for information",
    arg_desc={"query": "The search query"}
)

class Agent(dspy.Signature):
    query: str = dspy.InputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()  # Native tool calling
```

### Reasoning
```python
class MySig(dspy.Signature):
    question: str = dspy.InputField()
    reasoning: dspy.Reasoning = dspy.OutputField()  # Will enable native reasoning

# When parse, it becomes native `reasoning_content` in response
result = predict(question="Solve this")
str(result.reasoning)  # String content
```

### Document & Citations
```python
from dspy.experimental import Document, Citations

docs = [
    dspy.Document(data="Content", title="Source 1"),
    dspy.Document(data="More content", title="Source 2"),
]

class AnswerWithSources(dspy.Signature):
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()
```

## 6. Retrievers

### Embeddings Retriever
```python
embedder = dspy.EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
retriever = dspy.Retrieve(k=3, embedder=embedder)

corpus = ["Doc 1...", "Doc 2...", ...]
emb_retriever = dspy.Embeddings(corpus, embedder)
results = emb_retriever.forward("search query")
print(results.passages, results.indices, results.scores)

# Save/load
emb_retriever.save("./saved_embeddings")
loaded = dspy.Embeddings.from_saved("./saved_embeddings", embedder)
```

### Retrieve Parameter
```python
retrieve = dspy.Retrieve(k=3)  # Default
prediction = retrieve(query="Search query")
```

## 7. Evaluation

### Metrics
```python
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

def normalized_exact_match(example, pred, trace=None):
    from dspy.metrics import normalize_text
    return normalize_text(example.answer) == normalize_text(pred.answer)

from dspy.evaluate import EM, F1, HotPotF1
from dspy.evaluate.auto_evaluation import SemanticF1
```

### Evaluate Class
```python
devset = [
    dspy.Example(q="...", a="...").with_inputs("q"),
]

evaluate = dspy.Evaluate(
    devset=devset,
    metric=exact_match,
    num_threads=4,
    display_progress=True,
    display_table=True,
)

result = evaluate(program)
print(f"Score: {result.score}")  # e.g., 85.5%
```

### SemanticF1 (Auto-evaluation)
```python
semantic_metric = SemanticF1(threshold=0.66)
# Can also be decompositional
semantic_metric = SemanticF1(threshold=0.66, decompositional=True)
```

## 8. Teleprompters (Optimization)

### BootstrapFewShot
```python
teleprompter = dspy.Teleprompter(
    BootstrapFewShot(
        metric=exact_match,
        teacher_settings={"lm": teacher_lm},  # Better teacher model
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
    )
)

optimized_program = teleprompter.compile(
    student=my_module,
    trainset=trainset,
)
```

### LabeledFewShot
```python
tf = dspy.Teleprompter.LabeledFewShot(k=16)
compiled_student = tf.compile(student, trainset=trainset)
```

### InferRules
```python
infer_rules = dspy.InferRules(
    num_candidates=10,
    num_rules=10,
    num_threads=4
)
```

### Ensemble
```python
ensemble = dspy.Ensemble(
    reduce_fn=majority,  # dspy.majority by default
    size=None,           # Use all programs
    deterministic=False
)
```

## 9. Parallel Execution

### Parallel Module
```python
parallel = dspy.Parallel(
    num_threads=4,
    max_errors=10,
    disable_progress_bar=False,
)

examples = [ex1, ex2, ex3]
results = parallel([(my_module, ex1), (my_module, ex2)])
```

### Batch Method
```python
# On Module instance
results = my_module.batch(
    examples,
    num_threads=4,
    return_failed_examples=True
)
```

## 10. History & Debugging

### Inspect History
```python
# Global history
dspy.inspect_history(n=5)

# Per-module history
my_module.inspect_history(n=3)

# Disable history
dspy.configure(disable_history=True)
```

### Usage Tracking
```python
dspy.configure(track_usage=True)

# Context manager
with dspy.track_usage():
    # Code to track
    pass

tracker = tracker.get_total_tokens()
```

### Callbacks
```python
class LoggingCallback(dspy.utils.BaseCallback):
    def on_lm_start(self, call_id, instance, inputs):
        print(f"LM called with: {inputs}")
    
    def on_lm_end(self, call_id, outputs, exception):
        print(f"LM finished with: {outputs}")

dspy.configure(callbacks=[LoggingCallback()])
```

## 11. Settings Reference

```python
# Available settings
dspy.settings.lm                      # Language model
dspy.settings.adapter                 # Adapter
dspy.settings.rm                      # Retriever
dspy.settings.callbacks               # Callback handlers
dspy.settings.track_usage             # Bool - track usage stats
dspy.settings.num_threads             # Parallel threads
dspy.settings.max_errors              # Stop after X errors
dspy.settings.disable_history         # Disable history logging
dspy.settings.max_history_size        # Max entries to keep
dspy.settings.provide_traceback       # Include traceback in errors
dspy.settings.allow_tool_async_sync_conversion  # Async tool handling
```

## 12. Program Patterns

### Simple Pipeline
```python
class QAChain(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.Predict("query, passages -> answer")
    
    def forward(self, query):
        retrieved = self.retrieve(query=query)
        context = "\n\n".join(retrieved.passages)
        answer = self.generate_answer(query=query, passages=context)
        return answer
```

### Multi-Step Module
```python
class MultiStepModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.Predict("input -> intermediate")
        self.step2 = dspy.Predict("intermediate -> output")
    
    def forward(self, input):
        intermediate = self.step1(input=input)
        final = self.step2(intermediate=intermediate.intermediate)
        return final
```

### Composed Module with Teleprompting
```python
class OptimizableModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")
        self.reasoner = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        answer_direct = self.predictor(question=question)
        answer_cot = self.reasoner(question=question)
        
        # Could choose best one, return tuple, etc.
        return answer_cot  # Chain of thought typically performs better
```

## 13. Advanced Features

### Streaming
```python
def stream_result(module, question):
    with streamify(module)(stream_listeners=[StreamListener()]):
        yield module(question=question)
```

### Async/Await
```python
async def async_predict():
    result = await my_module(aforward)
    return result
```

### Save/Load
```python
# Save entire program state
program.dump_state()

# Load state later
program.load_state(saved_state)

# Save modules specifically
save_as_cloudpickle(my_module, "./my_module.pkl")

# Load from pickle
restored_module = load("./my_module.pkl")
```

## 14. Common Error Handling

```python
try:
    prediction = predict(question="Hard task")
except dspy.utils.exceptions.ContextWindowExceededError:
    logger.warning("Context window exceeded, truncating prompt")
    # Fall back strategy
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    return None

# Max errors configuration
dspy.configure(max_errors=5)
# Stop early if too many failures occur
```