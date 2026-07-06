# DSPy Complete Cheatsheet
## 🎯 Core Architecture & Quick Start

### Module Composition (Foundation)
```python
import dspy

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")
        self.classifier = dspy.ChainOfThought("entities -> sentiment")

    def forward(self, question):
        return self.predictor(question=question)
```

### Global Configuration (DO THIS FIRST!)
```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini"),
    adapter=dspy.ChatAdapter(use_json_adapter_fallback=True),
    track_usage=True,
    max_errors=10,
    num_threads=8
)

# Temporarily override
with dspy.context(temperature=0.9, max_tokens=2000):
    result = predictor(**kwargs)

# Apply LM to entire program tree
program.set_lm(dspy.LM("anthropic/claude-3-5-sonnet"))
```

## 📝 SIGNATURES & FIELDS

### Class-based Definition (RECOMMENDED FOR COMPLEX TASKS)
```python
class QASignature(dspy.Signature):
    question: str = dspy.InputField(desc="User question")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")
    answer: str = dspy.OutputField(desc="Final answer")

pred = QASignature(question="What is 1+1?")
print(pred.answer)  # "2"
```

### String Format (Quick Setup)
```python
sig = dspy.Signature("question, context -> answer", "Translate to French.")
sig = dspy.Signature({
    "q": (str, dspy.InputField()), 
    "a": (str, dspy.OutputField())
})
```

### Field Operations
```python
NewSig = MySig.prepend("context", dspy.InputField(desc="Context info"))
NewSig = MySig.append("confidence", dspy.OutputField(desc="Confidence score"))
NewSig = MySig.delete("temp_field")
NewSig = MySig.with_instructions("Custom prompt here")
NewSig = MySig.with_updated_fields("input_text", desc="Updated description")
pred.demos = [{"question": "Hello", "answer": "Hi there"}]
```

## 🔮 Core Prediction Patterns

### Basic Predict Module
```python
predict = dspy.Predict("input -> output", temperature=0.7)
pred = predict(input="value")
print(pred.output)  # Access prediction field

# Per-call overrides
pred = predict(input="value", config={"temperature": 1.0, "max_tokens": 500})

# Reset demos/traces
predict.reset()

# Save/Load state
state = predict.dump_state(json_mode=True)
restored = predict.load_state(state, allow_unsafe_lm_state=True)
```

### Chain of Thought (Reasoning)
```python
cot = dspy.ChainOfThought("question -> reasoning -> answer")
pred = cot(question="What is 1+1?")
print(f"Reasoning: {pred.reasoning}")
print(f"Answer: {pred.answer}")
```

### ReAct (Tool-Agents with Reasoning)
```python
def search_db(query: str) -> str:
    return f"Search results for {query}"

def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny"

react = dspy.ReAct(signature="q->a", tools=[search_db, get_weather], max_iters=10)
pred = react(question="Find weather in Tokyo?")
print(pred.trajectory)  # See full reasoning steps with tool calls
```

### ProgramOfThought (Code Generation + Execution)
```python
# Requires deno installed: https://docs.deno.com/runtime/getting_started/installation/
pot = dspy.ProgramOfThought("question -> answer", max_iters=3)
result = pot(question="Calculate complex math using Python?")
```

### KNN Retrieval-Based Few-Shot
```python
from sentence_transformers import SentenceTransformer

trainset = [
    dspy.Example(q="hello", a="world"),
    dspy.Example(q="hi there", a="friend")
]

knn = dspy.KNN(
    k=3,                              # Number of neighbors
    trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2"))
)

similar_examples = knn(q="hello world")  # Returns list of similar examples
```

### BestOfN (Optimization Pattern with Rewards)
```python
module = dspy.ChainOfThought("question -> answer")

def reward_fn(args, pred):
    # Return 0.0-1.0 score (higher is better)
    return len(pred.answer.split()) < 10 if isinstance(pred.answer, str) else 0.5

best = dspy.BestOfN(
    module=module,
    N=5,                    # Try 5 times
    reward_fn=reward_fn,
    threshold=0.8,          # Stop early if we hit this score
    fail_count=None         # Allow up to N failures
)
result = best(question="Short answer test?")
```

## 🔧 Optimization / Teleprompting

### Labeled Few-Shot (Vanilla)
```python
vanilla = dspy.LabeledFewShot(k=16)  # Sample k labeled demos
optimized = vanilla.compile(student, trainset=trainset)
```

### Bootstrap Few-Shot (Self-Generated Demos)
```python
bootstrap = dspy.teleprompt.BootstrapFewShot(
    metric=my_metric,
    metric_threshold=0.9,
    teacher_settings={"lm": teacher_lm},  # Optional separate teacher
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=3
)
optimized = bootstrap.compile(student, teacher=teacher, trainset=trainset)
```

### Ensemble Methods (Combining Multiple Programs)
```python
ensemble = dspy.teleprompt.Ensemble(
    size=5,                  # Random sample size
    reduce_fn=dspy.majority   # Combine multiple outputs
)
optimized_ensemble = ensemble.compile([prog1, prog2, prog3])
```

### BetterTogether (Meta-Optimizer: Prompt + Weight)
```python
better = dspy.teleprompt.BetterTogether(
    metric=my_metric,
    p=dspy.teleprompt.BootstrapFewShotWithRandomSearch(metric=my_metric),  # Prompt opt
    w=dspy.teleprompt.BootstrapFinetune(metric=my_metric)      # Weight opt
)

student.set_lm(lm)  # Required for weight optimizers!
result = better.compile(student, trainset=trainset, strategy="p -> w")
```

### GEPA (Gradient Evolutionary Prompt Optimization)
```python
gepa = dspy.teleprompt.GEPA(
    metric=my_metric,
    auto="medium",              # "none", "light", "medium", "heavy"
    reflection_lm=dspy.LM("gpt-5", temperature=1.0, max_tokens=32000),  # REQUIRED
    candidate_selection_strategy="pareto",
    component_selector="round_robin",
    use_merge=True,
    track_stats=True
)
optimized = gepa.compile(student, trainset=trainset, valset=valset)
```

### MIPROv2 (Multi-Instruction Prompt Optimization)
```python
mipro = dspy.teleprompt.MIPROv2(
    metric=my_metric,
    auto="medium",
    num_trials=10,
    max_bootstrapped_demos=8
)
optimized = mipro.compile(student, trainset=trainset, valset=valset)
```

### COPRO (Successor to deprecated SignatureOptimizer)
```python
copro = dspy.teleprompt.COPRO(
    metric=my_metric,
    breadth=10,              # Number of new prompts per iteration
    depth=3,                 # Times to ask prompt model
    init_temperature=1.4
)
optimized = copro.compile(student.deepcopy(), trainset=trainset, eval_kwargs={})
```

**DEPRECATED**: `SignatureOptimizer` → Use `COPRO` instead.

## 💾 Data Structures

### Example Creation & Usage
```python
example = dspy.Example(
    question="What is 2+2?",
    answer="Four"
).with_inputs("question")

# Dict-like access
example.question            # Attribute access
example["answer"]          # Item access
list(example.keys())       # ['question', 'answer']
example.inputs().toDict()  # {'question': '...'}
```

### Tracing & History
```python
dspy.settings.trace  # Current trace list (per call)
trace[-1]            # Last trace entry (module, inputs, outputs)
module.history       # Call history for debugging
```

### Adapter Selection
```python
# Default Chat adapter with JSON fallback (BEST CHOICE)
dspy.ChatAdapter(use_json_adapter_fallback=True)

# Force specific adapter
dspy.JSONAdapter()  # Structured outputs (recommended for code)
dspy.XMLAdapter()   # XML-formatted responses
settings.adapter = dspy.ChatAdapter()
```

## ⚙️ Settings & Configuration Reference

| Setting | Description | Default |
|---------|-------------|--------|
| `dspy.settings.lm` | Global language model | None |
| `dspy.settings.adapter` | Message formatting adapter | ChatAdapter |
| `dspy.settings.num_threads` | Parallel threads | 8 |
| `dspy.settings.max_errors` | Max errors before stopping | 10 |
| `dspy.settings.track_usage` | Track token usage | False |
| `dspy.settings.max_trace_size` | Max traces to keep | 10000 |

### Common Configurations
```python
# Minimal setup
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Production-ready
dspy.configure(
    lm=dspy.LM("anthropic/claude-3-5-sonnet"),
    adapter=dspy.JSONAdapter(),  # For reliable structured outputs
    track_usage=True,
    num_threads=16,
    max_errors=20
)
```

## 📊 Evaluation Framework

### Running Evaluation
```python
from dspy.evaluate import Evaluate

metric = lambda gold, pred: normalize(gold.answer.lower()) == normalize(pred.answer.lower())

evaluator = Evaluate(
    devset=trainset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=20,
    save_as_csv="results.csv"
)

result = evaluator(my_program)
print(f"Score: {result.score}%")  # Overall score
print(result.results)             # Detailed per-example results
```

### Common Metrics (from dspy.evaluate.metrics)
```python
EM("Paris", ["Paris", "London"])           # True (exact match after normalization)
F1("Eiffel Tower", "the Eiffel Tower")     # 1.0 (token-level F1)
HotPotF1("yes", "no")                      # 0.0 (special handling for yes/no cases)

# Custom metric example
def exact_match(gold, pred):
    return normalize(gold) == normalize(pred)

def f1_score(gold, pred):
    return compute_f1(normalize(gold), normalize(pred))
```

### Text Normalization Used
```python
normalize_text(s):
    1) Unicode NFD normalization
    2) Lowercase
    3) Remove punctuation
    4) Strip articles (a/an/the)
    5) Collapse whitespace
```

## 🧩 Advanced Module Composition

### Creating Complex Pipelines
```python
class ComplexPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict("text -> entities")
        self.classifier = dspy.ChainOfThought("entities -> sentiment")

    def forward(self, text):
        entities = self.extractor(text=text)
        sentiment = self.classifier(entities=entities.entities)
        return {
            "entities": entities.entities,
            "sentiment": sentiment.sentiment
        }
```

### Parallel Execution
```python
parallel = dspy.Parallel(module=predictor, n_threads=4)
results = parallel(trainset)  # Apply to multiple examples

# With error handling
results, failed_examps, exceptions = parallel.exec_pairs_with_errors(exec_pairs)
```

## 🛠 Common Errors & Fixes

| Error | Cause | Solution |
|-------|-------|----------|
| `No LM is loaded` | Missing configure call | `dspy.configure(lm=...)` |
| `Type mismatch` | Input value type ≠ signature | Add explicit types to signature |
| `Context window exceeded` | Prompt too long | Reduce demonstrations; use streaming |
| `LM must be BaseLM instance` | Using string instead | Wrap in dspy.LM() |
| `Positional args not allowed` | Wrong calling style | Use keyword args only: `predict(key=value)` |
| `Student must be uncompiled` | Trying to compile compiled program | Call `reset_copy()` first |
| `Teacher/student structure mismatch` | Different number of predictors | Ensure same architecture |
| `GEPA requires reflection_lm` | Missing reflection model | Provide strong LM for reflection |
| `SignatureOptimizer deprecated` | Using legacy optimizer | Use COPRO instead |

## 📚 Quick Command Reference

| Command | Purpose |
|---------|--------|
| `dspy.configure(X)` | Set global defaults |
| `dspy.context(X)` | Temporary override |
| `module.set_lm(X)` | Apply LM to entire tree |
| `module.reset()` | Clear demos/traces/history |
| `Evaluate(devset, metric)` | Run evaluation |
| `Teleprompter.compile(student, ...)` | Optimize program |
| `module.named_parameters()` | Get all sub-modules |
| `module.predictors()` | Get all Predict instances |

---
**Last Updated**: Comprehensive DSPy v3.0+ reference based on extensive repository exploration