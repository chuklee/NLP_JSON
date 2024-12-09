# Grammar-Constrained JSON Generation

This implementation adds lightweight context-free grammar constraints to the JSON generation process using the Facebook OPT-350M model. The grammar constraints help ensure structural validity of generated JSON while maintaining generation flexibility.

## Components

1. `json_grammar.py`: Implements the grammar constraints and constrained generation
   - `JSONGrammarConstraints`: Defines and enforces JSON grammar rules
   - `GrammarConstrainedGenerator`: Handles generation with grammar constraints

2. `evaluation.py`: Provides evaluation tools and metrics
   - Compares standard vs. grammar-constrained generation
   - Measures performance, accuracy, and resource usage

## Features

- Lightweight context-free grammar for JSON validation
- Token-level constraints during generation
- Real-time structure validation
- Comprehensive evaluation metrics
- Memory and performance optimization

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run evaluation:
```bash
python evaluation.py
```

## Evaluation Metrics

1. **Structural Metrics**
   - Valid JSON rate
   - Structure accuracy
   - Field accuracy

2. **Performance Metrics**
   - Inference time
   - Memory usage
   - Generation success rate

## Grammar Rules

The implementation uses a simplified context-free grammar for JSON:

```
START -> OBJECT
OBJECT -> { MEMBERS }
MEMBERS -> PAIR | PAIR,MEMBERS
PAIR -> STRING : VALUE
VALUE -> STRING | NUMBER | OBJECT | ARRAY | true | false | null
ARRAY -> [ VALUES ]
VALUES -> VALUE | VALUE,VALUES
```

## Results

The evaluation compares:
- Standard generation (baseline)
- Grammar-constrained generation

Key metrics include:
- JSON validity improvement
- Impact on inference time
- Memory overhead
- Structure accuracy
- Field accuracy (semantic correctness)
