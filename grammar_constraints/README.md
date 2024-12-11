# Grammar-Constrained JSON Generation

This module implements a grammar-constrained approach to JSON generation using a Finite State Machine (FSM) to ensure syntactically valid JSON output.

## Key Components

### JSONFSM (json_fsm.py)
A finite state machine that enforces JSON grammar rules during generation:
- Tracks the current state of JSON generation (e.g., expecting a key, value, or delimiter)
- Maintains a stack to handle nested structures (objects and arrays)
- Modifies model logits to only allow valid next tokens according to JSON syntax

### GrammarJSONGenerator (grammar_json_generator.py)
Combines a language model with the FSM to generate valid JSON:
- Uses a pre-trained language model for text generation
- Applies FSM constraints at each generation step
- Forces proper JSON structure by ensuring valid opening/closing of brackets and braces

## How It Works

1. The generator receives a text prompt requesting JSON data
2. At each generation step:
   - The language model proposes next token probabilities
   - The FSM filters these probabilities to only allow grammatically valid tokens
   - The highest probability valid token is selected
3. Generation continues until a valid end state is reached or max length is hit