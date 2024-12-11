import torch
from grammar_json_generator import GrammarJSONGenerator
from json_fsm import JSONState
import torch.nn.functional as F

def debug_token_generation(generator, prompt, max_length=100, temperature=1.0, top_p=1.0):
    """Debug token generation with FSM constraints"""
    # Tokenize the prompt
    input_ids = generator.tokenizer.encode(prompt, return_tensors="pt").to(generator.device)
    
    # Initialize output with an empty tensor
    output_ids = []
    
    # Generate tokens one at a time
    for _ in range(max_length):
        # Get logits for next token
        outputs = generator.model(input_ids, return_dict=True)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply FSM constraints
        next_token_logits = generator.fsm.modify_logits_for_json(next_token_logits, output_ids)
        
        # Get probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Apply top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=0, index=sorted_indices, src=sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum()
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Print generation step info
        print(f"\nStep {len(output_ids)}:")
        print(f"Current FSM state: {generator.fsm.state}")
        print(f"FSM stack: {generator.fsm.stack}")
        print("\nTop 5 tokens before FSM constraints:")
        top_logits, top_indices = torch.topk(outputs.logits[0, -1, :], k=5)
        for logit, idx in zip(top_logits, top_indices):
            try:
                token_str = generator.tokenizer.decode([idx.item()])
                print(f"Token: '{token_str}', Logit: {logit:.2f}")
            except:
                print(f"Token: <special>, Logit: {logit:.2f}")
        print("\nTop 5 tokens after FSM constraints:")
        top_logits, top_indices = torch.topk(next_token_logits, k=5)
        for logit, idx in zip(top_logits, top_indices):
            try:
                token_str = generator.tokenizer.decode([idx.item()])
                print(f"Token: '{token_str}', Logit: {logit:.2f}")
            except:
                print(f"Token: <special>, Logit: {logit:.2f}")
        try:
            token_str = generator.tokenizer.decode([next_token.item()])
            print(f"\nSelected token: '{token_str}'")
        except:
            print(f"\nSelected token: <special>")
        
        # Force opening brace at the start
        if len(output_ids) == 0 and next_token.item() != generator.fsm.open_brace_token:
            print("\nForced opening brace!")
            next_token = torch.tensor([generator.fsm.open_brace_token], device=generator.device)
        
        # Update output
        output_ids.append(next_token.item())
        try:
            print(f"\nCurrent output: {generator.tokenizer.decode(output_ids)}")
        except:
            print("\nCurrent output: <error decoding>")
        
        # Update FSM state
        generator.fsm.update_state(output_ids)
        
        # Stop if we've reached the end state
        if generator.fsm.state == JSONState.END:
            break
        
        # Prepare input for next iteration
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Check if we hit max_length
    if len(output_ids) == max_length:
        print("\nReached max length!")
    
    return output_ids

def main():
    # Initialize generator
    generator = GrammarJSONGenerator("facebook/opt-350m")
    
    # Test prompt
    prompt = "Convert the following sentence into a JSON object with clear key-value pairs: 'I bought 2 flowers and a flower pot.'"
    expected = "{'items_purchased': [{'type': 'flowers', 'quantity': 2}, {'type': 'flower pot', 'quantity': 1}]}"
    
    print("Testing Grammar-Constrained JSON Generation")
    print("=" * 50)
    print(f"Expected output: {expected}\n")
    
    # Generate with debug info
    generated = debug_token_generation(generator, prompt)
    
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Generated: {generator.tokenizer.decode(generated)}")

if __name__ == "__main__":
    main()
