import unittest
from transformers import AutoTokenizer
from .json_fsm import JSONFSM, JSONState

def test_fsm_transitions():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    fsm = JSONFSM(tokenizer) # type: ignore
    
    def process_json(json_str):
        print(f"\nTesting: {json_str}")
        # Process one token at a time to simulate token generation
        tokens = tokenizer.encode(json_str, add_special_tokens=False)
        for token in tokens:
            valid_tokens = fsm.get_valid_next_tokens([token])
            fsm.update_state([token])
            print(f"Token: {tokenizer.decode([token])}, State: {fsm.state}, Stack: {fsm.stack}")
    
    # Test cases
    test_cases = [
        '{"key": "value"}',
        '{"outer": {"inner": "value"}}',
        '{"array": [1, 2, 3]}',
        '{"mixed": {"arr": [1, {"x": "y"}, 3]}}'
    ]
    
    for test_case in test_cases:
        fsm = JSONFSM(tokenizer) # type: ignore
        process_json(test_case)

class TestJSONFSM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls.fsm = JSONFSM(cls.tokenizer)

    def setUp(self):
        """Reset FSM state before each test"""
        self.fsm = JSONFSM(self.tokenizer)

    def test_initialization(self):
        """Test if FSM initializes correctly"""
        self.assertEqual(self.fsm.state, JSONState.START)
        self.assertEqual(len(self.fsm.stack), 0)
        self.assertIsInstance(self.fsm.quote_token, int)
        self.assertIsInstance(self.fsm.comma_token, int)

    def test_valid_next_tokens(self):
        """Test if valid next tokens are returned correctly"""
        # At start, only open brace should be valid
        valid_tokens = self.fsm.get_valid_next_tokens([])
        print(f"\nInitial state: {self.fsm.state}")
        print(f"Number of valid tokens: {len(valid_tokens)}")
        self.assertIn(self.fsm.open_brace_token, valid_tokens)
        self.assertNotIn(self.fsm.close_brace_token, valid_tokens)
        self.assertNotIn(self.fsm.quote_token, valid_tokens)
        
        # After open brace, we should expect a quote for key or close brace for empty object
        self.fsm.update_state([self.fsm.open_brace_token])
        print(f"\nAfter open brace state: {self.fsm.state}")
        valid_tokens = self.fsm.get_valid_next_tokens([self.fsm.open_brace_token])
        print(f"Number of valid tokens: {len(valid_tokens)}")
        self.assertIn(self.fsm.quote_token, valid_tokens)
        self.assertIn(self.fsm.close_brace_token, valid_tokens)
        self.assertNotIn(self.fsm.open_brace_token, valid_tokens)

    def test_state_transitions(self):
        """Test state transitions"""
        # Start -> Expect Key
        self.fsm.update_state([self.fsm.open_brace_token])
        self.assertEqual(self.fsm.state, JSONState.EXPECT_KEY)
        
        # Expect Key -> In Key
        self.fsm.update_state([self.fsm.quote_token])
        self.assertEqual(self.fsm.state, JSONState.IN_KEY)

if __name__ == "__main__":
    test_fsm_transitions()
    unittest.main()
