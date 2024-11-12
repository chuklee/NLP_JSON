import json
import jsonschema
import pytest

def test_llm_json_output_structure(openai_client, character_schema):
    """Test if the LLM generates valid JSON according to our schema"""
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create 1-3 fictional characters"}
    ]
    
    response = openai_client.chat.completions.create(
        model="your-model",
        messages=messages,
        response_format=character_schema,
    )
    
    # Parse the response
    result = json.loads(response.choices[0].message.content)
    
    # Basic structure tests
    assert isinstance(result, dict), "Response should be a dictionary"
    assert "characters" in result, "Response should have 'characters' key"
    assert isinstance(result["characters"], list), "'characters' should be a list"
    assert len(result["characters"]) >= 1, "Should have at least 1 character"
    
    # Validate each character
    for character in result["characters"]:
        assert isinstance(character["name"], str), "Name should be a string"
        assert isinstance(character["occupation"], str), "Occupation should be a string"
        assert isinstance(character["personality"], str), "Personality should be a string"
        assert isinstance(character["background"], str), "Background should be a string"

def test_schema_validation(openai_client, character_schema):
    """Test if the output strictly conforms to the JSON schema"""
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create 1-3 fictional characters"}
    ]
    
    response = openai_client.chat.completions.create(
        model="your-model",
        messages=messages,
        response_format=character_schema,
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Validate against the schema
    try:
        jsonschema.validate(instance=result, schema=character_schema["json_schema"]["schema"])
    except jsonschema.exceptions.ValidationError as e:
        pytest.fail(f"Schema validation failed: {str(e)}")

def test_character_count_limits(openai_client, character_schema):
    """Test if the LLM respects the character count limits"""
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Create 152 fictional characters"}
    ]
    
    response = openai_client.chat.completions.create(
        model="your-model",
        messages=messages,
        response_format=character_schema,
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Check if number of characters is within bounds
    assert len(result["characters"]) == 152, "Number of characters should be between 1 and 50" 