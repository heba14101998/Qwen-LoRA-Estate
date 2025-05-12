import re
import json_repair 
from typing import Optional
from pydantic import BaseModel, Field

# Define system message
SYSTEM_MESSAGE = "\n".join([
    "You are an expert in real estate price estimation with experience in the housing market.",
    "Given the following house features, predict the final sale price.",
    "#### Critical notes:",
    "- Some feature values are missing.",
    "- Broker ID and street are encoded for privacy.",
    "- Do not include any introduction or conclusion."
])

class ResponseSchema(BaseModel):
    """
    Define the response schema.
    """
    estimated_house_price: float = Field(...,
                                description="Numerical value that expresses the estimated house price",
                                example=85000.0)

def apply_prompt_template(sample: dict, output_str: str ='instruction', tokenizer: Optional[object] = None) -> str:
    """
    Apply the prompt template to the sample. 
    """
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, 
                {"role": "user", "content": sample[output_str].strip()}]
    
    if tokenizer: # for pretrained model like Qwem3-0.6B
        # Apply the model template on the prompt
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else: # for gemini api
        return "\n".join([msg["content"] for msg in messages])


def decode_response(response_tokens_ids, input_tokens_ids, tokenizer) -> str:
    """
    Decode the response tokens to text. 
    """
    input_length = len(input_tokens_ids[0])
    output_tokens_ids = response_tokens_ids[0][input_length:]
    
    return tokenizer.decode(output_tokens_ids, skip_special_tokens=True)

def extract_house_price(response):
    # Extract all JSON code blocks
    json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    
    # Try parsing each block
    for block in json_blocks:
        data = json_repair.loads(block)
        if isinstance(data, dict) and 'estimated_house_price' in data:
            price = data['estimated_house_price']
            return price
    return -1
