from openai import OpenAI
from pydantic import BaseModel

def call_llm(prompt, api_key, llm_type="gpt-4o"):
    """Function to call the GPT-4V model 

    Args:
        prompt (str): Spatial instruction for GPT-4 model
        image_path (str): Path to the image
        api_key (str): OpenAI API key

    Returns:
        str: Output from GPT-4 model
    """

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]
   
    class MathReasoning(BaseModel):
        instruction_type: str
        reason: str
    
    client = OpenAI(api_key = api_key)
    
    kwargs = {
        "model": llm_type,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ],
        "response_format": MathReasoning,
    }

    if "-4o" in llm_type:
        kwargs["max_tokens"] = 10000
    else:
        kwargs["max_completion_tokens"] = 10000

    completion = client.beta.chat.completions.parse(**kwargs)

    instruction_type_output = completion.choices[0].message.parsed.instruction_type
    instruction_type_output = instruction_type_output.lower()
    reason = completion.choices[0].message.parsed.reason

    return instruction_type_output, reason