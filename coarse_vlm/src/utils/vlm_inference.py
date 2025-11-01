import os
import base64
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import yaml

def encode_image(image_path):
    """Function to encode the image in the format needed by GPT-4 model

    Args:
        image_path (str): Path to the image file

    Returns:
        base64: encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vlm_generator(prompt, image_path, api_key, output_shape, vlm_type="gpt-4o"):
    """Function to call the GPT-4V model 

    Args:
        prompt (str): Spatial instruction for GPT-4 model
        image_path (str): Path to the image
        api_key (str): OpenAI API key

    Returns:
        str: Output from GPT-4 model
    """
    # Getting the base64 string
    base64_image = encode_image(image_path)

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]
    class Step(BaseModel):
        explanation: str
        output: str
    
    if output_shape == "polygon" or output_shape == "point_fixed":
        class MathReasoning(BaseModel):
            steps: List[Step]  # Use List instead of list
            final_answer: List[List[int]]
            # relevant_objects: List[int]
            
    elif output_shape == "point":
        class MathReasoning(BaseModel):
            steps: List[Step]  # Use List instead of list
            center_coordinates: List[List[int]]
            radius: int
            # relevant_objects: List[int]
    
    elif output_shape == "ellipse":
        class MathReasoning(BaseModel):
            steps: List[Step]  # Use List instead of list
            center_coordinates: List[List[int]]
            semi_axes_lengths: List[List[int]]
            angle: List[int]
            # relevant_objects: List[int]
            
    
    client = OpenAI(api_key = api_key)
    
    kwargs = {
        "model": vlm_type,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
                    }],
            },
        ],
        "response_format": MathReasoning,
    }
    
    if "-4o" in vlm_type:
        kwargs["max_tokens"] = 10000
    else:
        kwargs["max_completion_tokens"] = 10000

    completion = client.beta.chat.completions.parse(**kwargs)


    if output_shape == "polygon" or output_shape == "point_fixed":
        list_of_points = completion.choices[0].message.parsed.final_answer
        steps_followed = completion.choices[0].message.parsed.steps
        # relevant_objects = completion.choices[0].message.parsed.relevant_objects
        return list_of_points, steps_followed, []#, relevant_objects
    elif output_shape == "ellipse":
        center_coordinates = completion.choices[0].message.parsed.center_coordinates
        axes = completion.choices[0].message.parsed.semi_axes_lengths
        angle = completion.choices[0].message.parsed.angle
        steps_followed = completion.choices[0].message.parsed.steps
        # relevant_objects = completion.choices[0].message.parsed.relevant_objects

        return center_coordinates, axes, angle, steps_followed, [] #, relevant_objects
    else:
        center_coordinates = completion.choices[0].message.parsed.center_coordinates
        radius = completion.choices[0].message.parsed.radius
        steps_followed = completion.choices[0].message.parsed.steps
        # relevant_objects = completion.choices[0].message.parsed.relevant_objects

        return center_coordinates, radius, steps_followed, [] #, relevant_objects

def call_vlm_validator(prompt, image_path, api_key, validation_type, vlm_type="gpt-4o"):
    """Function to call the GPT-4V model 

    Args:
        prompt (str): Spatial instruction for GPT-4 model
        image_path (str): Path to the image
        api_key (str): OpenAI API key

    Returns:
        str: Output from GPT-4 model
    """
    # Getting the base64 string
    base64_image = encode_image(image_path)

    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]
    
    class Step(BaseModel):
        question: str
        answer: str
        # suggestion: str
        final_verdict: str

    if validation_type == "collision" or validation_type == "semantics_multi_hop":    
        class MathReasoning(BaseModel):
            steps: List[Step]  # Use List instead of list
    elif validation_type == "semantics":
        class MathReasoning(BaseModel):
            segments: List[str]
            steps: List[Step]  # Use List instead of list

    
    client = OpenAI(api_key = api_key)
    
    kwargs = {
        "model": vlm_type,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
                    }],
            },
        ],
        "response_format": MathReasoning,
    }
    
    if "-4o" in vlm_type:
        kwargs["max_tokens"] = 10000
    else:
        kwargs["max_completion_tokens"] = 10000

    completion = client.beta.chat.completions.parse(**kwargs)
    
    validator_response_fail = []

    for step in completion.choices[0].message.parsed.steps:
        
        final_verdict = step.final_verdict.lower()
        
        if final_verdict == "fail":
            validator_response_fail.append(step)
        else:
            pass


    validator_response_full = completion.choices[0].message.parsed.steps

    return validator_response_fail, validator_response_full

def VLM_output_saver(prompt, datum_id, q_or_a, folder_name):
    """Function to save the reasoning output of the VLM and the prompts

    Args:
        prompt (str): textual prompts
        datum_id (int): Unique id given to each image-instruction pair
        iter (int): pipeline iteration number for the current datum
        q_or_a (str): Prompt or VLM output flag
        folder_name (str): Name of the folder to save
    """
    crnt_file_path = os.path.dirname(os.path.abspath(__file__))
    output_path = crnt_file_path + '/../../' + folder_name
    
    pth = output_path + "pipeline_outputs/" + datum_id + "/VLM_output.txt"
    VLM_output_file = open(pth, 'a')
    VLM_output_file.write(q_or_a + ":\n\n")
    VLM_output_file.write(str(prompt) + "\n\n")
    VLM_output_file.write("-------------------------------------" + "\n\n")

def prompt_loader(pipeline_stage, params, output_shape):
    """Function to load the prompt from a yaml file

    Args:
        pipeline_stage (str): stage of the pipeline
        params (dict): parameters to construct the prompt
        output_shape: polygon/points

    Returns:
        str: prompt for the LLM/VLM
    """
    prompt_path = prompt_path = os.path.dirname(os.path.abspath(__file__))+ \
            f"/../../config/prompts_{output_shape}.yaml"

    with open(prompt_path) as stream:
        try:
            prompts_all = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    prompt_format = prompts_all[pipeline_stage]
    locals().update(params)
    prompts = eval(prompt_format)
    return prompts