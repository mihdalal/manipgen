import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

API_KEY = ""   # Paste your OpenAI API key here

def np_to_b64(img: np.ndarray) -> str:
    img = Image.fromarray(img)
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64

def create_image_content(ims):
    img_urls = [np_to_b64(im) for im in ims]
    img_content = []
    for i in range(len(img_urls)):
        img_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"{img_urls[i]}",
                "detail": "high",
            }
        })
    return img_content

def get_headers_and_payload():
    if API_KEY == "":
        raise ValueError("Please paste your OpenAI API key in 'manipgen/real_world/vlm_planning/gpt_utils.py'.")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [],
        "max_tokens": 4096,
        "top_p":1,
        "seed":123,
    }
    
    return headers, payload

def prompt_gpt4v(text, ims=None, agent_type='workspace'):
    if agent_type == 'workspace':
        from manipgen.real_world.vlm_planning.prompts.workspace_prompt import SYSTEM_PROMPT, USER_EXAMPLES, ASSISTANT_EXAMPLES
    elif agent_type == 'tagging':
        from manipgen.real_world.vlm_planning.prompts.tagging_prompt import SYSTEM_PROMPT, USER_EXAMPLES, ASSISTANT_EXAMPLES
    elif agent_type == 'planning':
        from manipgen.real_world.vlm_planning.prompts.planning_prompt import SYSTEM_PROMPT, USER_EXAMPLES, ASSISTANT_EXAMPLES
    elif agent_type == 'segmentation_doublechecker':
        from manipgen.real_world.vlm_planning.prompts.segmentation_doublechecker_prompt import SYSTEM_PROMPT, USER_EXAMPLES, ASSISTANT_EXAMPLES
        
    headers, payload = get_headers_and_payload()
    if ims is not None:
        img_content = create_image_content(ims)
    payload["messages"] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]
    for user, assistant in zip(USER_EXAMPLES, ASSISTANT_EXAMPLES):
        user_imgs = user[1]
        user_text = user[0]
        if user_text is not None:
            payload["messages"] += [
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        if user_imgs is not None:
            payload["messages"] += [
                {
                    "role": "user",
                    "content": create_image_content(user_imgs)
                }
            ]
        if assistant is not None:
            payload["messages"] += [
                {
                    "role": "assistant",
                    "content": assistant
                }
            ]
    if text is not None:
        payload["messages"] += [
            {
                "role": "user",
                "content": text
            }
        ]
    if ims is not None:
        payload["messages"] += [
            {
                "role": "user",
                "content": img_content
            }
        ]
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json()["choices"][0]["message"]["content"])
    response_dict = eval(response.json()["choices"][0]["message"]["content"].split("```")[1].replace("\n", "").replace("json", ""))['output']
    return response_dict
    
def prompt_gpt4v_tagging(ims):
    headers, payload = get_headers_and_payload()
    img_content = create_image_content(ims)
    
    payload["messages"] = [
        {
            "role": "system",
            "content": open("real_world/vlm_planning/prompts/tagging_prompt.py", "r").read().replace("\n", "")
        },
    ]
    payload["messages"] += [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tag the following images:"
                }
            ] + img_content
        }
    ]

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def prompt_gpt4v_planning(language_instruction, ims, tags):
    headers, payload = get_headers_and_payload()
    img_content = create_image_content(ims)
    
    payload["messages"] = [
        {
            "role": "system",
            "content": open("real_world/vlm_planning/prompts/planning_prompt.py", "r").read().replace("\n", "")
        }
    ]
    # TODO: add in-context examples here
    payload["messages"] += [
        {
            "role": "system",
            "content": open("real_world/vlm_planning/prompts/planning_examples.py", "r").read().replace("\n", "")
        }
    ]
    
    payload["messages"] += [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is a list of objects in the scene, but you can also use the images to identify more objects: {tags}"
                },
            ]
        }
    ]
    payload["messages"] += [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is a list of objects in the scene, but you can also use the images to identify more objects: {tags}"
                },
                {
                    "type": "text",
                    "text": f"Your task is: {language_instruction}. REMEMBER, only return a list of the form [('object', 'skill'), ...] where 'object' is the object to interact with and 'skill' is the skill to perform.",
                }
            ] + img_content
        }
    ]

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def prompt_gpt4v_workspace(skill, obj, ims):
    headers, payload = get_headers_and_payload()
    img_content = create_image_content(ims)
    
    task_description = f"picking {obj}" if skill == "pick" else f"placeing something in {obj}"
    
    payload["messages"] = [
        {
            "role": "system",
            "content": open("real_world/vlm_planning/prompts/workspace_prompt.py", "r").read().replace("\n", "")
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please classify whether {task_description} is in table-top setting. REMEMBER, only return [True] or [False]."
                }
            ] + img_content
        }
    ]

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
    

if __name__ == "__main__":
    response = prompt_gpt4v_workspace("pick", "cup", [np.random.rand(224, 224, 3), np.random.rand(224, 224, 3)])
    print(response)
    