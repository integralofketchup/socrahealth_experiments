import openai
import csv
import json
import requests

#GPT Parameters
openai.api_key = ""

#Claude Parameters
API_ENDPOINT = "https://api.anthropic.com/v1/messages"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "",
    "anthropic-version": "2023-06-01"
}

REQUEST_DATA = {
    "messages": [
        {"role": "user", "content": ""}
    ],
    "model": "claude-3-haiku-20240307",
    "max_tokens": 1500
}

def csv_to_array(csv_file_path):
    data = []

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            disease = row['Disease']
            symptoms = [row[key].replace('_', ' ') for key in row if key != 'Disease' and row[key]]
            data.append([disease] + symptoms)

    return data

data = csv_to_array('dataset.csv')

def fetch_gpt4_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def fetch_claude_response(prompt):
    REQUEST_DATA["messages"][0]["content"] = prompt
    # REQUEST_DATA["prompt"] = "\n\nHuman:" + prompt + "\n\nAssistant:"
    response = requests.post(API_ENDPOINT, headers=HEADERS, json=REQUEST_DATA)

    if response.status_code == 200 or response.status_code == 400:
        result = response.json()
        generated_text = result
        # print(generated_text)
        return generated_text
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None

open_prompt = "What potential disease could be associated with the following symptoms:"
end_prompt = ("Please offer one prediction, supported by justifications. Additionally, include a list of "
            "supplementary symptom inquiries and recommend relevant lab tests to strengthen the confidence in your prediction.")
round_2_prompt = "Please defend your prediction and refute the corresponding diagnosis performed by another large languge model: "
round_3_prompt = "Please defend your prediction and refute the counterarguments presented by another large language model: "
round_4_prompt = ("Given the debate context, what is the most likely diagnosis. Furthermore, can you "
                "recommend additional symptom inquiries and lab tests. Context: ")

def test_claude():
    response = fetch_claude_response("hello")
    print(response)

def test_debate():
    symptom_list = ", ".join(data[0][1:])
    user_prompt = open_prompt + symptom_list + end_prompt
    gpt4_response = fetch_gpt4_response(user_prompt)
    claude_response = fetch_claude_response(user_prompt)["content"][0]["text"]
    print(f"GPT-4 response: {gpt4_response}\n")
    print(f"Claude response: {claude_response}\n")

    #Asking GPT to refute
    round_2_answer = fetch_gpt4_response(round_2_prompt + claude_response)
    #Asking Claude to refute
    round_3_answer = fetch_claude_response(round_3_prompt + round_2_answer)["content"][0]["text"]
    print(f"GPT-4 response: {round_2_answer}\n")
    print(f"Claude response: {round_3_answer}\n")

    #Asking for consensus from both LLMs
    debate_context = ("GPT4 first prediction: " + gpt4_response + "Claude first prediction: " + claude_response + 
                    "GPT4's rebuttal: " + round_2_answer + "Claude's rebuttal: " + round_3_answer)
    gpt4_final_answer = fetch_gpt4_response(round_4_prompt + debate_context)
    claude_final_answer = fetch_claude_response(round_4_prompt + debate_context)["content"][0]["text"]
    print(f"GPT-4 response: {gpt4_final_answer}\n")
    print(f"Claude response: {claude_final_answer}\n")

test_debate()

data = []
for d in data:
    symptom_list = ", ".join(d[1:])
    user_prompt = open_prompt + symptom_list + end_prompt
    gpt4_response = fetch_gpt4_response(user_prompt)
    claude_response = fetch_claude_response(user_prompt)