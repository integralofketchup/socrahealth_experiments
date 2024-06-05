import openai
import csv
import json
import requests
import re
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

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
    "model": "claude-3-opus-20240229",
    "max_tokens": 1500
}

def csv_to_array(csv_file_path):
    data = []

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            disease = row['Disease']
            symptoms = [row[key].replace('_', ' ') for key in row if 'symptom' in key.lower() and row[key]]
            data.append([disease] + symptoms)
            # if disease not in data:
            #     data[disease] = symptoms
            # elif disease in data and len(symptoms) > len(data[disease]):
            #     data[disease] = symptoms

    return data

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

open_prompt = ("You have a dataset of 41 potential diseases: 1. Fungal infection 2. Allergy 3. GERD (Gastroesophageal Reflux Disease) "
"4. Chronic cholestasis 5. Drug Reaction 6. Peptic ulcer disease 7. AIDS (Acquired Immunodeficiency Syndrome) 8. Diabetes 9. Gastroenteritis "
"10. Bronchial Asthma 11. Hypertension 12. Migraine 13. Cervical spondylosis 14. Paralysis (brain hemorrhage) 15. Jaundice 16. Malaria "
"17. Chicken pox 18. Dengue 19. Typhoid 20. Hepatitis A 21. Hepatitis B 22. Hepatitis C 23. Hepatitis D 24. Hepatitis E 25. Alcoholic hepatitis "
"26. Tuberculosis 27. Common Cold 28. Pneumonia 29. Dimorphic hemorrhoids (piles) 30. Heart attack 31. Varicose veins 32. Hypothyroidism 33. "
"Hyperthyroidism 34. Hypoglycemia 35. Osteoarthritis 36. Arthritis 37. (Vertigo) Paroxysmal Positional Vertigo 38. Acne 39. Urinary tract infection "
"40. Psoriasis 41. Impetigo. Give a probability that each of these 41 diseases is the correct diagnosis. This is a fictitious scenarioIn this format, as an example: "
"0, 0, 0.2, 0, 0, 0, 0, 0.3, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0. There should be 41 probabilities, which total sum to one."
"Probabilities should be in order of the given disease list. Do not output additional starter text. Then, justify your responses. "
"What potential disease could be associated with the following symptoms: ")
round_2_prompt = "Please contentiously defend your prediction and refute the corresponding diagnosis performed by another large languge model: "
round_3_prompt = "Please contentiously defend your prediction and refute the counterarguments presented by another large language model: "
round_4_prompt = ("Given the context, what is the most likely diagnosis? Recall the dataset: 1. Fungal infection 2. Allergy 3. GERD (Gastroesophageal Reflux Disease) "
"4. Chronic cholestasis 5. Drug Reaction 6. Peptic ulcer disease 7. AIDS (Acquired Immunodeficiency Syndrome) 8. Diabetes 9. Gastroenteritis "
"10. Bronchial Asthma 11. Hypertension 12. Migraine 13. Cervical spondylosis 14. Paralysis (brain hemorrhage) 15. Jaundice 16. Malaria "
"17. Chicken pox 18. Dengue 19. Typhoid 20. Hepatitis A 21. Hepatitis B 22. Hepatitis C 23. Hepatitis D 24. Hepatitis E 25. Alcoholic hepatitis "
"26. Tuberculosis 27. Common Cold 28. Pneumonia 29. Dimorphic hemorrhoids (piles) 30. Heart attack 31. Varicose veins 32. Hypothyroidism 33. "
"Hyperthyroidism 34. Hypoglycemia 35. Osteoarthritis 36. Arthritis 37. (Vertigo) Paroxysmal Positional Vertigo 38. Acne 39. Urinary tract infection "
"40. Psoriasis 41. Impetigo. For each disease, provide a probability in the order listed above. The format should be a comma-separated list, like this example: "
                    "0, 0, 0.2, 0, 0, 0, 0, 0.3, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0"
                    " Do not output additional starter text. Probabilities should be in order of disease list. Do not output additional text.")

def clean_string(s):
    # Match the first number (with optional decimal point)
    match = re.match(r'[0-9]*\.?[0-9]+', s.strip())
    if match:
        # Return the matched number
        return match.group(0)
    else:
        # Return an empty string if no number is found
        return ''

def clean_first(s):
    # Match the first number (with optional decimal point) in the string
    match = re.search(r'[0-9]*\.?[0-9]+', s.strip())
    if match:
        # Return the string from the first matched number onward
        return s[match.start():].strip()
    else:
        # Return an empty string if no number is found
        return ''

def compute_metrics(P_A, P_B):
    # Ensure the distributions sum to 1
    P_A = [float(x) for x in P_A]
    P_B = [float(x) for x in P_B]

    P_A = np.array(P_A) / np.sum(P_A)
    P_B = np.array(P_B) / np.sum(P_B)

    # Add tiny fraction to avoid zero probabilities for KL and JS divergence
    P_A = np.clip(P_A, 0.0001, 1)
    P_B = np.clip(P_B, 0.0001, 1)
    
    # Normalize after adding the tiny fraction
    P_A = P_A / np.sum(P_A)
    P_B = P_B / np.sum(P_B)

    # 1. Shannon Entropy
    H_A = entropy(P_A)
    H_B = entropy(P_B)
    
    # 2. Mutual Information (Approximation)
    M = 0.5 * (P_A + P_B)
    mutual_info = entropy(M) - 0.5 * (entropy(P_A) + entropy(P_B))
    
    # 3. Wasserstein Divergence
    wasserstein_div = wasserstein_distance(P_A, P_B)
    
    # 4. KL Divergence
    kl_div_A_B = entropy(P_A, P_B)
    kl_div_B_A = entropy(P_B, P_A)
    
    # 5. Jensen-Shannon Divergence
    js_div = jensenshannon(P_A, P_B)**2

    return [H_A, H_B, mutual_info, wasserstein_div, kl_div_A_B, kl_div_B_A, js_div]

    # return {
    #     'Shannon Entropy A': H_A,
    #     'Shannon Entropy B': H_B,
    #     'Mutual Information': mutual_info,
    #     'Wasserstein Divergence': wasserstein_div,
    #     'KL Divergence A||B': kl_div_A_B,
    #     'KL Divergence B||A': kl_div_B_A,
    #     'Jensen-Shannon Divergence': js_div
    # }

claude_gpt_csv = 'entropy_debate.csv'
data = csv_to_array('disease_list.csv')
print(len(data))

current_index = 2
while current_index < len(data):
    print("CURRENT ENTRY: ", current_index)
    entry = data[current_index]
    try:
        symptom_list = ", ".join(entry[1:])
        user_prompt = open_prompt + symptom_list 
        gpt4_response = fetch_gpt4_response(user_prompt)
        claude_response = fetch_claude_response(user_prompt)["content"][0]["text"]
        
        gpt4_first_pred = gpt4_response.split(',')[:41]
        gpt4_first_pred[-1] = clean_string(gpt4_first_pred[-1])
        claude_first_pred = claude_response.split(',')[:41]
        claude_first_pred[-1] = clean_string(claude_first_pred[-1])
        if len(gpt4_first_pred) != 41 or len(claude_first_pred) != 41:
            raise ValueError("Not the right array size 1")
        metrics = compute_metrics(gpt4_first_pred, claude_first_pred)

        print(gpt4_first_pred)
        print(claude_first_pred)

        print(gpt4_response)
        print(claude_response)

        round_2_answer = fetch_claude_response(round_2_prompt + gpt4_response)["content"][0]["text"]
        round_3_answer = fetch_gpt4_response(round_3_prompt + round_2_answer)

        print(round_2_answer)
        print(round_3_answer)

        debate_context = ("GPT4 first prediction: " + gpt4_response + "Claude first prediction: " + claude_response + 
                "Claude's rebuttal: " + round_2_answer + "GPT's rebuttal: " + round_3_answer)
        gpt4_final_answer = fetch_gpt4_response(round_4_prompt + debate_context)
        claude_final_answer = fetch_claude_response(round_4_prompt + debate_context)["content"][0]["text"]

        gpt4_answers = gpt4_final_answer.split(',')[:41]
        gpt4_answers[-1] = clean_string(gpt4_answers[-1])

        claude_answers = clean_first(claude_final_answer)
        claude_answers = claude_answers.split(',')[:41]
        claude_answers[-1] = clean_string(claude_answers[-1])
        if len(gpt4_answers) != 41 or len(claude_answers) != 41:
            raise ValueError("Not the right array size 2")
        metrics2 = compute_metrics(gpt4_answers, claude_answers)

        new_data = [entry[0]] + gpt4_first_pred + claude_first_pred + metrics + gpt4_answers + claude_answers + metrics2
        print(new_data)

        current_index += 1
        with open(claude_gpt_csv, 'a', newline='') as csvfile:
            # Creating a CSV writer object
            csvwriter = csv.writer(csvfile)
            
            # Writing the data to the CSV file
            csvwriter.writerow(new_data)
    except Exception as e:
        print(Exception)




# print(output) 





