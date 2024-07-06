import openai
import json

from app.glance import Corpus, normalize_activations, get_active_sections, active_sections_across_samples


def feature_representation(feature, tokenizer):
    sections = active_sections_across_samples(feature['samples'], tokenizer)

    sample_description = ""
    for word_tokens, norm_activations in sections:
        sample_description += f"----> "

        for token_word, activation in zip(word_tokens, norm_activations):
            sample_description += token_word
            if activation > 0.0:
                sample_description += f"|{round(activation.item(), 2)}|"
            sample_description += " "
        
        sample_description += "\n"

    return sample_description

def get_prompt(sample_description):
    prompt = f"""The following are text samples which were passed into a LLM regression model. This model is trained to always detects a certain feature in the text. Your job is to determine which feature it is and describe that feature in 50 words or less. The model is not always coherent. Sometimes it fails to identify a feature coherently. Here are the samples:

{sample_description}

What feature is the model detecting? Give the feature a name, describe it in 50 words or less and give it a rating between 1.0 and 0.0 for coherence (1.0 is highly coherent). State your assessment as JSON in the following format: """ + '{"feature_name": string, "feature_description": string, "coherence": float}\n'

    return prompt

def describe_feature(sample_description):
    prompt = get_prompt(sample_description)
    response = openai.completions.create(
        response_format={ "type": "json_object" },
        model="gpt-4o",  
        messages=[
            { "role": "system", "content": prompt }
        ],  
        max_tokens=100
    )

    return json.loads(response.choices[0].message.content)

def assess():
    with open('.credentials.json') as f:
        credentials = json.load(f)

    openai.api_key = credentials['openai_api_key']

    dataset = 'data/gpt2-imdb-pos'
    c = Corpus(dataset)

    ft_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    features = c.features_by_idx(ft_indices, samples_per_feature=10)

    assessment = []
    for f in features:
        descriptions = []
        machine_readable_representation = feature_representation(f)
        description = describe_feature(machine_readable_representation)

        descriptions.append(description)

        assessment.append({
            'feature': f.index,
            'descriptions': descriptions,
            'machine_representation': machine_readable_representation
        })

    return assessment

def assess_and_save(filename):
    assessment = assess()

    with open(filename, 'w') as f:
        json.dump(assessment, f)
