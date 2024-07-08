from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import json
from tqdm import tqdm
from transformers import GPT2Tokenizer

from app.glance import Corpus, active_sections_across_samples


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

    return sample_description, [(word_tokens, norm_activations.tolist()) for word_tokens, norm_activations in sections]

def get_prompt(sample_description):
    prompt = f"""The following are text samples which were passed into a LLM regression model. This model is trained to always detects a certain feature in the text, though it may be faulty and pick up an incoherent feature. The model will activate 0.0 for most tokens, indicating that the feature is not present in that token. When it does activate, the intensity of the activation (between 0.0 and 1.0) will be printed after the token, e.g. "apples and|0.3| pears in|0.1| the gard en" is a sentence where the words "and" and "in" were assessed as having the feature. Your job is to determine which feature is being detected is and describe that feature in 50 words or less. The model is not always coherent. Sometimes it fails to identify a feature coherently. Here are the samples:

{sample_description}

What feature is the model detecting? Is the feature coherent at all? Is the feature complex or simple? Give the feature a name, describe it in 50 words or less and give it a rating between 1.0 and 0.0 for coherence (1.0 is highly coherent). Also give a rating between 1.0 and 0.0 for complexity (1.0 is a very rich, sophisticated feature). State your assessment as JSON in the following format: """ + '{"feature_name": string, "feature_description": string, "feature_coherence": float, "feature_complexity": float }\n'

    return prompt

def describe_feature(sample_description, id):
    prompt = get_prompt(sample_description)
    response = openai.chat.completions.create(
        response_format={ "type": "json_object" },
        model="gpt-4o",  
        messages=[
            { "role": "system", "content": prompt }
        ],  
        max_tokens=100
    )
    
    return json.loads(response.choices[0].message.content), id

def assess(data_dir, n_feats_to_analyse, samples_per_feature):
    with open('.credentials.json') as f:
        credentials = json.load(f)

    openai.api_key = credentials['OPENAI_API_KEY']

    c = Corpus(data_dir)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    ft_indices = list(range(n_feats_to_analyse))

    features = c.features_by_idx(ft_indices, samples_per_feature=samples_per_feature)

    assessment = {}
    for f in features:
        machine_readable_representation, sections = feature_representation(f, tokenizer)

        assessment[f['index']] ={
            'feature': f['index'],
            'machine_representation': machine_readable_representation,
            'raw_info': sections,
        }

    description_futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for f in assessment.values():
            description_futures.append(executor.submit(describe_feature, f['machine_representation'], f['feature']))

    for future in tqdm(as_completed(description_futures), desc="Getting human descriptions", total=len(description_futures)):
        result, id = future.result()
        assessment[id]['human_description'] = result


    return list(assessment.values())

def llm_assessment(data_dir, filename, n_feats_to_analyse, samples_per_feature):
    assessment = assess(data_dir, n_feats_to_analyse, samples_per_feature)

    with open(filename, 'w') as f:
        json.dump(assessment, f, indent=4)

if __name__ == '__main__':
    llm_assessment('data/small', 'cruft/small-comp.json')