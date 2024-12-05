from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy.spatial.distance import cdist
import json

# Function to normalize scholia scores
def normalize_scholia_scores(scholia_topics):
    # Extract raw scores
    scores = np.array([float(topic.get('score', 0)) for topic in scholia_topics])
    
    if len(scores) == 0:  # Handle case with no scholia topics
        return []

    # Determine normalization range
    max_score = scores.max()
    normalization_max = max(max_score, 100)  # Use max(max_score, 100) as the upper bound

    # Normalize to [0, normalization_max]
    normalized_scores = (scores / normalization_max) * 100
    return normalized_scores

def main():
    # Load the data
    course_df = pd.read_csv("data/course_df.csv")

    with open("data/all_professors.json", "r") as file:
        professors_data = json.load(file)

    # Load the transformer model
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

    # Prepare professor embeddings (reuse your code for weighted text)
    professor_embeddings = []
    professor_names = list(professors_data.keys())
    
    for prof_name, prof_data in professors_data.items():
        # Start with Profile_desc
        profile_desc = prof_data.get("Profile_desc", "").strip()
        weighted_text = profile_desc if profile_desc else ""

        # Add Fingerprint concepts, weighted by their percentages
        for fp in prof_data.get("Fingerprint", []):
            concept = fp['Concept']
            weight = int(float(fp['Value'].strip('%')) if fp['Value'] else 0)
            weighted_text += " " + (" ".join([concept] * weight))

        # Add scholia topics, weighted by their normalized scores
        normalized_scores = normalize_scholia_scores(prof_data.get("scholia_topics", []))
        for idx, topic in enumerate(prof_data.get("scholia_topics", [])):
            concept = topic['topic']
            weight = int(normalized_scores[idx])
            weighted_text += " " + (" ".join([concept] * weight))
        
        # Encode the weighted text
        embedding = model.encode(weighted_text, convert_to_tensor=True)
        professor_embeddings.append(embedding)

    # Stack professor embeddings for pairwise similarity
    professor_embeddings_tensor = torch.stack(professor_embeddings)
    professor_embeddings_np = professor_embeddings_tensor.cpu().numpy()

    # Compute similarity matrix between professors
    professor_similarity_matrix = cosine_similarity(professor_embeddings_np)


    # Create a DataFrame for readability
    similarity_df = pd.DataFrame(professor_similarity_matrix, index=professor_names, columns=professor_names)

    # Save the professor similarity DataFrame
    similarity_df.to_csv("data/professor_cosine_weighted_text.csv", index=True)

    # Compute Euclidean distance
    euclidean_distance_matrix = cdist(
        professor_embeddings_np,
        professor_embeddings_np,
        metric='euclidean'
    )

    # Convert to similarity (smaller distances imply greater similarity)
    euclidean_similarity_matrix = 1 / (1 + euclidean_distance_matrix)

    # Create a DataFrame
    similarity_llm_euclidean = pd.DataFrame(euclidean_similarity_matrix, index=professor_names, columns=professor_names)

    # Save the DataFrame
    similarity_llm_euclidean.to_csv("data/professor_euclidian_transformer_weighted_text.csv", index=True)

    # Compute Manhattan distance
    manhattan_distance_matrix = cdist(
        professor_embeddings_np,
        professor_embeddings_np,
        metric='cityblock'  # L1 norm
    )

    # Convert to similarity (smaller distances imply greater similarity)
    manhattan_similarity_matrix = 1 / (1 + manhattan_distance_matrix)

    # Create a DataFrame
    similarity_llm_manhattan = pd.DataFrame(manhattan_similarity_matrix, index=professor_names, columns=professor_names)

    # Save the DataFrame
    similarity_llm_manhattan.to_csv("data/professor_manhattan_transformer_weighted_text.csv", index=True)

if __name__ == "__main__":
    main()