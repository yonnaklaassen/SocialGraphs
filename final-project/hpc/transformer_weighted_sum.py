from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import json
from scipy.spatial.distance import cdist

# Load the model
model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

# Helper function to normalize weights
def normalize_weights(weights):
    weights = np.array(weights, dtype=np.float32)
    return weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    tokens = [word for word in tokens if word != "br"]
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

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
    # Prepare professor embeddings with weighted averaging
    course_df = pd.read_csv("data/course_df.csv")

    with open("data/all_professors.json", "r") as file:
        professors_data = json.load(file)

    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

    # Combine course data into a single textual input
    course_texts = {
        row["COURSE"]: " ".join([
            row["COURSE_DESCRIPTION"],
            row.get("LEARNING_OBJECTIVES", ""),
            row.get("COURSE_CONTENT", "")
        ])
        for course_name, row in course_df.iterrows()
    }

    courses_text_processed = {}

    for course_name, text in course_texts.items():
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        courses_text_processed[course_name] = preprocessed_text

    professor_embeddings = []
    c = 0
    for prof_name, prof_data in professors_data.items():
        # Start with the Profile_desc embedding (or a zero vector if empty)
        profile_desc = prof_data.get("Profile_desc", "").strip()
        if profile_desc:
            profile_desc = preprocess_text(profile_desc)
            profile_embedding = model.encode(profile_desc, convert_to_tensor=True)
        else:
            profile_embedding = torch.zeros((model.get_sentence_embedding_dimension(),), device='cuda')
        
        # Generate embeddings for Fingerprint concepts with weights
        fingerprint_weights = []
        fingerprint_embeddings = []
        for fp in prof_data.get("Fingerprint", []):
            concept = fp['Concept']
            concept = preprocess_text(concept)
            weight = float(fp['Value'].strip('%')) if fp['Value'] else 0
            fingerprint_weights.append(weight)
            fingerprint_embeddings.append(model.encode(concept, convert_to_tensor=True))
        
        # Normalize and generate embeddings for scholia topics
        scholia_weights = []
        scholia_embeddings = []
        normalized_scores = normalize_scholia_scores(prof_data.get("scholia_topics", []))
        for idx, topic in enumerate(prof_data.get("scholia_topics", [])):
            concept = topic['topic']
            weight = normalized_scores[idx]
            scholia_weights.append(weight)
            scholia_embeddings.append(model.encode(concept, convert_to_tensor=True))
        
        # Normalize weights
        all_weights = normalize_weights([1] + fingerprint_weights + scholia_weights)
        all_embeddings = [profile_embedding] + fingerprint_embeddings + scholia_embeddings
        
        # Compute weighted average embedding
        weighted_embedding = torch.sum(torch.stack([w * emb for w, emb in zip(all_weights, all_embeddings)]), dim=0)
        professor_embeddings.append(weighted_embedding)

        c+=1

        print(f"Professor embeddings: {c}/{len(professors_data)}")

    # Prepare course embeddings
    course_embeddings = model.encode(list(courses_text_processed.values()), convert_to_tensor=True)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(
        torch.stack(professor_embeddings).cpu().numpy(),
        course_embeddings.cpu().numpy()
    )

    # Create a DataFrame
    professor_names = list(professors_data.keys())
    course_names = list(courses_text_processed.keys())
    similarity_llm_approach_1 = pd.DataFrame(similarity_matrix, index=professor_names, columns=course_names)


    # Optional: Save the DataFrame for further analysis
    similarity_llm_approach_1.to_csv("data/similarity_cosine_transformer_weighted_sum.csv", index=True)

    # Compute Euclidean distance
    euclidean_distance_matrix = cdist(
        torch.stack(professor_embeddings).cpu().numpy(),
        course_embeddings.cpu().numpy(),
        metric='euclidean'
    )

    # Convert to similarity (smaller distances imply greater similarity)
    euclidean_similarity_matrix = 1 / (1 + euclidean_distance_matrix)

    # Create a DataFrame
    similarity_llm_euclidean = pd.DataFrame(euclidean_similarity_matrix, index=professor_names, columns=course_names)

    # Save the DataFrame
    similarity_llm_euclidean.to_csv("data/similarity_euclidian_transformer_weighted_sum.csv", index=True)

    # Compute Manhattan distance
    manhattan_distance_matrix = cdist(
        torch.stack(professor_embeddings).cpu().numpy(),
        course_embeddings.cpu().numpy(),
        metric='cityblock'  # L1 norm
    )

    # Convert to similarity (smaller distances imply greater similarity)
    manhattan_similarity_matrix = 1 / (1 + manhattan_distance_matrix)

    # Create a DataFrame
    similarity_llm_manhattan = pd.DataFrame(manhattan_similarity_matrix, index=professor_names, columns=course_names)

    # Save the DataFrame
    similarity_llm_manhattan.to_csv("data/similarity_manhattan_transformer_weighted_sum.csv", index=True)



if __name__ == "__main__":
    main()