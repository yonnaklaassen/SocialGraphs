from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from scipy.spatial.distance import cdist

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


def main():
    # Prepare professor embeddings with weighted text

    course_df = pd.read_csv("data/course_df.csv")

    with open("data/all_professors.json", "r") as file:
        professors_data = json.load(file)

    # Load the model
    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

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
    for prof_name, prof_data in professors_data.items():
        # Start with Profile_desc
        profile_desc = prof_data.get("Profile_desc", "").strip()
        weighted_text = profile_desc if profile_desc else ""

        # Add Fingerprint concepts, weighted by their percentages
        for fp in prof_data.get("Fingerprint", []):
            concept = fp['Concept']
            concept = preprocess_text(concept)
            weight = int(float(fp['Value'].strip('%')) if fp['Value'] else 0)
            weighted_text += " " + (" ".join([concept] * weight))

        # Add scholia topics, weighted by their normalized scores
        normalized_scores = normalize_scholia_scores(prof_data.get("scholia_topics", []))
        for idx, topic in enumerate(prof_data.get("scholia_topics", [])):
            concept = topic['topic']
            concept = preprocess_text(concept)
            weight = int(normalized_scores[idx])
            weighted_text += " " + (" ".join([concept] * weight))
        
        # Encode the weighted text
        embedding = model.encode(weighted_text, convert_to_tensor=True)
        professor_embeddings.append(embedding)

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
    similarity_llm_approach_2 = pd.DataFrame(similarity_matrix, index=professor_names, columns=course_names)


    similarity_llm_approach_2.to_csv("data/similarity_transformer_weighted_text.csv", index=True)

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
    similarity_llm_euclidean.to_csv("data/similarity_euclidian_transformer_weighted_text.csv", index=True)

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
    similarity_llm_manhattan.to_csv("data/similarity_manhattan_transformer_weighted_text.csv", index=True)


if __name__ == "__main__":
    main()
