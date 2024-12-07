import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import string
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')


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

# Function to preprocess and extract concepts from text
def extract_keywords(text):
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Lowercase and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def main():
    course_df = pd.read_csv("data/course_df.csv")

    with open("data/all_professors.json", "r") as file:
        professors_data = json.load(file)

    model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

    # Combine professor data into a single textual input
    professor_texts = {
        prof_name: " ".join([
            prof_data.get("Profile_desc", ""),
            " ".join(prof_data.get("Keywords", [])),
            " ".join([fp['Concept'] for fp in prof_data.get("Fingerprint", [])]),
            " ".join([topic['topic'] for topic in prof_data.get("scholia_topics", [])])
        ])
        for prof_name, prof_data in professors_data.items()
    }

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

    professors_text_processed = {}
    for prof_name, text in professor_texts.items():
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        professors_text_processed[prof_name] = preprocessed_text

    # Generate embeddings
    professor_embeddings = model.encode(list(professors_text_processed.values()), convert_to_tensor=True)
    course_embeddings = model.encode(list(courses_text_processed.values()), convert_to_tensor=True)

    # Compute similarity
    similarity_matrix = cosine_similarity(professor_embeddings.cpu().numpy(), course_embeddings.cpu().numpy())

    # Create a DataFrame for readability

    similarity_llm = pd.DataFrame(similarity_matrix, index=professors_text_processed.keys(), columns=courses_text_processed.keys())


    # save the similarity matrix to a csv file
    similarity_llm.to_csv("data/similarity_cosine_transformer.csv", index=True)
    # Compute Euclidean distance
    euclidean_distance_matrix = cdist(
        professor_embeddings.cpu().numpy(),
        course_embeddings.cpu().numpy(),
        metric='euclidean'
    )

    # Convert to similarity (smaller distances imply greater similarity)
    euclidean_similarity_matrix = 1 / (1 + euclidean_distance_matrix)

    # Create a DataFrame
    similarity_llm_euclidean = pd.DataFrame(euclidean_similarity_matrix, index=professors_text_processed.keys(), columns=courses_text_processed.keys())

    # Save the DataFrame
    similarity_llm_euclidean.to_csv("data/similarity_euclidian_transformer.csv", index=True)

    # Compute Manhattan distance
    manhattan_distance_matrix = cdist(
        professor_embeddings.cpu().numpy(),
        course_embeddings.cpu().numpy(),
        metric='cityblock'  # L1 norm
    )

    # Convert to similarity (smaller distances imply greater similarity)
    manhattan_similarity_matrix = 1 / (1 + manhattan_distance_matrix)

    # Create a DataFrame
    similarity_llm_manhattan = pd.DataFrame(manhattan_similarity_matrix, index=professors_text_processed.keys(), columns=courses_text_processed.keys())

    # Save the DataFrame
    similarity_llm_manhattan.to_csv("data/similarity_manhattan_transformer.csv", index=True)

if __name__ == "__main__":
    main()