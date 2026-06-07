# AI Recommendation Engine: The Digital Matchmaker 🤖🤝

## Description
This project is part of my Artificial Intelligence Internship at DecodeLabs (Project 3). It focuses on solving the real-world "Choice Overload" problem by building a Content-Based Recommendation System from scratch. 

Instead of relying on random suggestions, this "Digital Matchmaker" acts as a prediction engine. It ingests a user's top tech skills and maps them against a dataset of tech job roles. By utilizing TF-IDF Vectorization and Cosine Similarity, the engine calculates the exact angular alignment between the user's profile and job attributes to recommend the most optimal career paths.

## Key Features
- **Content-Based Filtering:** Recommends items based on item attributes and user preferences.
- **Mathematical Similarity Engine:** Uses **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text to vectors and **Cosine Similarity** to measure the distance between them.
- **Top-N Filtering:** Prevents choice overload by truncating the output to the Top 3 most relevant matches with their alignment percentages.
- **Failsafe Data Generation:** Automatically generates a sample `raw_skills.csv` dataset if the original file is missing, ensuring the pipeline never breaks.

## Technologies Used
- **Python**
- **Pandas** (Data Handling & DataFrame manipulation)
- **Scikit-Learn** (`TfidfVectorizer`, `cosine_similarity`)
- **NumPy**

## How to Run the Project
1. Ensure Python is installed on your system.
2. Install the required dependencies via terminal:
   ```bash
   pip install pandas scikit-learn numpy
