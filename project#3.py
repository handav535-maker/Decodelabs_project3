import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# STAGE 0: DATA PREPARATION (Failsafe Mechanism)
# =====================================================================
def check_and_create_data():
    """
    Creates a dummy dataset if 'raw_skills.csv' is missing in the folder.
    This ensures the pipeline never crashes due to missing files.
    """
    file_name = 'raw_skills.csv'
    if not os.path.exists(file_name):
        print(f"[INFO] '{file_name}' not found. Creating a sample dataset for testing...")
        data = {
            'Job_Role': ['Data Scientist', 'Frontend Developer', 'Backend Developer', 
                         'Cloud Architect', 'DevOps Engineer', 'UI/UX Designer', 
                         'Mobile App Developer', 'Cybersecurity Analyst'],
            'Required_Skills': ['python machine learning sql data analysis ai',
                                'html css javascript react vue ui',
                                'python java nodejs sql database api django',
                                'aws azure gcp linux networking cloud server',
                                'docker kubernetes jenkins linux automation',
                                'figma adobe design wireframing prototyping visual',
                                'flutter dart swift kotlin android ios',
                                'security networking ethical hacking firewalls linux']
        }
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        print("[INFO] Sample dataset 'raw_skills.csv' created successfully.\n")
    return file_name

# =====================================================================
# STAGE 1: INPUT (Ingestion & The Cold Start Solution)
# =====================================================================
def get_user_input():
    """
    Collects 3 distinct tech skills from the user to build their profile vector.
    """
    print("\n" + "="*50)
    print("   🚀 WELCOME TO THE AI TECH MATCHMAKER 🚀")
    print("="*50)
    print("Please enter your top 3 tech skills or interests.")
    
    skill_1 = input("Skill 1 (e.g., Python, AWS, Figma): ").strip()
    skill_2 = input("Skill 2: ").strip()
    skill_3 = input("Skill 3: ").strip()
    
    # Combine user skills into a single raw text document
    user_profile = f"{skill_1} {skill_2} {skill_3}".lower()
    print(f"\n[INFO] User Profile Registered: '{user_profile}'")
    return user_profile

# =====================================================================
# STAGE 2: PROCESS (TF-IDF & Cosine Similarity Engine)
# =====================================================================
def process_recommendations(file_name, user_profile):
    """
    Applies TF-IDF to convert text into vectors, then uses Cosine Similarity
    to find the angular alignment between the user and available job roles.
    """
    print("[INFO] Loading dataset and extracting core features...")
    df = pd.read_csv(file_name)
    
    job_skills = df['Required_Skills'].tolist()
    
    # Append the user profile at the end of the job list for combined vectorization
    job_skills.append(user_profile)
    
    print("[INFO] Applying TF-IDF Vectorization Engine...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(job_skills)
    
    print("[INFO] Calculating Cosine Similarity Matrix...")
    # Separate the user vector (last element) from the job vectors
    user_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1] 
    
    # Calculate angular distance (Similarity Scores)
    similarity_scores = cosine_similarity(user_vector, job_vectors)
    
    # Flatten the array and add scores to our dataframe
    df['Match_Score'] = similarity_scores.flatten()
    return df

# =====================================================================
# STAGE 3: OUTPUT (Filtering & Top-N Selection)
# =====================================================================
def display_top_matches(df, top_n=3):
    """
    Sorts the data based on similarity scores and outputs the top 3 matches
    to prevent 'Choice Overload'.
    """
    print(f"\n================ TOP {top_n} RECOMMENDATIONS ================")
    
    # Sort dataframe by Match_Score in descending order
    sorted_df = df.sort_values(by='Match_Score', ascending=False)
    top_matches = sorted_df.head(top_n)
    
    rank = 1
    for index, row in top_matches.iterrows():
        role = row['Job_Role']
        score = row['Match_Score'] * 100  # Convert 0-1 range to percentage
        skills = row['Required_Skills']
        
        # Only display matches that have at least some relevance (>0%)
        if score > 0:
            print(f"Rank {rank}: {role} (Alignment Match: {score:.1f}%)")
            print(f"   -> Key Skills: {skills.title()}\n")
            rank += 1
            
    if rank == 1:
        print("No matching roles found for your current skills.")
        print("Suggestion: Try exploring different tech keywords!")
        
    print("=========================================================")

# =====================================================================
# PIPELINE EXECUTION (MAIN)
# =====================================================================
if __name__ == "__main__":
    # Stage 0: Failsafe
    file_name = check_and_create_data()
    
    # Stage 1: Input
    user_profile = get_user_input()
    
    # Stage 2: Process
    results_df = process_recommendations(file_name, user_profile)
    
    # Stage 3: Output
    display_top_matches(results_df, top_n=3)
    
    print("\n>>> DIGITAL MATCHMAKER PIPELINE COMPLETED <<<")
