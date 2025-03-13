# Music-Genre-Classifier 🎶


# Idea Behind the Project

When classifying songs, one of the fundamental ideas is to categorize them by genre.
This can be achieved using simple machine learning (ML) classification techniques. However, a key challenge arises: a single song can belong to multiple genres. To address this, we developed a specialized classification pipeline.

# Pipeline
To account for the fact that a song may belong to multiple genres, we designed the following pipeline.

## How it works
1. Use seven classifiers to determine the probability of a song belonging to each genre.
1. Stack the probabilities returned by the classifiers into an input vector for logistic regression.
1. Use a trained logistic regression model (each genre has a dedicated logistic regression model) to determine the final probability of a song belonging to a specific genre, based on the input vector. This approach ensures that classifiers performing better for a particular genre are given appropriate importance.
1. Repeat steps 1–3 for each of the ten genres.
1. Aggregate the probability results and display the percentage likelihood for each genre per song.

# Training Process
1. We trained seven classifiers for each genre, resulting in 70 classifiers in total.
1. Training was based on features such as:
    - Duration
    - Zero crossing rate
    - Spectral centroid (mean & standard deviation)
    - Spectral rolloff (mean)
    - Spectral bandwidth (mean)
    - Tempo
    - MFCC (Mel-frequency cepstral coefficients) 1–13 (mean & standard deviation)
    - Chroma (mean & standard deviation)
    - Root mean square (RMS) energy (mean & standard deviation)

1. We trained ten logistic regression classifiers using probability vectors for stacking.

# About the data
1. Collected approximately 1,000 songs.
1. Extracted and saved relevant features in a CSV file.
1. Split the data into 80% training and 20% testing, ensuring stratification to maintain class distribution ratios.

# Implementation Details

## Training Process
1. Consider only a single genre at a time (e.g., Category A).
1. Use the training and test sets.
1. Train seven classifiers to detect Category A. Each classifier outputs:
    - The probability of each test song belonging to Category A.
    - The corresponding true labels (identical across all classifiers since they use the same test split).
1. Construct a 200×7 probability matrix:
    - Rows represent test songs (test dataset size: 0.2 × total dataset size).
    - Columns represent probabilities from the seven classifiers for each song being of Category A.

1. Example matrix given to logistic regression training:
    - logistic regression gets matrix of form (each row being probabilities from each Classifier that specific test song is of Category A):
    
    | Classifier 1 | Classifier 2 | ... | Classifier 7 |
    |-------------|-------------|-----|-------------|
    | 1.0         | 0.4         | ... | 0.2         |

1. Train a logistic regression model on this probability matrix, using the true labels from classifiers. The dataset is again split into 80% training (160 songs) and 20% testing (40 songs).

1. Save the trained logistic regression model and repeat the process for all other genres.

# Demo



## System Requirements 🛠️

To set up and run this project, ensure that you have the following tools installed:

- **Python**: Version 3.9 or newer
- **Node.js**: Version ^18.19.1, ^20.11.1, or ^22.0.0
- **Angular CLI** (optional): Version 19.x or newer
> **Note**: Angular CLI is optional, as the project can be run using `npx` without requiring a global installation of Angular CLI.
> 
## SETUP 🚀
### Clone repository
> git clone https://github.com/PT00/Music-Genre-Classifier.git

### 1. Python Backend

#### 1.1. Change directory:
> cd Music-Genre-Classifier/backend

#### 1.2. Create VENV (Optional)

> python3 -m venv .venv
>
 or
> python -m venv .venv

##### Linux / macOS:

> source .venv/bin/activate

##### Windows:

> .venv\Scripts\activate

#### 1.3. Install Python packages

> pip install -r requirements.txt

#### 1.4. Run Local Server
> uvicorn main:app --reload --host 127.0.0.1 --port 8000

### 2. Angular Frontend

#### 2.1. Change directory:
> cd Music-Genre-Classifier/frontend/mgc-client

#### 2.2. Install packages
> npm install

#### 2.3. Run Client App
> npx ng serve
