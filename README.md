# Music-Genre-Classifier 🎶

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

> python -m venv .venv

##### Linux / macOS:

> source .venv/bin/activate

##### Windows:

> env\Scripts\activate

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
