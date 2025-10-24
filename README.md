# 🧩 Project Setup Instructions

## 📁 Step 1: Create Environment File
Inside the **`webapp`** folder, create a file named **`.env`** and add the following credentials:

```bash
# .env file
GEMINI_API_KEY=AIzaSyCpNpGsMDoMXaU91_55oUpXu5zsTA2bHDU
SECRET_KEY=dasdas781ew219ndsadnasdasdasd12312
DB_USERNAME=siwakos01
DB_PASSWORD=siwakos01
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dsProjects
```

> ⚠️ **Note:** Never share your `.env` file publicly or commit it to GitHub.  
> Use `.gitignore` to keep it private.

---

## 🐍 Step 2: Create and Activate Virtual Environment
From the **project root directory** (outside the `webapp` folder):

```bash
# Create a virtual environment
python3 -m venv env
# OR (if virtualenv is installed)
virtualenv -p python3 env
```

Activate the environment:

```bash
# On macOS / Linux
source env/bin/activate

# On Windows
env\Scripts\activate
```

Verify activation:
```bash
which python
```
You should see something like:
```
/your_project_path/env/bin/python
```

---

## 📦 Step 3: Install Dependencies
Once the virtual environment is active, install all required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Step 4: Run the Application
Start the Flask web application:

```bash
python webapp/main.py
```

If everything is configured correctly, you should see:
```
* Running on http://127.0.0.1:5000
```

Open that link in your browser to access the web app.

---

## 🧠 Optional (Best Practices)
- Add `.env` to `.gitignore`
- Regularly update dependencies with:
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- To deactivate the environment:
  ```bash
  deactivate
  ```
