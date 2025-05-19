# solar-challenge-week1.

## Reproducing the Environment

To set up and reproduce the Python environment for this project, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd solar-challenge
   ```

2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

5. **(Optional) Deactivate the environment when done:**
   ```sh
   deactivate
   ```

> **Note:** The `.gitignore` is configured to prevent your virtual environment and any `.env` files from being tracked by Git.
