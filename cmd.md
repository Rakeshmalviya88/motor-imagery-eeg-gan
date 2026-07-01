1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Rakeshmalviya88/motor-imagery-eeg-gan.git
   cd motor-imagery-eeg-gan
   ```

2. **Download and Place the Dataset:**
   Ensure the `BCICIV_2a_gdf` folder (containing the `.gdf` and `.mat` files) is placed in the project root directory.

3. **Create a Virtual Environment (`.venv`):**
   * Make sure Python 3.10 or 3.11 is installed.
   * Run the command to create a virtual environment:
     ```bash
     python -m venv .venv
     ```

4. **Activate the Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```

5. **Install Dependencies:**
   Run the following command to install all required libraries (including `tensorflow`, `mne`, `scikit-learn`, `scipy`, `matplotlib`, and Jupyter dependencies):
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Start Jupyter and Run the Notebooks:**
   Start the Jupyter server:
   ```bash
   jupyter notebook
   ```
   Or open the files in VS Code and select the `.venv` kernel to run cells interactively.
