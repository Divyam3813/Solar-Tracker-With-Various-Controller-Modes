# ☀️ Solar Tracker Control Studio

**Solar Tracker Control Studio** is a fully interactive, browser-based control systems laboratory and simulation suite designed for engineering students, educators, and control practitioners. Built using **Streamlit**, **Plotly**, and the **`python-control`** library, it models a single-axis solar tracking mechanism that follows the sun across the sky ($0^\circ$ east to $180^\circ$ west) under rigorous real-world physical constraints, disturbances, and feedback tuning methodologies.

---

## 🌐 Live Application

Experience the live interactive studio directly in your browser:
* **[Access Solar Tracker Control Studio Here](https://solar-tracker.streamlit.app/)**

---

## 🌟 Key Features & Architecture

The application is meticulously structured into distinct modules to separate user interface, numerical simulation, plant physics, graphical rendering, and AI tutoring:

1. **🌞 Live Tracker & Visualizer (`scene.py` & `simulation.py`)**
   * Renders a dynamic, real-time SVG vector scene illustrating the sun's trajectory, drifting cloud layers, sun rays, and a responsive panel normal vector.
   * Features a $100\text{ s}$ compressed day-cycle simulation timeline with adjustable cloud arrival windows, severities, and random sensor noise.

2. **🎛️ Controller & Tuning Suite (`tuning.py`)**
   * Supports **P**, **PI**, **PD**, and **PID** controller architectures.
   * Implements industry-standard tuning rules:
     * **Ziegler-Nichols:** Closed-loop ultimate cycle method utilizing ultimate gain ($K_u$) and ultimate period ($T_u$).
     * **Cohen-Coon:** Open-loop reaction curve method tailored for integrating processes.
   * Includes robust features like anti-windup via conditional integration and derivative filtering ($\tau_D$).

3. **📊 Simulation & Performance Metrics (`app.py`)**
   * Computes comprehensive quantitative performance indexes: **IAE** (Integral Absolute Error), **ISE** (Integral Squared Error), peak errors, cloud disturbance rejection, and actuator saturation percentages ($\pm 5\text{ V}$ limit).
   * Generates comparative dataframes across all 8 standard controller-tuning configurations simultaneously.

4. **🧮 Theoretical Background & Root Locus (`derivation.py`)**
   * **Plant Derivation Tab:** Breaks down electrical armature equations ($R_a, K_b$), mechanical torque equations ($J, B$), and driver dynamics ($T_a$) into a rigorous third-order transfer function.
   * **Poles & Root Locus Tab:** Visualizes open-loop pole distributions, imaginary axis crossings, and closed-loop root locus branches as proportional gains vary.

5. **💬 AI Tutor - "Dr. Dipak Adhyaru" (`chatbot.py`)**
   * Powered by Google Gemini, this assistant provides pedagogical guidance on control systems theory, plant behavior, and troubleshooting, styled after Prof. Dipak Adhyaru's academic profile (Nirma University).

---

## 📂 Project Directory Structure

```text
solar-tracker-studio/
├── app.py              # Main Streamlit user interface, tabs, and session state manager
├── tuning.py           # Transfer functions, plant parameters, and ZN/CC tuning rules
├── simulation.py       # Time-domain discrete simulation engine, noise models, and metrics
├── scene.py            # Dynamic SVG generator for the live visualizer and HUD
├── derivation.py       # Detailed educational text and LaTeX equations for plant modeling
├── chatbot.py          # Google Gemini AI assistant wrapper and prompt engineering
└── requirements.txt    # Required Python dependencies
```

---

## ⚙️ Installation & Local Setup

### Prerequisites

Ensure you have **Python 3.10 or higher** installed on your system along with pip.

### Step 1: Clone or Download the Repository

Clone this repository or place all project files (`app.py`, `chatbot.py`, `derivation.py`, `scene.py`, `simulation.py`, `tuning.py`, `requirements.txt`) into a single directory.

### Step 2: Install Dependencies

Install the required scientific, data processing, and visualization libraries via pip:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Gemini API Key (Optional)

If you wish to use the **"Ask Dr. Adhyaru"** AI assistant tab, you must configure a Google Gemini API key:

* **Local Environment Variable:**
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key-here"
  ```
* **Streamlit Secrets (`.streamlit/secrets.toml`):**
  Create a folder named `.streamlit` in your project root and add a file named `secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your-gemini-api-key-here"
  # GEMINI_MODEL = "gemini-flash-latest"  # Optional model override
  ```

---

## 🚀 Running the Application Locally

Launch the Streamlit server from your terminal inside the project directory:

```bash
streamlit run app.py
```

This will automatically open the web application in your default browser at `http://localhost:8501`.

---

## 🧮 Mathematical & Control Model Summary

The plant models a DC motor driving a single-axis solar tracker panel:

1. **Electrical Armature Law (Neglecting Inductance):**
   $$V(t) = R_a\, i(t) + K_b\,\omega(t)$$

2. **Mechanical Dynamics:**
   $$J\,\dot\omega(t) + B\,\omega(t) = K_t\, i(t)$$

3. **Motor + Panel Transfer Function ($G(s)$):**
   $$G(s) = \frac{\Theta(s)}{V(s)} = \frac{K}{s(\tau s + 1)}$$
   where:
   * Gain: $K = \frac{K_t}{R_a B + K_t K_b} \approx 1.905\text{ deg/s per V}$
   * Time Constant: $\tau = \frac{R_a J}{R_a B + K_t K_b} \approx 1.429\text{ s}$

4. **Complete Plant with Actuator/Driver Lag ($G_p(s)$):**
   $$G_p(s) = \frac{1}{T_a s + 1} \cdot \frac{K}{s(\tau s + 1)} = \frac{K}{\tau T_a s^3 + (\tau + T_a)s^2 + s}$$
   where driver lag $T_a = 0.5\text{ s}$.

---

## 🛠️ Troubleshooting & FAQ

* **Q: The AI chatbot states an API key is missing.**
  * *A:* Ensure you have correctly set the `GEMINI_API_KEY` environment variable or added it to your Streamlit Cloud secrets manager under **App Settings → Secrets**.
* **Q: The simulation throws a numerical instability warning.**
  * *A:* Excessively high controller gains ($K_p$, $K_i$, $K_d$) can push closed-loop poles into the right half-plane. Reset the sliders to the default Ziegler-Nichols or Cohen-Noon presets to restore stability.

---

## 🤝 Contributing & Feedback

Contributions, bug reports, and feature requests are warmly welcomed! Please feel free to open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
