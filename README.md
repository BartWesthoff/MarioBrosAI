# Mario Bros Wii Reinforcement Learning Agent

This project implements a reinforcement learning agent for New Super Mario Bros Wii using the Dolphin emulator with Python scripting.

## Setup Instructions

### Prerequisites

- Python 3.11.5
- Dolphin Emulator (scripting-preview2-4802-dirty version)
- New Super Mario Bros Wii ROM

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/MarioBrosAI.git
   cd MarioBrosAI
   ```

2. Create a virtual environment with Python 3.11.5 and install dependencies:
   ```
   pip install portableenv
   python -m portableenv venv -v 3.11.5
   pip install -r requirements.txt
   ```

3. Set up Dolphin:
   - Copy your Dolphin emulator to the `Experiment` folder
   - Make sure you have the New Super Mario Bros Wii ROM in your Dolphin games directory

### Running the Agent

1. Start the emulator with the script:
   ```
   cd Experiment
   start.bat
   ```

2. The script will:
   - Automatically set up the necessary Python modules for Dolphin
   - Copy required libraries from your virtual environment to Dolphin's Python
   - Load New Super Mario Bros Wii
   - Display debug information on the screen
   - Capture screenshots and movement data for training (if PIL is available)
   - Resize the window (if pygetwindow is available)

## Project Structure

```
MarioBrosAI/
├── Experiment/
│   ├── dolphin/              # Dolphin emulator directory
│   ├── data/                 # Directory for storing movement data
│   ├── screenshots/          # Directory for storing screenshots
│   └── start.bat             # Script to start Dolphin with the agent
│
├── scripts/
│   ├── myscript.py           # Main script that runs in Dolphin
│   ├── utils_func.py         # Utility functions for the agent
│   ├── model.py              # Neural network model for the agent
│   └── online_trainer2.py    # Training script for the agent
│
├── StateSaves/
│   └── SMNP01.s01            # Save state for New Super Mario Bros Wii
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
