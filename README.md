<div align="center">
  <img src="web_ui/public/pilot_logo_black.png" alt="DeePTB Pilot Logo" width="800"/>
  <h3>AI Agent for DeePTB</h3>
  <p>An intelligent copilot for the DeePTB software package, integrating LLM capabilities with material science tools.</p>
</div>

---

## 📖 Introduction

**DeePTB Pilot** is a sophisticated AI agent designed to assist researchers in using the DeePTB software. It combines a React-based frontend with a Python backend powered by LLMs and the Model Context Protocol (MCP).

### ✨ Key Features

*   **Interactive Chat**: Natural language interface to query DeePTB knowledge and perform tasks.
*   **RAG System**: Retrieval-Augmented Generation for accurate answers based on DeePTB documentation.
*   **Structure Visualization**: Interactive 3D visualization of crystal structures and Brillouin zones.
*   **MCP Tools**: Extensible tool system for file operations, materials search (Materials Project), and DeePTB calculations.
*   **Session Management**: Persistent chat sessions and history.

## 🛠️ Architecture

*   **Frontend**: React, Ant Design, 3Dmol.js (located in `web_ui/`)
*   **Backend**: Python, LiteLLM, MCP, Bohr Agent SDK (located in `dptb_pilot/`)
    *   `core/`: Core agent logic.
    *   `server/`: API Server.
    *   `tools/`: Collection of MCP tools.

## 🚀 Getting Started

### Prerequisites

*   **uv** (Required for Python package management):
    ```bash
    # Install uv via curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Node.js & npm** (Required for Frontend):
    *   **Mac/Windows**: [Download Official Installer](https://nodejs.org/) (LTS version)
    *   **Linux** (Recommended via `nvm`):
        ```bash
        # 1. Install nvm
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        # 2. Activate nvm
        source ~/.nvm/nvm.sh
        # 3. Install Node.js LTS (includes npm)
        nvm install --lts
        
        # 4. Verify installation
        node -v
        npm -v
        ```
*   **Git**
*   **DeePTB**: The pilot depends on the [DeePTB](https://github.com/deepmodeling/DeePTB) package, which is automatically installed by `uv`.

### 1. Installation
Clone the repository:
```bash
git clone https://github.com/DeePTB-Lab/dptb-pilot.git
cd dptb-pilot
```

Choose one of the following installation methods:

#### Option A: One-Click Installation (Recommended)
This script automatically handles python dependencies (with correct torch versions), builds the frontend, and suggests helpful aliases.
```bash
./install.sh
# Follow the on-screen prompts to add aliases to your shell
```

#### Option B: Manual Installation
If you prefer to install dependencies manually:

**1. Backend Dependencies**
```bash
uv sync
```

**2. Frontend Build**
```bash
cd web_ui
npm install
npm run build
cd ..
```

### 2. Configuration

Copy the example environment file and configure your settings:

```bash
cp env.example .env
```

Edit `.env` with your API keys and preferences:

```env
# workspace root
WORK_ROOT=./workspace

# materials project api key
MP_API_KEY=your_materials_project_key

# LLM Configuration
LLM_MODEL=openai/custom_model_name
LLM_API_BASE=https://xxx.xxx.xxx/v1
LLM_API_KEY=your_llm_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=50003
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=50002
BACKEND_HOST=localhost
MCP_TOOLS_PORT=50001
```

### 3. Running the Application

#### Option A: One-Click Startup (Recommended)
If you added the alias from `install.sh`, simply run:
```bash
dptb-ai-run
```
Or run the script directly:
```bash
./activate.sh
```
This will launch both the backend and tools server in parallel, handling clean shutdown when you press `Ctrl+C`.

#### Option B: Manual Startup
If you prefer to run services individually, you need two terminals:

**Terminal 1: Start MCP Tools**
```bash
uv run dptb-tools
```

**Terminal 2: Start Pilot App**
```bash
uv run dptb-pilot
```

The application should automatically open in your browser at `http://localhost:50001`.

## 📚 Documentation

*   [**Usage Guide**](docs/USAGE.md): Detailed instructions on how to use the chat, file management, and visualization features.

## ⚠️ Troubleshooting

### Proxy Issues

If you are behind a proxy or VPN and cannot connect to `localhost`, set the `NO_PROXY` environment variable:

```bash
export NO_PROXY="localhost,127.0.0.1"
```

Or run the commands with the variable prepended:

```bash
NO_PROXY="localhost,127.0.0.1" dptb-pilot
```

## 📄 License

[License Information]
