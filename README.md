# Strategy Robustness App V2

Upload a TradeStation **Strategy Performance Report** (saved as CSV) and the **bar data** the
strategy ran on (Data Window export, same symbol and interval). The app joins the trade list
to the bars and runs the trade-list subset of our robustness battery. No files? Click **Try
the demo** in the sidebar to run the same battery against a synthetic strategy instead.

| Card | Question | Verdict type |
|---|---|---|
| Baseline | edge vs the drift, not vs zero | reference |
| T8a random entries | better than random timing with the same holds? | gate: p < 0.05 |
| T3 eras + crisis | every era? high-volatility bars? | score: k of 4 windows |
| T7 cost stress | how much friction kills it? | gate: net lift after 1× cost > 0 |
| Drawdown & capital | CDaR-80, capital = 5 × CDaR-80, annual % | reference |

The tests take the strategy as given. They do not know how many strategies or parameter sets
were tried, so there is no multiple-testing correction — a pass means "we could not break it
with these tests", not "it works".

## Exporting the two files from TradeStation

1. **Strategy Performance Report** — open it for the strategy, then save it as **CSV** (not
   Excel). The app reads its Trades List and Settings.
2. **Bars** — export the chart's Data Window for the **same symbol, same interval, covering
   the whole traded range**. Extra indicator (PLOT) columns are ignored.

Both exports must come from the same workspace so their timestamps agree.

## Run with Docker (step by step)

You need nothing installed except Docker Desktop. No Python, no packages.

### 1. Install and start Docker Desktop

- Download it from https://www.docker.com/products/docker-desktop/ and install. On Windows
  accept the default WSL 2 backend and reboot if asked.
- Start Docker Desktop and wait until the whale icon in the tray stops animating and the
  bottom-left of the window says **Engine running**. Every `docker` command below fails with
  "cannot connect to the Docker daemon" until it does.

### 2. Get the code

Open a terminal (PowerShell, Windows Terminal, or Git Bash) and run (published later; until
then copy the folder):

    git clone https://github.com/Hesseman/strategy-robustness-V2.git
    cd strategy-robustness-V2

No Git? On the GitHub page click **Code → Download ZIP**, unzip it, and `cd` into the folder.

### 3. Build the image and start the app

    docker compose up --build

The first run downloads the Python base image and installs the packages (a few minutes,
about 900 MB). Later runs reuse that work and start in seconds. The app is ready when the log
shows:

    You can now view your Streamlit app in your browser.
    Local URL: http://localhost:8502
    Network URL: http://172.17.0.3:8501
    External URL: http://...:8501

Use the **Local URL**. The Network and External URLs are the container's own view and do not
work from your browser. Leave this terminal open: the app runs as long as the command does. To run it in the background instead, add `-d`
(`docker compose up --build -d`) and stop it later with `docker compose down`.

### 4. Open the app

Go to **http://localhost:8502** in your browser. Upload the two TradeStation exports, or click
**Try the demo** in the sidebar. Nothing is written to disk; uploads live in memory for the
session and are gone when the container stops.

### 5. Stop the app

Press **Ctrl+C** in the terminal, then:

    docker compose down

This removes the container. The built image stays, so the next start is fast.

### 6. Start it again later

    docker compose up

No `--build` needed unless the code changed. After a `git pull`, run step 3 again so the
image picks up the new code.

### Using Docker Desktop instead of the terminal

Once the image exists (step 3 has run once), you can drive it from the GUI:

- **Images** tab → `strategy-robustness-v2` → **Run** → expand *Optional settings* → Host port
  `8501` → **Run**.
- **Containers** tab shows it running. Click the `8501:8501` port link to open the app, and
  use the **Stop** / **Start** buttons instead of steps 5 and 6.

Compose binds the port to `127.0.0.1` (this machine only). Starting from the GUI binds it to
all interfaces, so the app may be reachable from other machines on your network.

### Share the image as a file (no GitHub, no rebuild)

On the machine that built it:

    docker save -o strategy-robustness-v2.tar strategy-robustness-v2

Copy the `.tar` (about 900 MB) to the other machine, then there:

    docker load -i strategy-robustness-v2.tar
    docker run --rm -p 127.0.0.1:8502:8501 strategy-robustness-v2

### Without compose

    docker build -t strategy-robustness-v2 .
    docker run --rm -p 127.0.0.1:8502:8501 strategy-robustness-v2

### Run the test suite inside the image (no browser needed)

    docker run --rm strategy-robustness-v2 python -m pytest -q

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `cannot connect to the Docker daemon` or `error during connect` | Docker Desktop is not running. Start it, wait for **Engine running**, retry. |
| `port is already allocated` | Something else uses 8502. In `docker-compose.yml` change `127.0.0.1:8502:8501` to `127.0.0.1:8503:8501` and open http://localhost:8503. |
| Browser says connection refused | The app has not finished starting, or you opened the Network/External URL. Wait for the "You can now view" line and use http://localhost:8502. |
| Code changes do not show up | Rebuild: `docker compose up --build`. |

## Develop

    py -3.13 -m venv .venv
    .venv/Scripts/python -m pip install -r requirements.txt
    .venv/Scripts/python -m pytest -q
    .venv/Scripts/python -m streamlit run app/streamlit_app.py

Dev-only: set `SR_SAMPLE_REPORT` and `SR_SAMPLE_BARS` to local file paths and a
"Load sample files" button appears. Real report/bar files are never committed.

On Git Bash, prefix the sample `docker run` with `MSYS_NO_PATHCONV=1` so `/samples/...` is
not rewritten.

Design notes: docs/superpowers/specs/2026-09-21-strategy-robustness-app-design.md (in the
C:/Projects estate).
