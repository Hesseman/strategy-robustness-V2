# Strategy Robustness App V2

Upload a TradeStation **Strategy Performance Report** (saved as CSV) and the **bar data** the
strategy ran on (Data Window export, same symbol and interval). The app joins the trade list
to the bars and runs the trade-list subset of our robustness battery against the five cards
below. No files? Click **Try the demo** in the sidebar to run the same battery against a
synthetic strategy instead. Below the five cards, a timing-sensitivity section shifts every
entry or exit by a few bars (reference, no verdict). Three more cards are optional and read a
MultiWalk optimisation instead — see "MultiWalk surface tests (optional)" below.

| Card | Question | Verdict type |
|---|---|---|
| Baseline | edge vs the drift, not vs zero | reference |
| T8a random entries | better than random timing with the same holds? | gate: p < 0.05 |
| T3 eras + crisis | every era? high-volatility bars? | score: k of 4 windows |
| T7 cost stress | how much friction kills it? | gate: net lift after 1× cost > 0 |
| Drawdown & capital | CDaR-80, capital = 5 × CDaR-80, annual % | reference |
| Entry delay | does the edge live in the first bars after the signal? | reference (timing section) |
| Exit delay | is the exit precisely timed? | reference (timing section) |
| WFC — Walk Forward Correlation | did in-sample results predict out-of-sample results across the whole parameter grid? | gate: pooled p < 0.05 and ≥ half of the positive-IS combinations positive OOS (MultiWalk section, optional) |
| Plateau | does the chosen parameter set sit on a plateau or a spike? | score 0–1 (MultiWalk section, optional) |
| Selection haircut | how good does the best of N look when there is nothing to find? | reference (MultiWalk section, optional) |

The tests take the strategy as given. They do not know how many strategies or parameter sets
were tried, so there is no multiple-testing correction — a pass means "we could not break it
with these tests", not "it works".

## Explanations in the app

The app explains itself as you go: the **How we judge a strategy - the concept in one
minute** expander at the top covers lift vs drift, gates vs scores vs reference, and how
CDaR-80 turns into capital and an annual return; every input in the sidebar carries a `?`
tooltip, and so do the timing section's two controls; **How to export** popovers next to the report and bars uploaders repeat the
TradeStation steps below without leaving the page; and a capital toggle in the sidebar
switches the drawdown card between capital = 5 × CDaR-80 (the default) and a fixed starting
amount you choose — CDaR-80 itself is shown either way.

## Exporting the two files from TradeStation

1. **Strategy Performance Report** — open it for the strategy, then save it as **CSV** (not
   Excel). The app reads its Trades List and Settings.
2. **Bars** — export the chart's Data Window for the **same symbol, same interval, covering
   the whole traded range**. Extra indicator (PLOT) columns are ignored.

Both exports must come from the same workspace so their timestamps agree.

## Timing sensitivity (same two files)

Below the five cards, the same two exports answer one more question: **how sensitive is the
return to the timing of the trades?** Two cards, each a curve over a shift of up to 10 bars
(the section's slider goes to 20): the gross $ when every entry is taken k bars late with the
exits as reported, and when every exit is taken k bars late with the entries as reported.
k = 0 is the report's own fills. It is a picture of fragility, not a gate: no verdict, no
p-value, not counted in "Gates passed".

Each chart also runs the other way, left of zero: the entry (or exit) taken 1..k bars
**earlier**. No strategy can act before its signal fires, so that side is hindsight and not
tradable. It shows how much of the move each signal lags, and a "Where timing matters" block
compares the two legs per trade: if an earlier entry gains more than an earlier exit, the entry
trigger is the one worth working on, and the other way round.

A moved leg fills at the open of the bar it moves to; a trade whose delayed entry reaches its
exit bar, or whose delayed exit falls past the last bar, is skipped at that k and counted
(`n alive`). Returns are gross, one contract, no costs, trades independent; a *fixed hold*
mode moves the exit along with a delayed entry so the hold length is kept. The section has its
own JSON download (`timing_results.json`).

Out of scope: treating the earlier (hindsight) side as a tradable result, re-running the
strategy's own stop/target logic on the shifted position, cost haircuts, position limits or
netting of overlapping trades, and any gate or p-value.

## MultiWalk surface tests (optional)

Three more cards for strategies optimised in MultiWalk Pro: they read every parameter
combination MultiWalk tried, not just the winner, and need no report and no bars.

**Needs MultiWalk Pro.** MultiWalk normally stores the optimisation results in a sealed
`.dat` file. Switch the project to the readable text format once:

1. Open `MultiWalkSetup.txt` in the project folder (next to `Optimization Files` and
   `Walkforward Files`) in Notepad and change `iUseLegacyOptimizationTextFileFormat: false` to
   `iUseLegacyOptimizationTextFileFormat: true`. Save.
2. In MultiWalk: **Settings -> Project Files/Folders -> Import Project Setup**, select that
   file, then **Save Project Setup** on the same screen. Editing the file alone is not enough -
   MultiWalk rewrites it from memory.
3. **Operations -> Run Project.** The `Optimization Files` folder now holds `..._MultiWalk.txt`
   instead of the `.dat` (about 40 MB for 240 combinations over ten years).

Upload two files: the `..._MultiWalk.txt` from `Optimization Files` and `WalkforwardData.db`
from `Walkforward Files`. Nothing else is needed - these tests use no bars and no report.

What the three cards mean:

- **WFC (Walk Forward Correlation)** — gate: whether in-sample results predicted
  out-of-sample results across the whole parameter grid, not just for the one combination
  MultiWalk picked; pooled p < 0.05 and at least half of the in-sample-positive combinations
  stayed positive out-of-sample.
- **Plateau** — score 0–1: whether the chosen parameter set sits on a plateau of neighbours
  that also worked out-of-sample (flat and positive), or on an isolated spike (0 = spike).
- **Selection haircut** — reference: how much of the best-of-N in-sample result a
  block-bootstrap null with no real edge would produce anyway, just from trying many
  combinations.

These tests know how many variants this one optimisation tried. They know nothing about
other strategies, other symbols, or other parameter grids you may have tried elsewhere — they
sit alongside the five cards above, not in place of them.

## Run with Docker (step by step)

You need nothing installed except Docker Desktop. No Python, no packages.

### 1. Install and start Docker Desktop

- Download it from https://www.docker.com/products/docker-desktop/ and install. On Windows
  accept the default WSL 2 backend and reboot if asked.
- Start Docker Desktop and wait until the whale icon in the tray stops animating and the
  bottom-left of the window says **Engine running**. Every `docker` command below fails with
  "cannot connect to the Docker daemon" until it does.

### 2. Get the code

Open a terminal (PowerShell, Windows Terminal, or Git Bash) and run:

    git clone https://github.com/Hesseman/strategy-robustness-V2.git
    cd strategy-robustness-V2

No Git? On the GitHub page click **Code → Download ZIP**, unzip it, and `cd` into the folder.

### 3. Build the image and start the app

    docker compose up --build

The first run downloads the Python base image and installs the packages (a few minutes,
about 900 MB). Later runs reuse that work and start in seconds. The app is ready when the log
shows:

    You can now view your Streamlit app in your browser.
    Local URL: http://localhost:8501
    Network URL: http://172.17.0.3:8501
    External URL: http://...:8501

This log shows the container's own port (8501) on all three lines. Compose maps host
**8502** → container 8501, so open the app in your browser at **http://localhost:8502** — not
the URLs printed above. Leave this terminal open: the app runs as long as the command does. To run it in the background instead, add `-d`
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
  `8502` → **Run**.
- **Containers** tab shows it running. Click the `8502:8501` port link to open the app, and
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
"Load sample files" button appears. Set `SR_SAMPLE_MW_DIR` to a MultiWalk project folder
that has already been run with the legacy text format (see "MultiWalk surface tests
(optional)" above) and a "Load MultiWalk sample (dev)" button appears once both files
resolve inside it. Real report/bar/MultiWalk files are never committed.

On Git Bash, prefix the sample `docker run` with `MSYS_NO_PATHCONV=1` so `/samples/...` is
not rewritten.

Design notes (in the C:/Projects estate):
docs/superpowers/specs/2026-09-21-strategy-robustness-app-design.md (v1, the five cards) and
docs/superpowers/specs/2026-09-22-strategy-robustness-v2-design.md (V2).
