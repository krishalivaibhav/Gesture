# Hand Skeleton Component (OpenCV + MediaPipe)

This small program opens your webcam, detects up to **2 hands**, and draws a hand landmark skeleton so you can see tracking quality in real time.
It also tracks pinch counters for each hand separately.

It supports both MediaPipe APIs:

- Older API: `mp.solutions.hands`
- Newer API: `mp.tasks.vision.HandLandmarker` (used on Python 3.13+ builds where `mp.solutions` is unavailable)

## Code Layout

- `hand_skeleton_component.py`: entrypoint (CLI run target)
- `gesture_config.py`: runtime config + argument parsing
- `camera_utils.py`: camera backend probing/open helpers
- `hand_skeleton_app.py`: hand-tracking app logic, gesture state, rendering
- `mini_game.py`: Ping Pong game mode logic
- `sign_mode.py`: Sign Language learner/practice mode controller
- `sign_curriculum.py`: built-in words/sentences for sign lessons

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python hand_skeleton_component.py
```

On newer MediaPipe builds, the first run auto-downloads the hand model file to:

```bash
/Users/vaibhavkrishali/Desktop/musicGestures/models/hand_landmarker.task
```

If auto-download fails, download it manually:

```bash
mkdir -p /Users/vaibhavkrishali/Desktop/musicGestures/models
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  -o /Users/vaibhavkrishali/Desktop/musicGestures/models/hand_landmarker.task
```

Optional arguments:

```bash
python hand_skeleton_component.py --camera 0 --camera-backend avfoundation --target-fps 60 --capture-width 960 --capture-height 540 --inference-interval 1 --inference-scale 0.6 --max-hands 2 --min-det-confidence 0.5 --min-track-confidence 0.5
```

If you already have a model file downloaded:

```bash
python hand_skeleton_component.py --model-path /absolute/path/to/hand_landmarker.task
```

## 60 FPS Mode

Try this preset:

```bash
python hand_skeleton_component.py --camera-backend avfoundation --target-fps 60 --capture-width 960 --capture-height 540 --inference-interval 1 --inference-scale 0.6
```

Notes:

- The app now requests 60 FPS from the camera.
- Actual FPS depends on your camera + lighting + CPU load.
- The app uses a threaded inference worker by default for smoother rendering.
- The overlay shows processing FPS, inference FPS, camera FPS, and acceleration mode.
- For higher smoothness on slower CPUs, increase `--inference-interval` (e.g. `3` or `4`).
- For faster inference, lower `--inference-scale` (e.g. `0.5`).
- To force CPU path: add `--cpu-only`
- To disable threaded worker: add `--single-thread`

## Controls

- `q` or `Esc`: quit
- `n`: toggle landmark ID labels (0..20)
- `c`: toggle counters/HUD overlay
- `1`: toggle Ping Pong game mode on/off
- `2`: switch game to `Two Player` mode
- `3`: switch game to `Bot` mode
- `4`: toggle `Sign Language` mode on/off
- `r`: reset Ping Pong score (when game mode is active)

The display is fixed to selfie-style mirror mode so your left hand appears on the left side of the screen.

## Ping Pong Game

The app includes a basic pong-style game in game mode:

- Left paddle and right paddle move only vertically.
- Power-ups spawn from center and move toward left/right sides with weighted-random fairness.
- Paddle-touch pickup model:
  - touching a power-up with your paddle activates it for that side.
  - missed power-ups leave the court and are lost.
- Power-ups implemented:
  - `Speed Ball`: next hit from that side launches a very fast shot.
  - `Paddle Size`: paddle grows for ~5 seconds.
  - `Shield`: one-time wall save; missed ball reflects instead of conceding.
  - `Chaos Orb`: next hit adds a random directional chaos offset.
- `Two Player` mode:
  - waits for two tracked hands before starting.
  - shows detected hand combination (right/left counts) before kickoff.
  - left-most visible hand controls left paddle; right-most controls right paddle.
- `Bot` mode:
  - one visible hand controls left paddle.
  - right paddle is controlled by a bot.

## Sign Language Mode

Press `4` to enter Sign Language mode.

### Menu

- `Learner Mode`
- `Practice Mode`

Buttons can be clicked using:

- hand pinch click (`index + thumb` at cursor position)
- mouse click fallback

Press `4` again to exit Sign mode back to plain view.

### Learner Mode

Includes three levels:

- `Level 1`: Alphabet drill (single letter with reference image)
- `Level 2`: Word building (letter-by-letter progression with reference image)
- `Level 3`: Sentence building (word-style flow with spaces auto-skipped)

Each letter has a blank above it. Correct answer turns blank into a tick and advances.

### Practice Mode

- Uses sentence prompts
- No reference image shown
- Per-letter feedback:
  - wrong stable sign -> cross
  - correct stable sign -> tick and advance

### Template Images

Sign template images are loaded from:

- `assets/signs/alphabet`

See details and naming rules in:

- `assets/signs/alphabet/README.md`

The matcher currently supports static letters:

- `A-I` and `K-Y` (motion letters `J` and `Z` are excluded in v1)

## Pinch Counters

For each detected hand (`Left Hand` / `Right Hand`), the app increments counters for:

- `Index-Thumb` pinch
- `Middle-Thumb` pinch
- `Ring-Thumb` pinch
- `Pinky-Thumb` pinch

Each counter increments once per pinch event (it waits for release before counting the next pinch).

## Knob Gesture (Both Hands)

Each hand has its own knob state and counters.

- Activate knob mode by pinching `thumb + index + middle` on that hand.
- While pinched, rotate your hand/fingers to generate:
  - `CW` (clockwise) steps
  - `CCW` (counter-clockwise) steps
- The overlay shows, per hand:
  - `Value`
  - `CW`
  - `CCW`
  - current state (`Hold`, `CW`, `CCW`, `Idle`)

## Tips for better accuracy

- Use bright, even lighting.
- Keep your hands fully inside the camera frame.
- Avoid motion blur (move slightly slower if needed).
- If hands are lost during fast movement, keep `--disable-recovery-full-res` **off** (default); recovery mode re-runs inference at full resolution when tracking drops.

## macOS Camera Permission Troubleshooting

If you see `camera access has been denied` or `not authorized to capture video`:

```bash
tccutil reset Camera com.apple.Terminal
```

Then:

1. Open **System Settings -> Privacy & Security -> Camera**
2. Enable camera access for your terminal app (Terminal, iTerm, Warp, etc.)
3. Quit and reopen the terminal app

You can probe available camera indices:

```bash
python hand_skeleton_component.py --list-cameras --camera-backend avfoundation
```

Then run with the detected index:

```bash
python hand_skeleton_component.py --camera 0 --camera-backend avfoundation
```
