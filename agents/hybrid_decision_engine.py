"""
hybrid_decision_engine.py

Complete Hybrid Intelligence Decision Engine.

Searchable integration markers:
  - INTEGRATION_POINT_MODEL_OUTPUT   -> where perception model outputs are expected (main loop)
  - INTEGRATION_POINT_SAVE_CLIP      -> where to save evidence clips / frames
  - INTEGRATION_POINT_NOTIFY_SYSTEM  -> where to send notifications (webhook / socket / DB)
  - INTEGRATION_POINT_BEEP           -> where to trigger physical alarm / beep
  - INTEGRATION_POINT_UAV_COMMAND    -> where to send commands to UAV/IoV orchestrator

How to run:
  - Provide detection dicts (see `example_detection` in `if __name__ == "__main__":`)
  - Use DecisionEngine.process(detection) per frame
"""

import time
import collections
import math
import threading
from typing import Dict, List, Any, Optional

# --------------------------
# Constants & Enums
# --------------------------
class Threat:
    NONE = 0.0
    LOW = 1.0
    MEDIUM = 2.0
    HIGH = 3.0
    CRITICAL = 4.0

class State:
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ARMED = "ARMED"
    VIOLENT = "VIOLENT"
    CRITICAL = "CRITICAL"

# --------------------------
# Helper utilities
# --------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Exponential moving average smoothing
class EMA:
    def __init__(self, alpha=0.5, initial: Optional[float]=None):
        self.alpha = alpha
        self.value = initial

    def update(self, x):
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

# Minimal Kalman-like 1D smoothing for confidence
class SimpleKalman1D:
    def __init__(self, process_uncertainty=1e-3, measurement_uncertainty=1e-1):
        self.q = process_uncertainty
        self.r = measurement_uncertainty
        self.x = None  # state
        self.p = 1.0

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        # prediction (no control input)
        p_pred = self.p + self.q
        # kalman gain
        k = p_pred / (p_pred + self.r)
        # update
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred
        return self.x

# --------------------------
# Decision Engine
# --------------------------
class DecisionEngine:
    """
    Hybrid Decision Engine combining:
      - Rule-based quick decisions
      - FSM per tracked person
      - Temporal smoothing
      - Bayesian fusion helper
      - MCDM threat scoring
      - Action manager (beep/save/notify/route UAV)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Per-person history & smoothing
        self.history = {}          # id -> deque of recent detection dicts
        self.max_history_len = 60  # number of frames to buffer per ID
        self.state = {}            # id -> FSM state
        self.confidence_kalman = {}  # id -> SimpleKalman1D
        self.ema_conf = {}         # id -> EMA

        # Configurable thresholds & weights
        self.config = {
            "gun_threshold": 0.45,
            "knife_threshold": 0.4,
            "fight_threshold": 0.5,
            "suspicious_duration_frames": 10,
            "ema_alpha": 0.4,
            "score_weights": {
                "severity": 0.6,
                "confidence": 0.25,
                "duration": 0.15
            },
            "critical_score": 3.2,
            "high_score": 2.4,
            "medium_score": 1.6
        }
        if config:
            self.config.update(config)

        # For thread-safe saving / notifying
        self.lock = threading.Lock()

    # --------------------------
    # Per-frame processing entry point
    # --------------------------
    def process(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single detection dict coming from perception / tracker per frame.

        Expected detection dict schema:
        {
            "id": <int or str>,
            "bbox": [x,y,w,h] or None,
            "person_conf": float (0..1),
            "gun_conf": float (0..1),
            "knife_conf": float (0..1),
            "fight_conf": float (0..1),
            "pose": {...} optional,
            "timestamp": unix timestamp float,
            "frame": optional raw frame (numpy array)  # use only if needed for saving
        }

        Returns: action dict with 'id','state','threat_score','action','reason'
        """
        pid = detection.get("id")
        if pid is None:
            raise ValueError("detection must include 'id'")

        # Ensure history structure
        if pid not in self.history:
            self.history[pid] = collections.deque(maxlen=self.max_history_len)
        if pid not in self.confidence_kalman:
            self.confidence_kalman[pid] = SimpleKalman1D()
        if pid not in self.ema_conf:
            self.ema_conf[pid] = EMA(alpha=self.config["ema_alpha"])
        if pid not in self.state:
            self.state[pid] = State.NORMAL

        # Append detection to history
        self.history[pid].append(detection)

        # Compute smoothed confidences
        raw_conf = detection.get("person_conf", 0.0)
        kalman_conf = self.confidence_kalman[pid].update(raw_conf)
        ema_conf = self.ema_conf[pid].update(raw_conf)
        # Use combined confidence (simple average) for scoring
        combined_confidence = (raw_conf + kalman_conf + ema_conf) / 3.0
        combined_confidence = clamp(combined_confidence, 0.0, 1.0)

        # Rule-based initial severity
        severity = self._rule_severity(detection)

        # Bayesian fusion between detection channels (weapon + action)
        bayes_prob = self._bayesian_fusion(detection)

        # Duration (how many frames recently indicate threat)
        duration_frames = self._count_recent_positive_frames(pid, detection)

        # Compute unified threat score (MCDM)
        score = self._compute_threat_score(severity, bayes_prob, combined_confidence, duration_frames)

        # Update FSM
        new_state = self._fsm_update(pid, severity, bayes_prob, duration_frames, score)

        # Action decision
        action, reason = self._decide_action(pid, new_state, score, detection)

        # Enact action (save/notify/beep) in separate thread to keep loop responsive
        # (these are stubs to be replaced by integration)
        self._execute_action_async(pid, action, reason, detection)

        result = {
            "id": pid,
            "state": new_state,
            "severity": severity,
            "bayes_prob": bayes_prob,
            "confidence": combined_confidence,
            "duration_frames": duration_frames,
            "threat_score": round(score, 3),
            "action": action,
            "reason": reason,
            "timestamp": detection.get("timestamp", time.time())
        }
        return result

    # --------------------------
    # Rule-based severity (fast)
    # --------------------------
    def _rule_severity(self, detection: Dict[str, Any]) -> float:
        """
        Deterministic rule: returns severity (0..4)
        """
        # Highest: fight
        if detection.get("fight_conf", 0.0) >= self.config["fight_threshold"]:
            return Threat.CRITICAL
        # Gun
        if detection.get("gun_conf", 0.0) >= self.config["gun_threshold"]:
            return Threat.HIGH
        # Knife
        if detection.get("knife_conf", 0.0) >= self.config["knife_threshold"]:
            return Threat.MEDIUM
        # fallback: person activity (running, loitering) may be available in detection['meta']
        meta = detection.get("meta", {})
        if meta.get("running", False):
            return Threat.MEDIUM
        if meta.get("loitering", False):
            return Threat.LOW
        return Threat.NONE

    # --------------------------
    # Bayesian fusion helper
    # --------------------------
    def _bayesian_fusion(self, detection: Dict[str, Any]) -> float:
        """
        Combine multiple detector confidences into a posterior-like probability.
        Simple approach: convert confidences to odds and multiply, then normalize.
        """
        # Get channels
        c_person = clamp(detection.get("person_conf", 0.0), 1e-6, 1 - 1e-6)
        c_gun = clamp(detection.get("gun_conf", 0.0), 1e-6, 1 - 1e-6)
        c_knife = clamp(detection.get("knife_conf", 0.0), 1e-6, 1 - 1e-6)
        c_fight = clamp(detection.get("fight_conf", 0.0), 1e-6, 1 - 1e-6)

        # Convert to odds
        def odds(p): return p / (1.0 - p)
        o = 1.0
        # weight channels differently
        o *= odds(c_person) ** 0.5
        o *= odds(max(c_gun, c_knife)) ** 1.2
        o *= odds(c_fight) ** 1.5

        # convert back to probability
        prob = o / (1 + o)
        prob = clamp(prob, 0.0, 1.0)
        return prob

    # --------------------------
    # Count recent positive threat frames for this id
    # --------------------------
    def _count_recent_positive_frames(self, pid, current_detection) -> int:
        """Count how many of the last N frames had severity > NONE for this id."""
        buf = self.history.get(pid, [])
        count = 0
        for d in reversed(buf):
            sev = self._rule_severity(d)
            if sev > Threat.NONE:
                count += 1
            else:
                # optionally break if you want contiguous duration only
                pass
        return count

    # --------------------------
    # MCDM Threat score
    # --------------------------
    def _compute_threat_score(self, severity, bayes_prob, confidence, duration_frames) -> float:
        """
        Score ranges roughly 0..4. Use config weights:
          score = w_sev * normalized_sev + w_conf * (confidence*4) + w_dur * min(duration_norm*4, 4)
        """
        w = self.config["score_weights"]
        # Normalize severity (0..4) -> (0..4)
        sev_norm = clamp(severity, 0.0, 4.0)
        conf_component = clamp(confidence * 4.0, 0.0, 4.0)
        dur_component = clamp(min(duration_frames / max(1, self.config["suspicious_duration_frames"]), 1.0) * 4.0, 0.0, 4.0)
        bayes_component = clamp(bayes_prob * 4.0, 0.0, 4.0)

        # Combine with weights; we can fold bayes into confidence by blending:
        conf_combined = 0.7 * conf_component + 0.3 * bayes_component

        score = w["severity"] * sev_norm + w["confidence"] * conf_combined + w["duration"] * dur_component
        # clamp to max 4
        score = clamp(score, 0.0, 4.0)
        return score

    # --------------------------
    # FSM update logic
    # --------------------------
    def _fsm_update(self, pid, severity, bayes_prob, duration_frames, score) -> str:
        curr = self.state.get(pid, State.NORMAL)
        new = curr

        # Very small helper to escalate/downgrade based on score
        if curr == State.NORMAL:
            if score >= self.config["medium_score"]:
                new = State.SUSPICIOUS
        elif curr == State.SUSPICIOUS:
            if severity >= Threat.MEDIUM or score >= self.config["high_score"]:
                new = State.ARMED
            elif score < self.config["medium_score"]:
                new = State.NORMAL
        elif curr == State.ARMED:
            if severity >= Threat.HIGH or score >= self.config["critical_score"]:
                new = State.VIOLENT
            elif score < self.config["high_score"]:
                new = State.SUSPICIOUS
        elif curr == State.VIOLENT:
            if score >= self.config["critical_score"]:
                new = State.CRITICAL
            elif score < self.config["high_score"]:
                new = State.ARMED
        elif curr == State.CRITICAL:
            # remain critical until score drops significantly
            if score < (self.config["high_score"] * 0.6):
                new = State.VIOLENT

        self.state[pid] = new
        return new

    # --------------------------
    # Decide action from state & score
    # --------------------------
    def _decide_action(self, pid, state, score, detection) -> (str, str):
        """
        Return (action, reason)
        action can be:
         - "NO_ACTION"
         - "LOG"
         - "LOCAL_ALARM" (beep)
         - "SAVE_EVIDENCE"
         - "NOTIFY_OPERATOR"
         - "DISPATCH_UAV"
         - combination like "LOCAL_ALARM|SAVE_EVIDENCE|NOTIFY_OPERATOR"
        """
        actions = []
        reasons = []
        if state == State.NORMAL:
            actions.append("NO_ACTION")
            reasons.append("normal")
        elif state == State.SUSPICIOUS:
            actions.append("LOG")
            reasons.append("suspicious behavior")
            if score > self.config["medium_score"] + 0.2:
                actions.append("SAVE_EVIDENCE")
                reasons.append("persisting behavior")
        elif state == State.ARMED:
            actions.append("SAVE_EVIDENCE")
            actions.append("LOCAL_ALARM")
            reasons.append("weapon detected")
            if score >= self.config["high_score"]:
                actions.append("NOTIFY_OPERATOR")
                reasons.append("high threat score")
        elif state == State.VIOLENT:
            actions.append("SAVE_EVIDENCE")
            actions.append("LOCAL_ALARM")
            actions.append("NOTIFY_OPERATOR")
            reasons.append("violent activity")
            if score >= self.config["critical_score"]:
                actions.append("DISPATCH_UAV")
                reasons.append("critical")
        elif state == State.CRITICAL:
            actions = ["SAVE_EVIDENCE", "LOCAL_ALARM", "NOTIFY_OPERATOR", "DISPATCH_UAV"]
            reasons = ["critical immediate response"]

        action_str = "|".join(actions)
        reason_str = ";".join(reasons)
        return action_str, reason_str

    # --------------------------
    # Execute action (async to avoid blocking perception loop)
    # --------------------------
    def _execute_action_async(self, pid, action, reason, detection):
        # Run in background so the main loop can continue
        t = threading.Thread(target=self._execute_action, args=(pid, action, reason, detection))
        t.daemon = True
        t.start()

    def _execute_action(self, pid, action, reason, detection):
        with self.lock:
            if "SAVE_EVIDENCE" in action:
                # INTEGRATION_POINT_SAVE_CLIP
                # Replace the following stub with your code to save frames or a short video clip.
                # Example: call save_clip(frames_buffer, f"evidence_{pid}_{timestamp}.mp4")
                try:
                    self._save_evidence_stub(pid, detection)
                except Exception as e:
                    print(f"[DecisionEngine] save_evidence failed: {e}")

            if "LOCAL_ALARM" in action:
                # INTEGRATION_POINT_BEEP
                self._beep_stub()

            if "NOTIFY_OPERATOR" in action:
                # INTEGRATION_POINT_NOTIFY_SYSTEM
                # Replace with your webhook, socket, message bus, or DB write.
                try:
                    self._notify_stub(pid, action, reason, detection)
                except Exception as e:
                    print(f"[DecisionEngine] notify failed: {e}")

            if "DISPATCH_UAV" in action:
                # INTEGRATION_POINT_UAV_COMMAND
                # Replace with your UAV/IoV command call
                try:
                    self._dispatch_uav_stub(pid, detection)
                except Exception as e:
                    print(f"[DecisionEngine] dispatch UAV failed: {e}")

    # --------------------------
    # Stubs for integration - replace these with your production code
    # --------------------------
    def _save_evidence_stub(self, pid, detection):
        # TODO: INTEGRATION_POINT_SAVE_CLIP
        # Replace this code with actual saving of frames/video.
        # `detection.get("frame")` might contain a numpy array if you passed frames.
        # You can accumulate frames from self.history[pid] when saving a multi-second clip.
        ts = detection.get("timestamp", time.time())
        print(f"[STUB_SAVE] Save evidence for id={pid} time={ts} reason={detection.get('meta','')}")
        # Example: collect frames from history, encode to mp4 -> write to disk or S3.

    def _beep_stub(self):
        # TODO: INTEGRATION_POINT_BEEP
        # Replace with code to activate a buzzer, play sound, or UI alert.
        print("[STUB_BEEP] Beep! Local alarm triggered.")

    def _notify_stub(self, pid, action, reason, detection):
        # TODO: INTEGRATION_POINT_NOTIFY_SYSTEM
        # Replace with network call to your operator dashboard, e.g. HTTP POST or MQTT publish.
        payload = {
            "id": pid,
            "action": action,
            "reason": reason,
            "timestamp": detection.get("timestamp", time.time()),
            "bbox": detection.get("bbox")
        }
        # Example: requests.post("https://your-operator-endpoint/alerts", json=payload)
        print(f"[STUB_NOTIFY] Notify operator: {payload}")

    def _dispatch_uav_stub(self, pid, detection):
        # TODO: INTEGRATION_POINT_UAV_COMMAND
        # Replace with API calls to your UAV orchestrator to dispatch nearest UAV/vehicle.
        print(f"[STUB_UAV] Dispatch UAV to id={pid} bbox={detection.get('bbox')}")

# --------------------------
# Example perception integration & loop
# --------------------------
def perception_emulator_once():
    """
    This emulator returns a detection dict to simulate your model output.
    Replace this whole function in your real integration:
      INTEGRATION_POINT_MODEL_OUTPUT
    with your model inference pipeline which returns the same dict schema.
    """
    # EXAMPLE: detection from model/tracker for one frame
    example_detection = {
        "id": 42,
        "bbox": [100, 66, 80, 200],
        "person_conf": 0.95,
        "gun_conf": 0.02,
        "knife_conf": 0.01,
        "fight_conf": 0.0,
        "meta": {"running": False, "loitering": False},
        "timestamp": time.time(),
        # "frame": <numpy array frame optional>,
    }
    return example_detection

# --------------------------
# If run as script, demo usage
# --------------------------
if __name__ == "__main__":
    engine = DecisionEngine()
    # Simulate a stream of frames with varying confidences
    # In your real integration, call engine.process(detection) each frame
    # and feed detection dicts produced by your detection+tracker pipeline.
    print("=== Decision Engine Demo ===")
    # Simple scenario: normal person first, then knife appears, then fight.
    frames = []
    # 1. Normal person frames
    for _ in range(5):
        d = {
            "id": 1,
            "bbox": [10, 20, 60, 120],
            "person_conf": 0.9,
            "gun_conf": 0.0,
            "knife_conf": 0.0,
            "fight_conf": 0.0,
            "meta": {"running": False},
            "timestamp": time.time()
        }
        frames.append(d)

    # 2. Knife drawn frames
    for _ in range(12):
        d = {
            "id": 1,
            "bbox": [10, 20, 60, 120],
            "person_conf": 0.9,
            "gun_conf": 0.0,
            "knife_conf": 0.6,   # above knife threshold in config
            "fight_conf": 0.0,
            "meta": {"running": False},
            "timestamp": time.time()
        }
        frames.append(d)

    # 3. Fight frames
    for _ in range(6):
        d = {
            "id": 1,
            "bbox": [10, 20, 60, 120],
            "person_conf": 0.9,
            "gun_conf": 0.0,
            "knife_conf": 0.1,
            "fight_conf": 0.8,   # fight detected strongly
            "meta": {"running": False},
            "timestamp": time.time()
        }
        frames.append(d)

    # Run through frames
    for i, f in enumerate(frames):
        out = engine.process(f)
        print(f"Frame {i:03d} -> state={out['state']} score={out['threat_score']} action={out['action']} reason={out['reason']}")
        time.sleep(0.02)  # simulate frame interval
