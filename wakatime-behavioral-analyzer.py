#!/usr/bin/env python3
"""
WakaTime Behavioral Pattern Analyzer
A research tool for studying developer behavior patterns and time-tracking accuracy.

Author: Security Research Team
Purpose: Understanding and improving time-tracking detection systems
"""

import subprocess
import time
import random
import logging
import sys
import io
import json
import os
import argparse
import configparser
from datetime import datetime
from typing import Dict, List, Optional

# Handle Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class BehavioralPatternAnalyzer:
    """
    Analyzes and simulates developer behavioral patterns for time-tracking research.

    This tool helps understand how different coding patterns appear in time-tracking
    systems and can be used to improve detection algorithms.
    """

    def __init__(self, config_path: str = "config.ini"):
        """
        Initialize the analyzer with configuration settings.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_configuration(config_path)
        self._setup_logging()
        self._initialize_project_structure()
        self._initialize_behavioral_state()

    def _load_configuration(self, config_path: str) -> dict:
        """Load configuration from INI file or use defaults."""
        config = configparser.ConfigParser()

        # Default configuration
        defaults = {
            "wakatime": {
                "cli_path": rf"C:\Users\{os.getenv('USERNAME')}\.wakatime\wakatime-cli-windows-amd64.exe",
                "config_path": rf"C:\Users\{os.getenv('USERNAME')}\.wakatime.cfg",
                "log_path": rf"C:\Users\{os.getenv('USERNAME')}\.wakatime\wakatime.log",
            },
            "project": {
                "name": "behavioral_analysis_test",
                "path": r"C:\Development\test_project",
                "files_config": "project_files.json",
            },
            "simulation": {
                "default_duration_minutes": "60",
                "log_file": "behavioral_analysis.log",
                "enable_mood_states": "true",
                "enable_fatigue_modeling": "true",
            },
        }

        if os.path.exists(config_path):
            config.read(config_path)
            logging.info(f"Configuration loaded from {config_path}")
        else:
            # Create default config file
            for section, values in defaults.items():
                config[section] = values
            with open(config_path, "w") as f:
                config.write(f)
            logging.info(f"Created default configuration at {config_path}")

        return {s: dict(config.items(s)) for s in config.sections()}

    def _setup_logging(self):
        """Configure logging for analysis output."""
        log_file = self.config["simulation"]["log_file"]

        # Create formatter for research-style logging
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logging.root.setLevel(logging.INFO)
        logging.root.handlers = [file_handler, console_handler]

    def _initialize_project_structure(self):
        """Load project file structure from configuration."""
        files_config_path = self.config["project"]["files_config"]

        if os.path.exists(files_config_path):
            with open(files_config_path, "r") as f:
                self.project_structure = json.load(f)
        else:
            # Default project structure for testing
            self.project_structure = {
                "main.py": {"lines": 150, "language": "Python"},
                "utils.py": {"lines": 75, "language": "Python"},
                "README.md": {"lines": 100, "language": "Markdown"},
                "config.json": {"lines": 30, "language": "JSON"},
                "test_suite.py": {"lines": 200, "language": "Python"},
            }

            # Save default structure
            with open(files_config_path, "w") as f:
                json.dump(self.project_structure, f, indent=2)
            logging.info(f"Created default project structure at {files_config_path}")

        # Initialize navigation state
        self.current_file = list(self.project_structure.keys())[0]
        self.current_line = 1
        self.cursor_position = 1

    def _initialize_behavioral_state(self):
        """Initialize behavioral simulation state variables."""
        self.session_metrics = {
            "start_time": None,
            "total_heartbeats": 0,
            "files_accessed": set(),
            "lines_modified": 0,
            "behavioral_events": [],
        }

        self.heartbeat_cache = {
            "last_sent": 0,
            "last_file": None,
            "last_line": 0,
            "last_cursor": 0,
        }

        # Behavioral state flags
        self.behavioral_states = {
            "productivity_level": 1.0,
            "focus_score": 1.0,
            "energy_level": 1.0,
            "in_flow_state": False,
            "experiencing_difficulty": False,
        }

        # Environmental awareness
        self.environmental_context = {}
        self.last_mood_evaluation = None

    def analyze_temporal_patterns(self) -> Dict[str, float]:
        """
        Analyze current temporal context for behavioral modeling.

        Returns:
            Dictionary of temporal factors affecting behavior
        """
        current_time = datetime.now()
        hour = current_time.hour
        weekday = current_time.weekday()

        # Research-based productivity patterns
        temporal_factors = {
            "circadian_productivity": self._calculate_circadian_factor(hour),
            "weekly_rhythm": self._calculate_weekly_factor(weekday),
            "post_meal_effect": self._calculate_meal_impact(hour),
            "end_of_day_fatigue": self._calculate_eod_factor(hour),
        }

        logging.debug(f"Temporal analysis: {temporal_factors}")
        return temporal_factors

    def _calculate_circadian_factor(self, hour: int) -> float:
        """Calculate productivity based on circadian rhythms."""
        # Based on research on developer productivity patterns
        productivity_curve = {
            6: 0.5,
            7: 0.6,
            8: 0.7,
            9: 0.85,
            10: 0.95,
            11: 1.0,
            12: 0.9,
            13: 0.7,
            14: 0.75,
            15: 0.85,
            16: 0.9,
            17: 0.85,
            18: 0.7,
            19: 0.6,
            20: 0.5,
            21: 0.4,
            22: 0.3,
            23: 0.2,
        }
        return productivity_curve.get(hour, 0.3)

    def _calculate_weekly_factor(self, weekday: int) -> float:
        """Calculate productivity based on day of week."""
        # Monday=0, Sunday=6
        weekly_pattern = [0.7, 0.9, 1.0, 0.95, 0.75, 0.4, 0.3]
        return weekly_pattern[weekday]

    def _calculate_meal_impact(self, hour: int) -> float:
        """Calculate post-meal productivity dip."""
        if 13 <= hour <= 14:
            return 0.7  # Post-lunch dip
        return 1.0

    def _calculate_eod_factor(self, hour: int) -> float:
        """Calculate end-of-day fatigue factor."""
        if hour >= 20:
            return 0.6
        elif hour >= 17:
            return 0.8
        return 1.0

    def generate_behavioral_weights(self) -> List[int]:
        """
        Generate action probability weights based on current behavioral state.

        Returns:
            List of weights for [typing, progression, navigation, file_switch, pause, long_pause]
        """
        base_weights = [35, 20, 15, 10, 15, 5]

        # Apply behavioral modifiers
        if self.behavioral_states["in_flow_state"]:
            logging.debug("Flow state detected - adjusting weights for focused work")
            base_weights[0] = int(base_weights[0] * 1.5)  # More typing
            base_weights[1] = int(base_weights[1] * 1.3)  # More progression
            base_weights[3] = int(base_weights[3] * 0.5)  # Less file switching

        if self.behavioral_states["experiencing_difficulty"]:
            logging.debug(
                "Difficulty state detected - adjusting for problem-solving behavior"
            )
            base_weights[2] = int(base_weights[2] * 1.5)  # More navigation
            base_weights[4] = int(base_weights[4] * 1.5)  # More pauses
            base_weights[5] = int(base_weights[5] * 2.0)  # More long pauses

        # Apply temporal factors
        temporal = self.analyze_temporal_patterns()
        productivity_modifier = (
            temporal["circadian_productivity"] * temporal["weekly_rhythm"]
        )

        if productivity_modifier < 0.5:
            base_weights[4] = int(base_weights[4] * 2.0)  # More pauses when tired
            base_weights[5] = int(base_weights[5] * 2.5)  # More long pauses

        # Add controlled randomness for realistic variation
        return [max(1, int(w * random.uniform(0.8, 1.2))) for w in base_weights]

    def simulate_developer_action(self, action_type: str):
        """
        Simulate a specific developer action with realistic patterns.

        Args:
            action_type: Type of action to simulate
        """
        action_handlers = {
            "typing": self._simulate_typing_pattern,
            "progression": self._simulate_line_progression,
            "navigation": self._simulate_code_navigation,
            "file_switch": self._simulate_file_switching,
            "pause": self._simulate_short_pause,
            "long_pause": self._simulate_extended_pause,
        }

        handler = action_handlers.get(action_type)
        if handler:
            handler()
        else:
            logging.warning(f"Unknown action type: {action_type}")

    def _simulate_typing_pattern(self):
        """Simulate realistic typing patterns with micro-behaviors."""
        burst_length = random.randint(3, 8)

        # Adjust burst length based on energy
        energy = self.behavioral_states.get("energy_level", 1.0)
        burst_length = max(1, int(burst_length * energy))

        for i in range(burst_length):
            # Mid-burst interruption check for low focus
            focus = self.behavioral_states.get("focus_score", 1.0)
            if random.random() < 0.005 * (1 - focus):
                logging.info("Lost concentration mid-typing")
                time.sleep(random.uniform(3, 10))
                return  # Abandon the burst

            # Simulate character progression
            self.cursor_position = min(
                self.cursor_position + random.randint(5, 15), 120
            )

            # Occasional backtracking (corrections) - more when tired
            if random.random() < 0.1 / energy:
                self.cursor_position = max(
                    1, self.cursor_position - random.randint(2, 8)
                )

            self._send_behavioral_heartbeat(is_write=True)

            # Realistic inter-keystroke timing
            base_delay = random.uniform(0.1, 0.5)

            # Adjust for behavioral states (same as original)
            if self.behavioral_states.get("in_flow_state", False):
                actual_delay = base_delay * 0.7
            elif self.behavioral_states.get("experiencing_difficulty", False):
                actual_delay = base_delay * (3.0 - energy)
            else:
                actual_delay = base_delay * (2.0 - energy)

            time.sleep(actual_delay)

    def _simulate_line_progression(self):
        """Simulate moving through code with purpose."""
        file_info = self.project_structure.get(self.current_file, {})
        max_lines = file_info.get("lines", 100)

        if self.current_line < max_lines - 10:
            # Forward progression with occasional jumps
            if random.random() < 0.3:
                jump = random.randint(5, 15)  # Skipping sections
            else:
                jump = random.randint(1, 3)  # Line by line

            self.current_line = min(self.current_line + jump, max_lines)
            self.cursor_position = random.randint(1, 20)  # Typical indentation

            self._send_behavioral_heartbeat(is_write=True)

    def _simulate_code_navigation(self):
        """Simulate navigating through code structure."""
        file_info = self.project_structure.get(self.current_file, {})
        max_lines = file_info.get("lines", 100)

        # Different navigation patterns
        navigation_type = random.choice(["search", "jump_to_definition", "scroll"])

        if navigation_type == "search":
            # Random jumps (searching for something)
            self.current_line = random.randint(1, max_lines)
        elif navigation_type == "jump_to_definition":
            # Common code structure points
            structure_points = [1, 25, 50, 100, 150, 200]
            valid_points = [p for p in structure_points if p <= max_lines]
            if valid_points:
                self.current_line = random.choice(valid_points)
        else:
            # Scrolling up/down
            delta = random.randint(-30, 30)
            self.current_line = max(1, min(self.current_line + delta, max_lines))

        self.cursor_position = random.randint(1, 80)
        self._send_behavioral_heartbeat()

    def _simulate_file_switching(self):
        """Simulate switching between project files."""
        available_files = list(self.project_structure.keys())

        # Remove current file from options
        if self.current_file in available_files:
            available_files.remove(self.current_file)

        if available_files:
            # Prefer recently accessed files (realistic workflow)
            if self.session_metrics["files_accessed"] and random.random() < 0.4:
                recent_files = list(self.session_metrics["files_accessed"])
                recent_files = [f for f in recent_files if f in available_files]
                if recent_files:
                    self.current_file = random.choice(recent_files)
                else:
                    self.current_file = random.choice(available_files)
            else:
                self.current_file = random.choice(available_files)

            # Reset position in new file
            file_info = self.project_structure.get(self.current_file, {})
            self.current_line = random.randint(1, min(50, file_info.get("lines", 100)))
            self.cursor_position = 1

            logging.info(f"Context switch → {self.current_file}")
            self._send_behavioral_heartbeat()

    def _simulate_short_pause(self):
        """Simulate short thinking/reading pauses."""
        pause_duration = random.uniform(5, 30)

        # Adjust based on behavioral state (same as original)
        if self.behavioral_states.get("experiencing_difficulty", False):
            pause_duration *= 2.5
        elif self.behavioral_states.get("energy_level", 1.0) < 0.5:
            pause_duration *= 2.0 - self.behavioral_states["energy_level"]

        logging.info(f"Short pause: {pause_duration:.1f}s (thinking/reading)")
        time.sleep(pause_duration)

        # Sometimes lose focus during pause
        if random.random() < (1 - self.behavioral_states.get("focus_score", 1.0)) * 0.3:
            self.inject_behavioral_anomaly()

        # Send heartbeat to maintain presence
        self._send_behavioral_heartbeat()

    def _simulate_extended_pause(self):
        """Simulate extended breaks or deep thinking."""
        pause_duration = random.uniform(90, 180)

        # Exhaustion makes everything take longer (from original)
        energy = self.behavioral_states.get("energy_level", 1.0)
        if energy < 0.7:
            pause_duration *= 2.0 - energy

        logging.info(f"Extended pause: {pause_duration:.1f}s (deep focus/break)")
        time.sleep(pause_duration)

        # Force heartbeat after long pause
        self._send_behavioral_heartbeat(force=True)

    def _should_send_heartbeat(self, is_write: bool = False) -> bool:
        """
        Determine if a heartbeat should be sent based on activity patterns.

        Args:
            is_write: Whether this is a write operation

        Returns:
            Boolean indicating if heartbeat should be sent
        """
        if is_write:
            return True

        current_time = time.time()
        time_elapsed = current_time - self.heartbeat_cache["last_sent"]

        # Dynamic threshold based on activity
        threshold = 120  # Base 2-minute threshold

        if self.behavioral_states["in_flow_state"]:
            threshold = 90  # More frequent when focused
        elif self.behavioral_states["energy_level"] < 0.5:
            threshold = random.randint(120, 180)  # Irregular when tired

        # Send if enough time passed or significant change
        if time_elapsed >= threshold:
            return True

        # Send if file changed
        if self.current_file != self.heartbeat_cache["last_file"]:
            return True

        # Send if significant navigation
        line_diff = abs(self.current_line - self.heartbeat_cache["last_line"])
        if line_diff > 10:
            return True

        return False

    def _send_behavioral_heartbeat(self, is_write: bool = False, force: bool = False):
        """
        Send heartbeat with current activity state.

        Args:
            is_write: Whether this represents a write operation
            force: Force sending regardless of throttling
        """
        if not force and not self._should_send_heartbeat(is_write):
            return True

        # Prepare heartbeat data
        file_info = self.project_structure.get(self.current_file, {})
        entity_path = os.path.join(self.config["project"]["path"], self.current_file)

        # Read API credentials
        api_key = self._get_api_credentials()
        if not api_key:
            logging.error("API credentials not found - check configuration")
            return False

        # Build command
        cmd = [
            self.config["wakatime"]["cli_path"],
            "--entity",
            entity_path,
            "--plugin",
            "vscode/1.101.0 vscode-wakatime/25.0.5",  # behavioral-analyzer/1.0.0 for ease of detection
            "--lineno",
            str(self.current_line),
            "--cursorpos",
            str(self.cursor_position),
            "--lines-in-file",
            str(file_info.get("lines", 100)),
            "--key",
            api_key,
            "--alternate-project",
            self.config["project"]["name"],
            "--project-folder",
            self.config["project"]["path"],
            "--config",
            self.config["wakatime"]["config_path"],
            "--log-file",
            self.config["wakatime"]["log_path"],
        ]

        if file_info.get("language"):
            cmd.extend(["--language", file_info["language"]])

        if is_write:
            cmd.append("--write")

        # Execute heartbeat
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                action = "WRITE" if is_write else "READ"
                logging.info(
                    f"Heartbeat sent: {self.current_file}:{self.current_line} [{action}]"
                )

                # Update cache
                self.heartbeat_cache.update(
                    {
                        "last_sent": time.time(),
                        "last_file": self.current_file,
                        "last_line": self.current_line,
                        "last_cursor": self.cursor_position,
                    }
                )

                # Update metrics
                self.session_metrics["total_heartbeats"] += 1
                self.session_metrics["files_accessed"].add(self.current_file)
                if is_write:
                    self.session_metrics["lines_modified"] += 1

                return True
            else:
                logging.error(f"Heartbeat failed: {result.stderr}")
                return False

        except Exception as e:
            logging.error(f"Heartbeat error: {str(e)}")
            return False

    def _get_api_credentials(self) -> Optional[str]:
        """Retrieve API key from configuration file."""
        config_path = self.config["wakatime"]["config_path"]

        try:
            config = configparser.ConfigParser()
            with open(config_path, "r", encoding="utf-8-sig") as f:
                config.read_string(f.read())

            return config.get("settings", "api_key", fallback=None)
        except Exception as e:
            logging.error(f"Failed to read API credentials: {e}")
            return None

    def inject_behavioral_anomaly(self):
        """
        Inject realistic behavioral anomalies for pattern research.

        These represent common developer interruptions and behaviors.
        """
        if random.random() < 0.02:  # 2% chance per cycle
            anomaly_type = random.choices(
                ["context_switch", "debugging_pattern", "distraction", "micro_break"],
                weights=[25, 30, 25, 20],
            )[0]

            if anomaly_type == "context_switch":
                logging.info("Behavioral anomaly: Context switch detected")
                # Rapid file switching pattern
                for _ in range(random.randint(3, 5)):
                    self._simulate_file_switching()
                    time.sleep(random.uniform(2, 5))

            elif anomaly_type == "debugging_pattern":
                logging.info("Behavioral anomaly: Debugging pattern")
                # Repeated navigation in same file
                for _ in range(random.randint(5, 10)):
                    self._simulate_code_navigation()
                    time.sleep(random.uniform(1, 3))

            elif anomaly_type == "distraction":
                logging.info("Behavioral anomaly: Distraction event")
                # Pause with no activity
                time.sleep(random.uniform(60, 300))

            elif anomaly_type == "micro_break":
                logging.info("Behavioral anomaly: Micro break")
                time.sleep(random.uniform(180, 420))

            self.session_metrics["behavioral_events"].append(
                {"type": anomaly_type, "timestamp": datetime.now().isoformat()}
            )

    def update_behavioral_state(self, elapsed_hours: float):
        """
        Update behavioral state based on session duration.

        Args:
            elapsed_hours: Hours elapsed since session start
        """
        # Energy decay over time
        if elapsed_hours < 2:
            self.behavioral_states["energy_level"] = 0.9
        elif elapsed_hours < 4:
            self.behavioral_states["energy_level"] = 0.75
        elif elapsed_hours < 6:
            self.behavioral_states["energy_level"] = 0.6
        else:
            self.behavioral_states["energy_level"] = 0.4 + 0.2 * random.random()

        # Only roll for state changes once per hour
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        if (
            not hasattr(self, "last_state_evaluation_hour")
            or self.last_state_evaluation_hour != current_hour
        ):
            self.last_state_evaluation_hour = current_hour

            # Periodic state changes
            if random.random() < 0.1:  # 10% chance per hour
                if (
                    not self.behavioral_states["in_flow_state"]
                    and random.random() < 0.3
                ):
                    logging.info("Behavioral state: Entering flow state")
                    self.behavioral_states["in_flow_state"] = True
                elif self.behavioral_states["in_flow_state"] and random.random() < 0.2:
                    logging.info("Behavioral state: Exiting flow state")
                    self.behavioral_states["in_flow_state"] = False

            # Difficulty states
            if (
                random.random() < 0.1
                and not self.behavioral_states["experiencing_difficulty"]
            ):
                logging.info("Behavioral state: Encountering difficulty")
                self.behavioral_states["experiencing_difficulty"] = True
            elif (
                self.behavioral_states["experiencing_difficulty"]
                and random.random() < 0.3
            ):
                logging.info("Behavioral state: Difficulty resolved")
                self.behavioral_states["experiencing_difficulty"] = False

        # Time-based recovery (outside hourly check)
        if elapsed_hours > 3 and self.behavioral_states["experiencing_difficulty"]:
            if random.random() < 0.1:  # 10% chance per loop after 3 hours
                logging.info(
                    "Behavioral state: Second wind - pushing through difficulty"
                )
                self.behavioral_states["experiencing_difficulty"] = False

    def run_behavioral_simulation(self, duration_minutes: int):
        """
        Run the behavioral pattern simulation.

        Args:
            duration_minutes: Duration of simulation in minutes
        """
        self.session_metrics["start_time"] = time.time()
        end_time = self.session_metrics["start_time"] + (duration_minutes * 60)

        logging.info("=" * 60)
        logging.info("BEHAVIORAL PATTERN ANALYSIS SESSION")
        logging.info(f"Duration: {duration_minutes} minutes")
        logging.info(f"Project: {self.config['project']['name']}")
        logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 60)

        # Initial heartbeat
        self._send_behavioral_heartbeat(force=True)

        # Main simulation loop
        while time.time() < end_time:
            elapsed_hours = (time.time() - self.session_metrics["start_time"]) / 3600

            # Update behavioral state
            self.update_behavioral_state(elapsed_hours)

            # Check for behavioral anomalies
            self.inject_behavioral_anomaly()

            # Generate action weights based on current state
            action_weights = self.generate_behavioral_weights()

            # Difficulty adds extra delay before every action (from original)
            if self.behavioral_states.get("experiencing_difficulty", False):
                time.sleep(random.uniform(1, 3))

            # Select and execute action
            actions = [
                "typing",
                "progression",
                "navigation",
                "file_switch",
                "pause",
                "long_pause",
            ]
            selected_action = random.choices(actions, weights=action_weights)[0]

            logging.debug(f"Executing action: {selected_action}")
            self.simulate_developer_action(selected_action)

            # Inter-action delay with behavioral adjustments (from original)
            base_delay = random.uniform(2, 8)
            energy = self.behavioral_states.get("energy_level", 1.0)

            if self.behavioral_states.get("in_flow_state", False):
                actual_delay = base_delay * 0.5  # Faster when in flow
            elif self.behavioral_states.get("experiencing_difficulty", False):
                actual_delay = base_delay * 3.0  # Much slower on bad days
            else:
                actual_delay = base_delay * (2.0 - energy)

            time.sleep(actual_delay)

        # Session summary
        self._generate_session_report()

    def _generate_session_report(self):
        """Generate analysis report for the behavioral session."""
        elapsed = time.time() - self.session_metrics["start_time"]

        report = f"""
{"=" * 60}
BEHAVIORAL ANALYSIS REPORT
{"=" * 60}
Session Duration: {elapsed / 60:.1f} minutes
Total Heartbeats: {self.session_metrics["total_heartbeats"]}
Files Accessed: {len(self.session_metrics["files_accessed"])}
Lines Modified: {self.session_metrics["lines_modified"]}
Behavioral Events: {len(self.session_metrics["behavioral_events"])}

Average Heartbeat Interval: {elapsed / max(1, self.session_metrics["total_heartbeats"]):.1f} seconds
Heartbeats per Minute: {self.session_metrics["total_heartbeats"] / (elapsed / 60):.2f}

Files Accessed:
{chr(10).join(f"  - {f}" for f in self.session_metrics["files_accessed"])}

Behavioral Events:
{chr(10).join(f"  - {e['type']} at {e['timestamp']}" for e in self.session_metrics["behavioral_events"])}
{"=" * 60}
"""
        logging.info(report)

        # Save detailed metrics
        metrics_file = (
            f"session_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "config": self.config,
                    "metrics": self.session_metrics,
                    "final_state": self.behavioral_states,
                },
                f,
                indent=2,
                default=str,
            )

        logging.info(f"Detailed metrics saved to: {metrics_file}")


def main():
    """Main entry point for the behavioral analysis tool."""
    parser = argparse.ArgumentParser(
        description="WakaTime Behavioral Pattern Analyzer - Research Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duration 60                    # Run 60-minute analysis
  %(prog)s --config custom.ini --duration 120  # Use custom config
  %(prog)s --test-connection                # Test WakaTime connection
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config.ini",
        help="Path to configuration file (default: config.ini)",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=60,
        help="Simulation duration in minutes (default: 60)",
    )

    parser.add_argument(
        "--test-connection",
        "-t",
        action="store_true",
        help="Test WakaTime connection and exit",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.root.setLevel(logging.DEBUG)

    # Initialize analyzer
    try:
        analyzer = BehavioralPatternAnalyzer(args.config)
    except Exception as e:
        logging.error(f"Failed to initialize analyzer: {e}")
        sys.exit(1)

    # Test connection if requested
    if args.test_connection:
        logging.info("Testing WakaTime connection...")
        if analyzer._send_behavioral_heartbeat(force=True):
            logging.info("✓ Connection test successful")
            sys.exit(0)
        else:
            logging.error("✗ Connection test failed")
            sys.exit(1)

    # Run simulation
    try:
        analyzer.run_behavioral_simulation(args.duration)
    except KeyboardInterrupt:
        logging.info("\nSimulation interrupted by user")
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
