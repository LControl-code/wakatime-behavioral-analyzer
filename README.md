# WakaTime Behavioral Pattern Analyzer

A research tool for analyzing developer behavioral patterns in time-tracking systems. This tool helps security researchers and developers understand how coding patterns manifest in time-tracking data and can be used to improve detection algorithms.

## Purpose

This tool is designed for:

- Security research on time-tracking systems
- Understanding developer productivity patterns
- Improving anomaly detection algorithms
- Analyzing behavioral patterns in coding workflows

## Features

- **Realistic Behavioral Simulation**: Models actual developer patterns including:
  - Circadian rhythm-based productivity cycles
  - Day-of-week variations
  - Fatigue modeling over extended sessions
  - Flow states and difficulty periods
- **Comprehensive Metrics**: Tracks and reports:

  - Heartbeat frequency patterns
  - File access patterns
  - Behavioral anomalies
  - Session statistics

- **Configurable Environment**: Full control over:
  - Project structure
  - Session parameters
  - Behavioral modeling features
  - Output formats

## Installation

1. Clone the repository
2. Ensure Python 3.7+ is installed
3. Install WakaTime CLI
4. Configure your environment

## Configuration

### 1. Create `config.ini`:

```ini
[wakatime]
cli_path = /path/to/wakatime-cli
config_path = /path/to/.wakatime.cfg
log_path = /path/to/wakatime.log

[project]
name = your_project_name
path = /path/to/project
files_config = project_files.json

[simulation]
default_duration_minutes = 60
log_file = behavioral_analysis.log
enable_mood_states = true
enable_fatigue_modeling = true
```

### 2. Define Project Structure in `project_files.json`:

```json
{
  "main.py": { "lines": 250, "language": "Python" },
  "utils.py": { "lines": 150, "language": "Python" },
  "README.md": { "lines": 100, "language": "Markdown" }
}
```

## Usage

### Basic Usage

```bash
# Run 60-minute analysis with default config
python behavioral_analyzer.py

# Run 2-hour analysis
python behavioral_analyzer.py --duration 120

# Use custom configuration
python behavioral_analyzer.py --config custom.ini --duration 90

# Test connection
python behavioral_analyzer.py --test-connection

# Enable verbose logging
python behavioral_analyzer.py --verbose --duration 60
```

### Command Line Options

- `--config, -c`: Path to configuration file (default: config.ini)
- `--duration, -d`: Simulation duration in minutes (default: 60)
- `--test-connection, -t`: Test WakaTime connection and exit
- `--verbose, -v`: Enable verbose debug logging

## Output

The analyzer generates:

1. **Real-time Logs**: Behavioral events and state changes
2. **Session Report**: Summary statistics and patterns
3. **Metrics File**: Detailed JSON export of all session data

### Example Session Report:

```
============================================================
BEHAVIORAL ANALYSIS REPORT
============================================================
Session Duration: 60.2 minutes
Total Heartbeats: 342
Files Accessed: 8
Lines Modified: 156
Behavioral Events: 12

Average Heartbeat Interval: 10.6 seconds
Heartbeats per Minute: 5.68

Files Accessed:
  - main.py
  - utils.py
  - test_suite.py
  - README.md

Behavioral Events:
  - context_switch at 2024-06-18T10:15:32
  - debugging_pattern at 2024-06-18T10:23:45
  - micro_break at 2024-06-18T10:41:18
============================================================
```

## Research Applications

### 1. Detection Algorithm Development

Use the generated patterns to:

- Train anomaly detection models
- Identify behavioral signatures
- Validate detection thresholds

### 2. Productivity Analysis

Study how different factors affect coding patterns:

- Time of day impacts
- Fatigue accumulation
- Context switching costs

### 3. Security Research

Understand:

- Normal vs. anomalous patterns
- Detection evasion techniques
- Behavioral fingerprinting

## Behavioral Models

### Circadian Productivity Model

The tool models productivity based on research-backed circadian rhythms:

- Peak hours: 10-11 AM, 3-4 PM
- Low periods: Early morning, post-lunch, late evening

### Fatigue Accumulation

Performance degradation over time:

- 0-2 hours: 95% capacity
- 2-4 hours: 85% capacity
- 4-6 hours: 70% capacity
- 6+ hours: 40-60% capacity with high variance

### Behavioral States

- **Flow State**: Increased typing, less context switching
- **Difficulty Mode**: More navigation, longer pauses
- **Fatigue State**: Slower actions, more mistakes

## Ethical Considerations

This tool is designed for legitimate research purposes:

- Understanding developer behavior
- Improving time-tracking accuracy
- Developing better detection systems

Users are responsible for complying with all applicable laws and policies.

## Contributing

Contributions are welcome! Areas of interest:

- Additional behavioral models
- Integration with other time-tracking systems
- Machine learning pattern analysis
- Visualization tools

## License

MIT

## Disclaimer

This tool is for research and educational purposes. Users must comply with all relevant terms of service and applicable laws when using this software.
