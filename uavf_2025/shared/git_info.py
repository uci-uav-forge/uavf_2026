import subprocess


def get_git_info():
    """Retrieve current Git branch and commit hash."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        return branch, commit_hash
    except subprocess.CalledProcessError:
        return "unknown", "unknown"


def get_git_diff():
    """Retrieve current Git diff."""
    try:
        diff = subprocess.check_output(["git", "diff"], text=True).strip()
        return diff
    except subprocess.CalledProcessError:
        return "unknown"
