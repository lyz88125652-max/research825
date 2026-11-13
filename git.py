#!/usr/bin/env python3
"""Run git add . / commit / push from this project's directory.

    python git.py
    python git.py -m "Update experiment pipeline"
    python git.py --dry-run
"""

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def git(*arguments, capture=False, check=True):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=check,
                          text=True, capture_output=capture)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--message", help="Commit message (default: current local date/time).")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without changing or pushing anything.")
    args = parser.parse_args(argv)
    message = args.message if args.message is not None else datetime.now().astimezone().strftime("Update %Y-%m-%d %H:%M:%S %z")
    if not message.strip():
        parser.error("Commit message must not be empty.")
    try:
        branch_result = git("symbolic-ref", "--quiet", "--short", "HEAD", capture=True, check=False)
        if branch_result.returncode:
            raise RuntimeError("ブランチが選択されていません。ブランチに切り替えてから実行してください。")
        branch = branch_result.stdout.strip()
        upstream = git("rev-parse", "--abbrev-ref", "@{upstream}", capture=True, check=False)
        if upstream.returncode == 0:
            remote = git("config", "--get", f"branch.{branch}.remote", capture=True).stdout.strip()
            target = git("config", "--get", f"branch.{branch}.merge", capture=True).stdout.strip()
            if remote == ".":
                raise RuntimeError("追跡先がローカルブランチです。リモートの追跡先を設定してください。")
            push = ["push", remote, f"HEAD:{target}"]
        else:
            remotes = git("remote", capture=True).stdout.splitlines()
            remote = "origin" if "origin" in remotes else remotes[0] if len(remotes) == 1 else None
            if remote is None:
                raise RuntimeError("送信先を決定できません。git remote add origin <URL> などで設定してください。")
            push = ["push", "--set-upstream", remote, f"HEAD:refs/heads/{branch}"]
        print(f"Project: {ROOT}\nBranch: {branch}", flush=True)
        if args.dry_run:
            for command in (["add", "."], ["commit", "-m", message], push):
                print(shlex.join(["git", *command]))
            return 0
        git("add", ".")
        diff = git("diff", "--cached", "--quiet", check=False)
        if diff.returncode == 1:
            git("commit", "-m", message)
        elif diff.returncode == 0:
            print("変更がないためcommitを省略し、pushを実行します。", flush=True)
        else:
            raise RuntimeError("ステージ済み変更の確認に失敗しました。")
        git(*push)
        print("完了しました。", flush=True)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"停止しました: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
