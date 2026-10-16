#!/usr/bin/env sh
set -e
BIN=${BIN:-./abyss}
PAR=${PAR:-3}
$BIN tune --preset conversation-70b --parallel "$PAR"
$BIN tune --preset toolagent-70b --parallel "$PAR"
$BIN tune --preset mixed-hetero --parallel "$PAR"
$BIN tune --preset agent-8b --parallel "$PAR"
