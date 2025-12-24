#!/bin/sh

# Stop ui_and_ai (aligned with ai_demo style)
if pgrep -x ui_and_ai >/dev/null; then
    pkill ui_and_ai
    echo "ui_and_ai stopped"
else
    echo "ui_and_ai not running"
fi