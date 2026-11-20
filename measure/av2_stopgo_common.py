#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_stopgo_common.py — the "go/stop" classification (a criterion applied mechanically and consistently).
#
# 1. Classification by trajectory (primary): follow the output trajectory (64 points at 0.1 s intervals, ego coordinates) from the origin up to 3.0 s (the 30th point);
#    if the distance traveled is 2.0 m or more it is "go", otherwise "stop". 2.0 m / 3 s is an average of 0.67 m/s (corresponds to the distance
#    when starting from standstill at 0.44 m/s²). Edging forward slowly (creep) counts as "stop".
# 2. Classification by reasoning text (secondary): classified by the verb phrase at the head of the text.
#    stop  : wait / remain (stopped) / stay / stop / yield / hold / decelerate (and|to) (come to a)? stop / decelerate to yield /
#            brake to (a) stop / keep stopped
#    go    : proceed / steer / turn / accelerate / advance / continue / resume / go / move / merge / change lanes / nudge / keep /
#            sentences where the reason comes first and the verb after, like "the traffic light has turned green, so proceed",
#            are classified by the verb phrase after ", so "
#    slow  : decelerate / slow other than the stop cases above (e.g. leave a gap to the vehicle ahead)
#    (exception) continue/keep slowing, decelerating, to stop, etc.: stop if the first clause contains stop/wait/yield, otherwise slow
#    creep : creep / inch (edging forward slowly; counted separately, in neither stop nor go)
#    nominal: <NOMINAL> (no reasoning output)
#    other : matches none of these
import math
import re

GO_M, GO_T = 2.0, 30  # 2.0 m in 3.0 s


def travel(xyz, n=GO_T):
    p = [(0.0, 0.0)] + [tuple(q[:2]) for q in xyz[:n]]
    return sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1))


def traj_class(xyz):
    return "go" if travel(xyz) >= GO_M else "stop"


_STOP = re.compile(r"^(wait|remain|stay|stop|yield|hold|keep (stopped|still|waiting|holding)|come to a stop|decelerate (and|to) (come to a |a )?(full )?stop|"
                   r"decelerate (to|and) yield|slow (down )?(to|and) (a )?stop|brake (to|and) (come to )?(a )?(full )?stop)\b")
_GO = re.compile(r"^(proceed|steer|turn|accelerate|advance|continue|resume|go|move|merge|change lanes?|nudge|keep|drive|enter|"
                 r"pass|overtake|follow|maintain (our |the )?(current )?speed|make (a|the)|complete|start|begin|"
                 r"pull|navigate|cross|bear|veer|exit|take)\b")
_CONT_SLOW = re.compile(r"^(continue|keep) (slowing|decelerating|braking|to (slow|decelerate|brake|stop|wait|yield|hold|remain))\b")
_SLOW = re.compile(r"^(decelerate|slow|brake|reduce)\b")


def cot_class(cot: str) -> str:
    c = cot.strip()
    if c == "<NOMINAL>":
        return "nominal"
    s = c.lower()
    m = re.search(r",\s*so\s+(.*)$", s)
    if m and not (_STOP.match(s) or _GO.match(s) or _SLOW.match(s)):
        s = m.group(1)
    m2 = _CONT_SLOW.match(s)  # "continue slowing to stop" etc. is not go even though it starts with continue
    if m2:
        return "stop" if re.search(r"\bstop\b|\bwait\b|\byield\b", s.split(",")[0]) else "slow"
    if _STOP.match(s):
        return "stop"
    if re.match(r"^(creep|inch)\b", s):  # edging forward slowly (neither stop nor go, so kept separate)
        return "creep"
    if _GO.match(s):
        return "go"
    if _SLOW.match(s):
        return "slow"
    return "other"
