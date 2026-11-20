#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 nabe2030
# SPDX-License-Identifier: Apache-2.0
#
# av2_stopgo_common.py —  の「進む/止まる」の分け方(機械で一貫して付ける基準)。
#
# 1. 軌跡での分け方(主): 出力の軌跡(0.1 s きざみ 64 点、自車の座標)を原点から 3.0 s(30 点目)までたどった
#    道のりが 2.0 m 以上なら「進む」、未満なら「止まる」。2.0 m / 3 s は平均 0.67 m/s(止まった状態から
#    0.44 m/s² で動き出したときの距離にあたる)。じわじわ前に出る(クリープ)は「止まる」に入る。
# 2. 判断理由の文での分け方(副): 文の頭の動詞句で分ける。
#    stop  : wait / remain (stopped) / stay / stop / yield / hold / decelerate (and|to) (come to a)? stop / decelerate to yield /
#            brake to (a) stop / keep stopped
#    go    : proceed / steer / turn / accelerate / advance / continue / resume / go / move / merge / change lanes / nudge / keep /
#            "the traffic light has turned green, so proceed" のように理由が先で動詞があとの文は、
#            ", so " のあとの動詞句で分ける
#    slow  : 上の stop 以外の decelerate / slow(前の車との間をあける など)
#    (例外)continue/keep slowing・decelerating・to stop など: 最初の節に stop/wait/yield があれば stop、なければ slow
#    creep : creep / inch(ゆっくり前に出る。止まる・進むのどちらにも入れず、別に数える)
#    nominal: <NOMINAL>(判断理由を出さない)
#    other : どれにも当たらない
import math
import re

GO_M, GO_T = 2.0, 30  # 3.0 s で 2.0 m


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
    m2 = _CONT_SLOW.match(s)  # "continue slowing to stop" などは continue でも go にしない
    if m2:
        return "stop" if re.search(r"\bstop\b|\bwait\b|\byield\b", s.split(",")[0]) else "slow"
    if _STOP.match(s):
        return "stop"
    if re.match(r"^(creep|inch)\b", s):  # ゆっくり前に出る(止まる・進むのどちらでもないので分ける)
        return "creep"
    if _GO.match(s):
        return "go"
    if _SLOW.match(s):
        return "slow"
    return "other"
