# ZÈRTZ – Rules of Play

*A game by Kris Burm, part of the GIPF Project*  
GIPF, TAMSK, ZÈRTZ, DVONN, YINSH and PÜNCT ® & © Don & Co NV. Content © Kris Burm.

---

## Introduction

> “A game about making sacrifices! The third game of Project GIPF. For 2 players.”  
> The board shrinks as the game progresses and both players draw from the same pool of marbles. Master timing and placement to gain control and secure the winning combination.

---

## Components

- 6 white marbles  
- 8 grey marbles  
- 10 black marbles  
- 49 round board pieces (“rings”)

---

## Aim

Capture **one** of the following sets to win:

- 3 marbles of each color  
- 4 white marbles  
- 5 grey marbles  
- 6 black marbles

The first player to satisfy a set wins immediately.

---

## Preparation

In total there are 49 rings. The **basic game** uses only **37 rings**; keep the remaining 12 in reserve for larger-board play (see “Expanded Board”).

1. Assemble the 37 rings into a hexagonal board.  
2. Place all marbles beside the board as a shared **pool** that either player may draw from.  
3. Determine the starting player by lot.

---

## Making a Move

On your turn you must take **exactly one** of the two actions below:

1. **Place** a marble on the board and then **remove** a ring.  
2. **Capture** one or more marbles.

Play alternates until someone wins.

### Placing a Marble and Removing a Ring

1. Choose any marble from the pool and place it on any vacant ring. Marbles belong to both players; colors are unrestricted.  
2. After placing, you must remove a **free** ring. A ring is free if it is vacant, lies on the edge, and can be lifted away without disturbing adjacent rings.  
3. Placing and removing together constitute one turn. If no vacant edge ring can be removed safely, the placement stands and no ring is removed.  
4. Set removed rings aside—they are handy platforms for storing captured marbles.

### Capturing Marbles

1. Capturing is **compulsory**: if you can capture, you must.
2. Jump a marble over an adjacent marble into the vacant ring immediately beyond it (as in checkers). You may jump in any direction provided the landing ring is empty.
3. The color of the jumping and captured marbles is irrelevant.
4. If, after a jump, another capture is available with the same marble, you must continue until no further jump is possible.
5. When multiple capture routes exist (with different lengths), you may choose any of them.
6. **Important:** When multiple capture options are available (including captures created by a previous player's move), you may freely choose whichever capture you prefer. You are not required to capture the marble that was just moved by your opponent.
7. A capture sequence (one or more jumps) completes your turn; you do not place a marble or remove a ring afterward.

---

## Isolating Marbles

1. If your move disconnects one or more rings from the main board, you may claim those isolated rings and the marbles on them. Treat this as a second way to capture; it is not mandatory.
2. You may only claim an isolated group when **all rings in that group are occupied**. You either remove the ring that causes the isolation or place the final marble that fills a pre-isolated cluster. The capture happens as a consequence of the move that created the isolation.
3. **Important:** If you isolate a ring without a marble on it, that ring stays on the board. When you or your opponent later places a marble on it, that player removes both the ring and captures that marble. Alternatively, players can place marbles elsewhere and choose to remove the isolated ring as their required ring removal.

---

## End of the Game

The game ends immediately when a player holds one of the target sets listed under **Aim**. That player wins at once.

---

## Special Cases

1. If the pool is empty before anyone wins, continue playing using your **captured marbles**. As with the pool, you may choose any color from your captures on your turn.
2. If every ring becomes occupied before a win condition is met, the player who filled the final vacant ring wins by claiming the now-isolated board.
3. If both players repeat the same sequence of moves indefinitely (no matter what color marbles are used), the game is declared a tie.
4. **Tournament Rule:** If all marbles are placed on the board and the current player hasn't captured any marbles yet, this indicates prohibited player collaboration. In official tournament play, both players lose the game (ties are not permitted).

---

## Strategy Example

Consider the situation described in the official rules (Diagram 5): Player A, behind by five marbles, can still win. A places a black marble on ring 1 and removes ring 2, forcing B to capture but not to win. On the follow-up, A plays a white marble on ring 3, removes ring 4, and claims the isolated white marbles for the winning set of four whites. Precision timing and forced play are key themes throughout ZÈRTZ.

---

## Expanded Board

ZÈRTZ was originally released with just 37 rings—enough to learn the fundamentals. Once you are comfortable with sacrifices and forced sequences, you can enlarge the board by adding rings from GIPF Set 2 or the modern boxed edition.

- Adding a handful of rings (e.g., a 3-ring strip) creates an irregular hexagon and broadens the opening landscape without overwhelming complexity.  
- Larger additions (6–7 rings) offer a steeper challenge.  
- Tournament play traditionally uses **48 rings** (37 + 11).  
- With GIPF Set 2 you can add up to 24 extra rings, creating a 61-ring board—an ultimate, demanding configuration.

**Notes:**  
1. You do not need extra marbles; the supply and victory conditions stay the same.  
2. The 49th ring in the set is a spare piece, useful if you expand to 61 rings.

---

## Tournament Rules

1. Tournaments use a board of at least 48 rings (up to 61 when all extras are added).  
2. “Touch” rules apply: once you take a marble from the pool you must play it; once a marble touches a ring you must place it there; once you touch a removable edge ring you must remove it.  
3. Because capturing is compulsory, you may force an opponent to retract an illegal non-capturing move, including replacing any ring they removed, before play continues.

---

## Blitz Variant

The Blitz variant adapts the basic 37-ring board for a sharper, faster contest. Use one fewer marble of each color (5 white, 7 grey, 9 black). Victory now requires:

- 2 marbles of each color, or  
- 3 white marbles, or  
- 4 grey marbles, or  
- 5 black marbles.

Games are short, aggressive, and unforgiving—ideal once you know the fundamentals.

---

## Have Fun!

Enjoy exploring sacrifices, tempo, and control—the hallmarks of ZÈRTZ.

---

## Player Immobilization & Loop Detection (Engine Notes)

The official rulebook treats these situations implicitly; the Hiive implementation follows the conventions below for clarity:

- **Single-player immobilization:** If the active player has no legal move (no placements or captures), their turn is skipped automatically. Play resumes with the opponent.  
- **Mutual immobilization:** Two consecutive passes end the game immediately. Check the standard victory conditions; if neither player qualifies, the result is a tie.  
- **Loop detection:** If the last two move-pairs repeat the previous two move-pairs exactly (discounting marble color), the game is declared a tie to prevent infinite cycling.

---

## Summary of Key Principles

- All marbles are communal; color selection is strategic, not assigned.  
- Captures are mandatory whenever available.  
- The playable board shrinks over time as rings are removed.  
- Fully occupied isolated regions yield bonus captures when disconnected.  
- Successful play revolves around sacrifice, timing, and long forced sequences.

---

## Notation

The official notation system is documented at [http://www.gipf.com/zertz/notations/notation.html](http://www.gipf.com/zertz/notations/notation.html). The conventions below follow that reference.

### Marble Colors

- **W** = White  
- **G** = Grey  
- **B** = Black

### Placement Moves

Format: `[Color][Destination][,Removed Ring]`

- `Wd4` – place a white marble on d4, no ring removed  
- `Bd7,b2` – place a black marble on d7 and remove the ring at b2

### Capture Moves

Format: `x [Source][Captured Color][Destination]`

- `x e3Wg3` – capture from e3 to g3 by jumping over a white marble on f3  
- `x d1Gd3Wd5` – multiple jumps within a single capture sequence

### Isolation Captures

When a placement isolates occupied rings, list the captured marbles after an `x`.

- `Bd7,b2 x Wa1Wa2` – place black on d7, remove b2, claim the isolated white marbles on a1 and a2

### Pass

A pass is recorded as a single hyphen: `-`

### Move Sequence Example

```
Wd4,d1
Ge5,c3
Bf6,d2
x c2We4
-
```

1. White on d4, remove d1  
2. Grey on e5, remove c3  
3. Black on f6, remove d2  
4. Capture from c2 to e4, capturing the white marble on d3  
5. Pass
